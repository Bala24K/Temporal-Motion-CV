"use client";
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, X } from "lucide-react";
import { api } from "@/lib/api";
import { demoStage, drawSkeleton, startCamera, type PoseSource } from "@/lib/pose";
import type { Assessment, CalibrationSummary, CheckItem, Landmarks } from "@/lib/types";
import { Button } from "./ui/button";
import { Badge, Card, Input, Label, Meter, Notice } from "./ui/primitives";
import { toast, toastError } from "./toast";

interface ProtocolStage { id: string; instruction: string; optional: boolean }
interface Result extends CalibrationSummary { calibrated: boolean; missing_stages: string[]; stages: { stage: string; ok: boolean; message: string }[] }

const WINDOW = 45;              // ~1.5 s at 30 fps: judged as one contiguous window (stillness needs consecutive frames)
const ASSESS_EVERY_MS = 1200;
const STAGE_TIMEOUT_MS = 40_000;
const STAGE_LABEL: Record<string, string> = { front: "Front", arms_out: "Arms out", side: "Side", arms_up: "Arms up" };
const CHECK_LABEL: Record<string, string> = { visibility: "Body visible", in_frame: "In frame", distance: "Distance", centered: "Centred",
  facing: "Facing", pose: "Pose", still: "Holding still", confidence: "Tracking confidence" };

function Checklist({ checks }: { checks: CheckItem[] }) {
  if (!checks.length) return null;
  return (
    <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm" aria-label="Setup checklist">
      {checks.map((c) => (
        <li key={c.id} className="flex items-center gap-1.5">
          {c.ok ? <Check className="h-4 w-4 text-good" aria-hidden /> : <X className="h-4 w-4 text-bad" aria-hidden />}
          <span className={c.ok ? "text-mute" : "font-semibold"}>{CHECK_LABEL[c.id] ?? c.id}</span>
          <span className="sr-only">{c.ok ? "ok" : c.message}</span>
        </li>))}
    </ul>);
}

export function CalibrationWizard({ initialHeight }: { initialHeight: number | null }) {
  const qc = useQueryClient();
  const protocol = useQuery({ queryKey: ["calib-protocol"], queryFn: () => api<{ stages: ProtocolStage[] }>("/api/profile/calibrate/protocol") });
  const stages = protocol.data?.stages ?? [];
  const [height, setHeight] = useState<number | "">(initialHeight ?? ""), [idx, setIdx] = useState(0), [source, setSource] = useState<"camera" | "demo">("camera");
  const [frames, setFrames] = useState<Record<string, Landmarks[]>>({}), [assessment, setAssessment] = useState<Assessment | null>(null);
  const [capturing, setCapturing] = useState(false), [elapsed, setElapsed] = useState(0), [result, setResult] = useState<Result | null>(null);
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null), src = useRef<PoseSource | null>(null);
  const buf = useRef<Landmarks[]>([]), timer = useRef<ReturnType<typeof setInterval> | null>(null), inflight = useRef(false);
  const failures = useRef<Record<string, number>>({}), cameraLabel = useRef<string>("camera");
  const stage = stages[idx]?.id;

  const stopLoop = () => { if (timer.current) clearInterval(timer.current); timer.current = null; setCapturing(false); };
  const stopCamera = () => { src.current?.stop(); src.current = null; if (video.current) video.current.hidden = true; };
  useEffect(() => () => { stopLoop(); stopCamera(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const accept = (st: string, got: Landmarks[]) => {
    setFrames((f) => ({ ...f, [st]: got }));
    stopLoop();
    if (idx < stages.length - 1) setIdx(idx + 1);
  };

  const assess = async (st: string, window: Landmarks[]) =>
    api<Assessment>("/api/profile/calibrate/assess", { body: { stage: st, frames: window } });

  const capture = async () => {
    if (!stage) return;
    setAssessment(null); failures.current = {}; buf.current = [];
    try {
      if (source === "demo") {
        const got = await demoStage(stage), a = await assess(stage, got);
        setAssessment(a);
        if (a.ok) accept(stage, got); else toastError(new Error(a.message ?? "Demo pose rejected"));
        return;
      }
      if (!src.current) {
        src.current = await startCamera(video.current!, (lm) => {
          if (canvas.current) drawSkeleton(canvas.current, lm, video.current?.videoWidth ? video.current.videoWidth / video.current.videoHeight : 4 / 3);
          if (lm) { buf.current.push(lm); if (buf.current.length > WINDOW) buf.current.shift(); } else buf.current = [];   // gaps break the window
        });
        cameraLabel.current = src.current.stats().cameraLabel;
      }
      setCapturing(true);
      const t0 = Date.now(), st = stage;
      timer.current = setInterval(async () => {
        setElapsed(Date.now() - t0);
        if (Date.now() - t0 > STAGE_TIMEOUT_MS) {
          stopLoop();
          const worst = Object.entries(failures.current).sort((a, b) => b[1] - a[1])[0]?.[0];
          toastError(new Error(worst ? `Couldn't complete this pose. Most common problem: ${worst}` : "I can't see you - step into the frame."));
          return;
        }
        if (inflight.current || buf.current.length < WINDOW) return;
        inflight.current = true;
        const window = buf.current.slice();
        try {
          const a = await assess(st, window);
          setAssessment(a);
          if (a.ok) accept(st, window);
          else if (a.message) failures.current[a.message] = (failures.current[a.message] ?? 0) + 1;
        } catch (e) { stopLoop(); toastError(e); } finally { inflight.current = false; }
      }, ASSESS_EVERY_MS);
    } catch (e) { stopLoop(); stopCamera(); toastError(e instanceof Error && e.name === "NotAllowedError" ? new Error("Camera permission was denied. Choose Demo body or allow the camera.") : e); }
  };

  const submit = useMutation({
    mutationFn: () => api<Result>("/api/profile/calibrate/staged", { body: { height_cm: height || null, stages: frames, camera_label: source === "demo" ? "demo" : cameraLabel.current } }),
    onSuccess: (r) => { stopCamera(); setResult(r); qc.invalidateQueries({ queryKey: ["profile"] }); toast(r.status === "retry" ? "Calibration saved, but quality is low" : "Calibration saved"); },
    onError: toastError });

  const done = stages.filter((s) => frames[s.id]).map((s) => s.id);
  const requiredDone = stages.filter((s) => !s.optional).every((s) => frames[s.id]);
  const cur = stages[idx];
  const reset = () => { setResult(null); setFrames({}); setIdx(0); setAssessment(null); };

  return (
    <Card>
      <h2 className="text-xl font-bold">Body &amp; camera calibration</h2>
      <p className="text-sm text-mute">A few short poses measure your <i>relative</i> proportions for this camera setup, so movement is judged against your body.
        Not an anatomical scan; centimetres are only estimated if you enter your height.</p>
      <Label htmlFor="height">Height (cm, optional)</Label><Input id="height" type="number" className="w-32" value={height} onChange={(e) => setHeight(e.target.value ? +e.target.value : "")} />
      <ol className="mt-4 flex flex-wrap gap-2" aria-label="Calibration steps">{stages.map((s, i) => (
        <li key={s.id}><button type="button" disabled={capturing} onClick={() => { setIdx(i); setAssessment(null); }}>
          <Badge tone={frames[s.id] ? "good" : i === idx ? "brand" : "neutral"}>{frames[s.id] && <Check className="mr-1 inline h-3 w-3" />}{i + 1}. {STAGE_LABEL[s.id] ?? s.id}{s.optional ? " (optional)" : ""}</Badge></button></li>))}</ol>
      {!result && cur && (<div className="mt-4">
        <p className="font-semibold" data-testid="stage-instruction">{cur.instruction}</p>
        <div className="relative mt-2 aspect-[4/3] max-w-md overflow-hidden rounded-2xl border border-line bg-raise">
          <div className="absolute inset-0" style={{ transform: source === "camera" ? "scaleX(-1)" : undefined }}><video ref={video} playsInline muted hidden className="h-full w-full object-contain" /><canvas ref={canvas} className="absolute inset-0 h-full w-full" /></div>
          {assessment?.message && <div className="absolute inset-x-0 bottom-2 mx-auto w-fit max-w-[90%] rounded-full bg-black/70 px-4 py-1 text-center text-sm text-white" role="status">{assessment.message}</div>}
          {assessment?.ok && <div className="absolute inset-x-0 bottom-2 mx-auto w-fit rounded-full bg-good px-4 py-1 text-sm font-semibold text-black" role="status">Got it</div>}
        </div>
        {capturing && <Meter className="mt-2 max-w-md" value={(elapsed / STAGE_TIMEOUT_MS) * 100} />}
        {assessment && <Checklist checks={assessment.checks} />}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <select aria-label="Source" className="rounded-xl border border-line bg-raise px-3 py-2" value={source} onChange={(e) => setSource(e.target.value as "camera" | "demo")} disabled={capturing}><option value="camera">Camera</option><option value="demo">Demo body</option></select>
          {capturing ? <Button onClick={stopLoop}>Stop</Button>
            : <Button variant="primary" onClick={capture}>{frames[cur.id] ? "Retake" : "Capture"} {(STAGE_LABEL[cur.id] ?? cur.id).toLowerCase()}</Button>}
          {cur.optional && !frames[cur.id] && !capturing && <Button variant="ghost" onClick={() => idx < stages.length - 1 ? setIdx(idx + 1) : submit.mutate()}>Skip</Button>}
          <Button variant="primary" disabled={!frames.front || capturing || submit.isPending} onClick={() => submit.mutate()}>Finish ({done.length}/{stages.length})</Button></div>
        {frames.front && !requiredDone && <p className="mt-2 text-sm text-mute">You can finish now; the arms-out pose is needed to measure your arms.</p>}</div>)}
      {result && (<div className="mt-4 space-y-2" data-testid="calibration-result">
        <Notice tone={result.status === "good" ? "good" : result.status === "usable" ? "warn" : "bad"}>
          <b>{result.status === "good" ? "Calibrated" : result.status === "usable" ? "Calibrated (usable)" : "Low-quality calibration"}</b> · quality {Math.round(result.quality * 100)}%
          {result.method === "2d+3d" && result.ratio_agreement_3d != null ? ` · 2D/3D agreement ${Math.round(result.ratio_agreement_3d * 100)}%` : ""}</Notice>
        <div className="grid max-w-md grid-cols-3 gap-3 text-sm">{(["stability", "confidence", "coverage"] as const).map((k) => (
          <div key={k}><div className="flex justify-between text-mute"><span className="capitalize">{k}</span><span>{Math.round(result[k] * 100)}%</span></div><Meter value={result[k] * 100} /></div>))}</div>
        {result.warnings.map((w) => <Notice key={w}>{w}</Notice>)}
        {result.stages.filter((s) => !s.ok).map((s) => <Notice key={s.stage}><b>{STAGE_LABEL[s.stage] ?? s.stage}</b> pose wasn&apos;t clear: {s.message}</Notice>)}
        <Button onClick={reset}>Recalibrate</Button></div>)}
    </Card>);
}
