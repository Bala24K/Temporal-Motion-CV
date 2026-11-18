"use client";
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { Check, AlertTriangle, X } from "lucide-react";
import { api } from "@/lib/api";
import { useLiveSet } from "@/hooks/useLiveSet";
import { drawSkeleton, startCamera, startDemo, type PoseSource } from "@/lib/pose";
import type { Exercise, WorkoutItem } from "@/lib/types";
import { Button } from "./ui/button";
import { Input, Notice, Select } from "./ui/primitives";
import { toastError } from "./toast";

export interface QueueItem { item: WorkoutItem; set: number; of: number }
const FAULTS = [["", "Clean form"], ["shallow", "Shallow"], ["fast", "Too fast"], ["drift", "Elbow drift"], ["lean", "Torso lean"]];
const AUTO: [string, string] = ["auto", "Auto-detect (experimental)"];

/** Exercises come from the registry (so admin-added definitions appear without a redeploy). */
function useExerciseOptions(): [string, string][] {
  const q = useQuery({ queryKey: ["exercises"], queryFn: () => api<{ exercises: Exercise[] }>("/api/exercises") });
  return [...(q.data?.exercises ?? []).map((e): [string, string] => [e.id, e.name]), AUTO];
}

export function LiveWorkout({ sessionId, queue, onDone }: { sessionId: number; queue: QueueItem[]; onDone: () => void }) {
  const [i, setI] = useState(0);
  const [resting, setResting] = useState<number | null>(null);
  const cur = queue[i];
  const next = (restSeconds: number) => {
    if (i + 1 >= queue.length) return onDone();
    setI(i + 1);
    if (restSeconds > 0) setResting(restSeconds);
  };
  if (!cur) { onDone(); return null; }
  return (
    <div role="dialog" aria-label="Live workout" className="fixed inset-0 z-50 grid grid-rows-[auto_1fr_auto] bg-[#0a1016] text-[#eef3f8]" style={{ paddingTop: "env(safe-area-inset-top)", paddingBottom: "env(safe-area-inset-bottom)" }}>
      {resting !== null
        ? <Rest seconds={resting} next={cur} onDone={() => setResting(null)} />
        : <SetScreen key={`${i}`} sessionId={sessionId} cur={cur} onFinished={() => next(cur.item.rest_seconds)} onLeave={onDone} />}
    </div>);
}

function Rest({ seconds, next, onDone }: { seconds: number; next: QueueItem; onDone: () => void }) {
  const [t, setT] = useState(seconds);
  useEffect(() => { if (t <= 0) { onDone(); return; } const id = setTimeout(() => setT(t - 1), 1000); return () => clearTimeout(id); }, [t, onDone]);
  return (
    <div className="row-span-3 grid place-items-center text-center">
      <div><p className="text-[#93a3b2]">Rest</p><div className="font-display text-9xl font-extrabold" aria-live="off">{t}</div>
        <p>Next: {next.item.exercise_name} — set {next.set}/{next.of}</p><Button variant="primary" className="mt-4" onClick={onDone}>Skip rest</Button></div>
    </div>);
}

function SetScreen({ sessionId, cur, onFinished, onLeave }: { sessionId: number; cur: QueueItem; onFinished: () => void; onLeave: () => void }) {
  const { item } = cur;
  const ai = item.ai_analysis || item.exercise_id === "auto";
  const live = useLiveSet();
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null), src = useRef<PoseSource | null>(null);
  const [source, setSource] = useState("camera"), [fault, setFault] = useState("");
  // Quick set used to silently start a bicep-curl session ("auto" was mapped to "bicep_curl"). Now the user picks.
  const [exerciseId, setExerciseId] = useState(item.exercise_id && item.exercise_id !== "auto" ? item.exercise_id : "");
  const options = useExerciseOptions();
  const [recordOn, setRecordOn] = useState(false);
  const [weight, setWeight] = useState(item.weight_kg), [reps, setReps] = useState(item.reps);
  const [running, setRunning] = useState(false), [busy, setBusy] = useState(false);
  const mirror = source === "camera";

  useEffect(() => () => { src.current?.stop(); live.close(); }, []);
  // auto-detect locked onto an exercise: show it in the selector instead of "Auto-detect"
  useEffect(() => { const ex = live.state?.exercise; if (exerciseId === "auto" && ex && ex !== "auto") setExerciseId(ex); }, [live.state?.exercise]); // eslint-disable-line react-hooks/exhaustive-deps
  // connection dropped mid-set: prefill the manual log with what was counted so far
  useEffect(() => { if (live.lost && live.state) setReps(live.state.reps); }, [live.lost]); // eslint-disable-line react-hooks/exhaustive-deps // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!running) return; const id = setInterval(() => { const s = src.current?.stats(); if (s) live.sendStats(s.cameraFps, s.poseMs); }, 2000); return () => clearInterval(id); }, [running]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    try {
      if (!exerciseId) return;
      setRunning(true);
      if (recordOn) live.startRecording();
      live.connect(exerciseId, item.reps, sessionId);
      const onFrame = (lm: any, t: number) => { if (canvas.current) drawSkeleton(canvas.current, lm, video.current?.videoWidth ? video.current.videoWidth / video.current.videoHeight : 4 / 3); if (lm) live.sendFrame(lm, t); };
      src.current = source === "demo" ? await startDemo(exerciseId, reps, fault, onFrame) : await startCamera(video.current!, onFrame);
    } catch (e) { setRunning(false); live.close(); toastError(e instanceof Error && e.name === "NotAllowedError" ? new Error("Camera permission was denied. Choose Demo clip or allow the camera.") : e); }
  };
  const end = async () => {
    setBusy(true);
    try {
      if (recordOn) live.stopRecording({ source, exercise_selected: exerciseId });
      if (ai && running && !live.lost) { src.current?.stop(); await live.endSet(weight); }
      else await api(`/api/sessions/${sessionId}/sets`, { method: "POST", body: { exercise: options.find(([id]) => id === exerciseId)?.[1] || item.exercise_name, reps, weight_kg: weight, set_number: cur.set } });
      live.close(); onFinished();
    } catch (e) { toastError(e); setBusy(false); }
  };

  const s = live.state;
  return (
    <>
      <header className="flex items-center gap-3 px-4 py-3">
        <Button variant="ghost" size="sm" aria-label="Leave workout" className="text-[#eef3f8]" onClick={() => { if (confirm("Leave workout? Logged sets are kept.")) { src.current?.stop(); live.close(); onLeave(); } }}><X className="h-5 w-5" /></Button>
        <Select
          aria-label="Select exercise"
          className="font-display text-lg font-bold bg-[#15202b] text-[#eef3f8] cursor-pointer max-w-[240px]"
          value={exerciseId}
          onChange={(e) => {
            const nextEx = e.target.value;
            setExerciseId(nextEx);
            if (running) live.sendExercise(nextEx, weight);
          }}
        >
          {!exerciseId && <option value="" className="bg-[#15202b] text-[#eef3f8]">Choose exercise…</option>}
          {options.map(([val, label]) => (
            <option key={val} value={val} className="bg-[#15202b] text-[#eef3f8]">{label}</option>
          ))}
        </Select>
        <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-semibold">Set {cur.set}/{cur.of}</span>
        <span className="ml-auto rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-semibold" data-testid="phase">{s?.phase ?? (live.rec ? live.rec.action : "ready")}</span>
      </header>
      <div className="relative min-h-0">
        <div className="absolute inset-0" style={{ transform: mirror ? "scaleX(-1)" : undefined }}>
          <video ref={video} playsInline muted hidden className="absolute inset-0 h-full w-full object-contain opacity-80" />
          <canvas ref={canvas} className="absolute inset-0 h-full w-full" aria-label="Pose overlay" />
        </div>
        {!ai && <div className="absolute inset-x-4 top-4"><Notice><b>Manual tracking.</b> Reliable AI analysis isn&apos;t available for this exercise yet — log your set below.</Notice></div>}
        {s && (!s.quality.ok || s.tracking !== "ok") && s.quality.message && <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-full bg-warn px-4 py-1.5 text-sm font-semibold text-black" role="alert">{s.quality.message}</div>}
        {s && s.tracking === "degraded" && s.quality.ok && <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-full bg-white/15 px-4 py-1.5 text-sm font-semibold backdrop-blur" role="status">Tracking weak - keep your whole body in view</div>}
        {s && !s.view_ok && s.view_hint && <div className="absolute inset-x-4 top-14 mx-auto w-fit max-w-full rounded-2xl bg-black/60 px-4 py-2 text-center text-sm backdrop-blur" role="status">{s.view_hint}</div>}
        {ai && running && s && (
          <div className="absolute right-4 top-4 flex flex-col items-end gap-2 text-sm">
            {s.range_capture ? (
              <div className="flex max-w-[16rem] flex-col items-end gap-2 rounded-2xl bg-black/60 p-3 text-right backdrop-blur" role="status">
                <span>{live.rangeMsg?.text ?? "Do 2-3 slow reps through your full comfortable range."}</span>
                <span className="flex gap-2"><Button size="sm" variant="ghost" className="text-[#eef3f8]" onClick={() => live.sendRange("cancel")}>Cancel</Button>
                  <Button size="sm" variant="primary" onClick={() => live.sendRange("finish")}>Done</Button></span>
              </div>
            ) : (
              <Button size="sm" variant={s.personal_range ? "ghost" : "primary"} className={s.personal_range ? "text-[#93a3b2]" : ""} onClick={() => live.sendRange("start")}
                title="Calibrate counting to your own range of motion for this exercise">{s.personal_range ? "Recapture my range" : "Capture my range"}</Button>
            )}
            {live.rangeMsg && !s.range_capture && <span className={`max-w-[16rem] rounded-full px-3 py-1 text-right ${live.rangeMsg.ok ? "bg-good text-black" : "bg-warn text-black"}`} role="status">{live.rangeMsg.text}</span>}
          </div>)}
        {ai && (
          <div className="pointer-events-none absolute left-[clamp(4rem,17%,14rem)] top-1/2 -translate-x-1/2 -translate-y-1/2 text-center max-md:left-1/2 max-md:top-[26%]">
            <motion.div key={s?.reps ?? 0} initial={{ scale: 1.12 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 400, damping: 18 }}
              className="live-count font-display font-extrabold drop-shadow-[0_4px_40px_#000a]" data-testid="rep-count">{s?.reps ?? 0}</motion.div>
            <div className="text-[#93a3b2]">of {item.reps}{s?.partial_reps ? ` · ${s.partial_reps} partial` : ""}</div>
          </div>)}
        <ProgressRing value={s?.progress ?? 0} />
        <div className="absolute bottom-4 left-4 flex max-w-[calc(100%-7rem)] flex-col items-start gap-2" aria-live="polite">
          <AnimatePresence mode="popLayout">
            {live.rec && live.rec.action !== "wait" && (
              <motion.div key="rec" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="pointer-events-auto flex flex-wrap items-center gap-2 rounded-2xl border border-white/20 bg-black/50 px-4 py-2 text-sm font-semibold backdrop-blur">
                <span>{live.rec.message}</span>
                {live.rec.action === "confirm" && live.rec.top?.members.map((m) => (
                  <button key={m} type="button" className="rounded-full bg-brand px-3 py-1 text-brand-ink"
                    onClick={() => { setExerciseId(m); live.sendExercise(m, weight); }}>{options.find(([id]) => id === m)?.[1] ?? m.replace(/_/g, " ")}</button>))}
              </motion.div>)}
            {s?.feedback.map((f) => (
              <motion.span key={f.message} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                className={`flex items-center gap-1.5 rounded-full border bg-black/50 px-4 py-1.5 text-sm font-semibold backdrop-blur ${f.status === "warning" ? "border-warn" : "border-white/20"}`}>
                {f.status === "good" ? <Check className="h-4 w-4 text-[#4fe0a2]" /> : f.status === "warning" ? <AlertTriangle className="h-4 w-4 text-warn" /> : null}{f.message}</motion.span>))}
          </AnimatePresence>
          {live.notice && <span className="rounded-full bg-brand px-4 py-1.5 text-sm font-semibold text-brand-ink">{live.notice}</span>}
        </div>
      </div>
      <footer className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        {ai ? (
          <div className="flex flex-wrap items-center gap-2 [&_select]:w-auto [&_select]:bg-[#15202b] [&_select]:text-[#eef3f8]">
            <Select aria-label="Video source" value={source} disabled={running} onChange={(e) => setSource(e.target.value)}><option value="camera">Camera</option><option value="demo">Demo clip</option></Select>
            {source === "demo" && <Select aria-label="Demo form" value={fault} disabled={running} onChange={(e) => setFault(e.target.value)}>{FAULTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</Select>}
            <label className="flex items-center gap-1.5 text-xs text-[#93a3b2]" title="Saves the landmarks sent to the engine (no video) as a .jsonl file for replay and evaluation">
              <input type="checkbox" checked={recordOn} disabled={running} onChange={(e) => setRecordOn(e.target.checked)} />Record landmarks</label>
            <Button variant="primary" onClick={start} disabled={running || !exerciseId} title={!exerciseId ? "Choose an exercise first" : undefined}>{running ? "Running" : "Start"}</Button>
          </div>) : <span />}
        <div className="flex items-center gap-2 text-sm [&_input]:w-20 [&_input]:bg-[#15202b] [&_input]:text-[#eef3f8]">
          <label htmlFor="kg">kg</label><Input id="kg" type="number" min={0} step={0.5} value={weight} onChange={(e) => setWeight(+e.target.value)} />
          {(!ai || live.lost) && <><label htmlFor="rp">reps</label><Input id="rp" type="number" min={0} value={reps} onChange={(e) => setReps(+e.target.value)} /></>}
          <Button variant="primary" disabled={busy} onClick={end}>{ai && !live.lost ? "End set" : "Log set"}</Button>
        </div>
      </footer>
    </>);
}

function ProgressRing({ value }: { value: number }) {
  const p = Math.max(0, Math.min(1, value));
  return (
    <svg className="absolute bottom-4 right-4 h-[76px] w-[76px]" viewBox="0 0 36 36" aria-hidden>
      <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(255,255,255,.2)" strokeWidth="3" />
      <circle cx="18" cy="18" r="15" fill="none" stroke="#6f92ff" strokeWidth="3" strokeLinecap="round" strokeDasharray="94.2" strokeDashoffset={94.2 * (1 - p)} transform="rotate(-90 18 18)" style={{ transition: "stroke-dashoffset .1s" }} />
    </svg>);
}
