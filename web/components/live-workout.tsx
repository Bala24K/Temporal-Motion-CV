"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, Check, Volume2, VolumeX, X } from "lucide-react";
import { api } from "@/lib/api";
import { useLiveSet } from "@/hooks/useLiveSet";
import { drawSkeleton, startCamera, startDemo, type PoseSource, type SkeletonStyle } from "@/lib/pose";
import type { Exercise, RepDetail, WorkoutItem } from "@/lib/types";
import { cn } from "@/lib/utils";
import { CameraDiagram } from "./camera-diagram";
import { RepStrip, topIssues } from "./rep-strip";
import { Button } from "./ui/button";
import { useConfirm } from "./ui/dialog";
import { Input, Label, Select } from "./ui/primitives";
import { toastError } from "./toast";

export interface QueueItem { item: WorkoutItem; set: number; of: number }
const FAULTS = [["", "Clean form"], ["shallow", "Shallow"], ["fast", "Too fast"], ["drift", "Elbow drift"], ["lean", "Torso lean"]];
const AUTO_ID = "auto";
const WARN_PULSE_MS = 700;

/** Exercises come from the registry (so admin-added definitions appear without a redeploy). */
function useExercises() {
  const q = useQuery({ queryKey: ["exercises"], queryFn: () => api<{ exercises: Exercise[] }>("/api/exercises") });
  return q.data?.exercises ?? [];
}

/** Short rising beep for a counted rep (WebAudio, no asset). */
function beep() {
  try {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new Ctx(), o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.setValueAtTime(660, ctx.currentTime); o.frequency.linearRampToValueAtTime(990, ctx.currentTime + 0.08);
    g.gain.setValueAtTime(0.18, ctx.currentTime); g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);
    o.connect(g).connect(ctx.destination); o.start(); o.stop(ctx.currentTime + 0.2); o.onended = () => ctx.close();
  } catch { /* audio unavailable */ }
}

export function LiveWorkout({ sessionId, queue, onDone }: { sessionId: number; queue: QueueItem[]; onDone: () => void }) {
  const [i, setI] = useState(0);
  const [resting, setResting] = useState<number | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const cur = queue[i];
  const next = (restSeconds: number) => {
    if (i + 1 >= queue.length) return onDone();
    setI(i + 1);
    if (restSeconds > 0) setResting(restSeconds);
  };
  // modal: move focus in, keep the page behind inert to assistive tech
  useEffect(() => { root.current?.querySelector<HTMLElement>("select, button, input")?.focus(); }, [i, resting]);
  if (!cur) { onDone(); return null; }
  return (
    <div ref={root} role="dialog" aria-modal="true" aria-label="Live workout"
      className="force-dark fixed inset-0 z-50 grid grid-rows-[auto_1fr_auto] bg-bg text-ink"
      style={{ paddingTop: "env(safe-area-inset-top)", paddingBottom: "env(safe-area-inset-bottom)" }}>
      {resting !== null
        ? <Rest seconds={resting} next={cur} onDone={() => setResting(null)} />
        : <SetScreen key={`${i}`} sessionId={sessionId} cur={cur} onFinished={() => next(cur.item.rest_seconds)} onLeave={onDone} />}
    </div>);
}

function Rest({ seconds, next, onDone }: { seconds: number; next: QueueItem; onDone: () => void }) {
  const [t, setT] = useState(seconds);
  useEffect(() => { if (t <= 0) { onDone(); return; } const id = setTimeout(() => setT(t - 1), 1000); return () => clearTimeout(id); }, [t, onDone]);
  const frac = t / Math.max(1, seconds);
  return (
    <div className="row-span-3 grid place-items-center text-center">
      <div>
        <p className="text-sm font-semibold uppercase tracking-widest text-mute">Rest</p>
        <div className="relative mx-auto my-4 grid h-56 w-56 place-items-center">
          <svg viewBox="0 0 100 100" className="absolute inset-0 -rotate-90" aria-hidden>
            <circle cx="50" cy="50" r="45" fill="none" stroke="var(--raise-2)" strokeWidth="6" />
            <circle cx="50" cy="50" r="45" fill="none" stroke="var(--brand)" strokeWidth="6" strokeLinecap="round" strokeDasharray="282.7" strokeDashoffset={282.7 * (1 - frac)} style={{ transition: "stroke-dashoffset 1s linear" }} />
          </svg>
          <span className="live-count !text-8xl" aria-live={t <= 5 ? "assertive" : "off"}>{t}</span>
        </div>
        <p className="text-ink-2">Next: <b className="text-ink">{next.item.exercise_name}</b> · set {next.set}/{next.of}</p>
        <div className="mt-4 flex justify-center gap-2">
          <Button onClick={() => setT(t + 30)}>+30 s</Button>
          <Button variant="primary" onClick={onDone}>Skip rest</Button>
        </div>
      </div>
    </div>);
}

/** Vertical gauge: progress through the user's range (0 = start pose, 1 = turnaround), with the counting line. */
function PhaseGauge({ value, target, personal }: { value: number; target: number; personal: boolean }) {
  const p = Math.max(0, Math.min(1, value));
  return (
    <div className="absolute bottom-24 left-4 top-24 w-3 rounded-full bg-white/10 md:left-6" aria-hidden title="Progress through your range of motion">
      <div className="absolute inset-x-0 bottom-0 rounded-full bg-brand transition-[height] duration-100" style={{ height: `${p * 100}%` }} />
      <div className="absolute -left-1.5 -right-1.5 h-0.5 bg-ink" style={{ bottom: `${target * 100}%` }} />
      <span className="absolute left-5 whitespace-nowrap text-[10px] font-semibold uppercase tracking-wider text-mute" style={{ bottom: `calc(${target * 100}% - 6px)` }}>
        {personal ? "your range" : "count line"}</span>
    </div>);
}

function TrackingBadge({ tracking, reconnecting }: { tracking?: string; reconnecting: boolean }) {
  const [label, tone] = reconnecting ? ["Reconnecting…", "bg-info"] : tracking === "lost" ? ["Tracking lost", "bg-bad"]
    : tracking === "degraded" ? ["Tracking weak", "bg-warn"] : ["Tracking good", "bg-good"];
  return (
    <span className="flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-xs font-semibold" role="status" data-testid="tracking">
      <span className={cn("h-2 w-2 rounded-full", tone)} aria-hidden />{label}</span>);
}

function SetScreen({ sessionId, cur, onFinished, onLeave }: { sessionId: number; cur: QueueItem; onFinished: () => void; onLeave: () => void }) {
  const { item } = cur;
  const ai = item.ai_analysis || item.exercise_id === AUTO_ID;
  const live = useLiveSet();
  const ask = useConfirm();
  const exercises = useExercises();
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null), src = useRef<PoseSource | null>(null);
  const [source, setSource] = useState("camera"), [fault, setFault] = useState("");
  // Quick set used to silently start a bicep-curl session ("auto" was mapped to "bicep_curl"). Now the user picks.
  const [exerciseId, setExerciseId] = useState(item.exercise_id && item.exercise_id !== AUTO_ID ? item.exercise_id : "");
  const [recordOn, setRecordOn] = useState(false);
  const [sound, setSound] = useState(() => { try { return localStorage.getItem("kinetic.sound") === "1"; } catch { return false; } });
  const [askManual, setAskManual] = useState<null | "empty" | "failed">(null);
  const [summary, setSummary] = useState<null | { reps: number; partial_reps: number; avg_form_score: number | null; reps_detail: RepDetail[] }>(null);
  const [weight, setWeight] = useState(item.weight_kg), [reps, setReps] = useState(item.reps);
  const [running, setRunning] = useState(false), [busy, setBusy] = useState(false);
  const [announce, setAnnounce] = useState("");
  const mirror = source === "camera";
  const s = live.state;
  const ex = useMemo(() => exercises.find((e) => e.id === (s?.exercise && s.exercise !== AUTO_ID ? s.exercise : exerciseId)), [exercises, s?.exercise, exerciseId]);
  const style = useRef<SkeletonStyle & { warnUntil: number }>({ active: [], warnUntil: 0 });
  const prevReps = useRef(0);

  useEffect(() => () => { src.current?.stop(); live.close(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { const e = s?.exercise; if (exerciseId === AUTO_ID && e && e !== AUTO_ID) setExerciseId(e); }, [s?.exercise]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (askManual && s) setReps(s.reps); }, [askManual]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!running) return; const id = setInterval(() => { const st = src.current?.stats(); if (st) live.sendStats(st.cameraFps, st.poseMs); }, 2000); return () => clearInterval(id); }, [running]); // eslint-disable-line react-hooks/exhaustive-deps
  // skeleton: emphasise the joints this exercise measures; pulse amber when a check fails on the last rep
  useEffect(() => { style.current.active = ex?.rep_joints ?? []; }, [ex]);
  useEffect(() => { if (s?.feedback.some((f) => f.status === "warning")) style.current.warnUntil = performance.now() + WARN_PULSE_MS; }, [s?.last_rep]); // eslint-disable-line react-hooks/exhaustive-deps
  // rep announced to screen readers, optional beep
  useEffect(() => {
    const n = s?.reps ?? 0;
    if (n > prevReps.current) { setAnnounce(`Rep ${n}${item.reps ? ` of ${item.reps}` : ""}`); if (sound) beep(); }
    prevReps.current = n;
  }, [s?.reps]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    try {
      if (!exerciseId) return;
      setRunning(true);
      if (recordOn) live.startRecording();
      live.connect(exerciseId, item.reps, sessionId);
      const onFrame = (lm: any, t: number) => {
        if (canvas.current) drawSkeleton(canvas.current, lm, video.current?.videoWidth ? video.current.videoWidth / video.current.videoHeight : 4 / 3,
          { active: style.current.active, warn: performance.now() < style.current.warnUntil });
        if (lm) live.sendFrame(lm, t);
      };
      src.current = source === "demo" ? await startDemo(exerciseId, reps, fault, onFrame) : await startCamera(video.current!, onFrame);
    } catch (e) { setRunning(false); live.close(); toastError(e instanceof Error && e.name === "NotAllowedError" ? new Error("Camera permission was denied. Choose Demo clip or allow the camera.") : e); }
  };
  const logManual = () => api(`/api/sessions/${sessionId}/sets`, { method: "POST", body: { exercise: ex?.name || item.exercise_name, reps, weight_kg: weight, set_number: cur.set } });
  const finish = () => { live.close(); onFinished(); };
  const end = async () => {
    setBusy(true);
    try {
      if (recordOn) live.stopRecording({ source, exercise_selected: exerciseId });
      if (ai && running && !askManual) {
        src.current?.stop();
        const r = await live.endSet(weight);
        if (!r || (r.empty && !r.saved)) { setAskManual(r ? "empty" : "failed"); setBusy(false); return; }
        const sm = r.summary as typeof summary;
        if (sm && sm.reps_detail?.length) { setSummary(sm); setBusy(false); return; }     // show the set summary first
      } else await logManual();
      finish();
    } catch (e) { toastError(e); setBusy(false); }
  };
  const leave = () => ask({ title: "Leave workout?", body: "Sets you've already logged are kept.", confirmLabel: "Leave", danger: true,
    onConfirm: () => { src.current?.stop(); live.close(); onLeave(); } });

  const warning = s?.feedback.find((f) => f.status === "warning");
  const positive = !warning ? s?.feedback.find((f) => f.status === "good") : undefined;
  const target = ex?.count_progress ?? 0.55;

  return (
    <>
      <header className="flex items-center gap-3 px-4 py-3">
        <Button variant="ghost" size="sm" aria-label="Leave workout" className="text-ink" onClick={leave}><X className="h-5 w-5" /></Button>
        {running
          ? <span className="font-display text-xl font-bold" data-testid="exercise-name">{ex?.name ?? (exerciseId === AUTO_ID ? "Detecting exercise…" : item.exercise_name)}</span>
          : <span className="font-display text-xl font-bold">{ai ? "Set up" : item.exercise_name}</span>}
        <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-semibold">Set {cur.set}/{cur.of}</span>
        <span className="sr-only" data-testid="phase">{s?.phase ?? "ready"}</span>
        <div className="ml-auto flex items-center gap-2">{running && <TrackingBadge tracking={s?.tracking} reconnecting={live.reconnecting} />}</div>
      </header>

      <div className="relative min-h-0 overflow-hidden">
        <div className="absolute inset-0" style={{ transform: mirror ? "scaleX(-1)" : undefined }}>
          <video ref={video} playsInline muted hidden className="absolute inset-0 h-full w-full object-contain opacity-80" />
          <canvas ref={canvas} className="absolute inset-0 h-full w-full" aria-label="Pose overlay" />
        </div>
        <p className="sr-only" aria-live="polite" aria-atomic="true">{announce}</p>

        {/* ---------------- setup (before the set) ---------------- */}
        {!running && !summary && !askManual && (
          <div className="absolute inset-0 grid place-items-center overflow-y-auto p-4">
            <div className="w-full max-w-md rounded-3xl border border-line bg-surface/95 p-5 shadow-2xl backdrop-blur">
              {ai ? (<>
                <Label htmlFor="ex-select" className="mt-0">Exercise</Label>
                <Select id="ex-select" aria-label="Select exercise" className="font-display text-lg font-bold" value={exerciseId} onChange={(e) => setExerciseId(e.target.value)}>
                  {!exerciseId && <option value="">Choose exercise…</option>}
                  {exercises.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
                  <option value={AUTO_ID}>Auto-detect (experimental)</option>
                </Select>
                {ex && <div className="mt-3 rounded-2xl bg-raise p-3"><CameraDiagram view={ex.camera.preferred} fullBody={ex.camera.full_body} />
                  {ex.camera.notes && <p className="mt-2 text-xs text-mute">{ex.camera.notes}</p>}</div>}
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div><Label htmlFor="kg" className="mt-0">Weight (kg)</Label><Input id="kg" type="number" min={0} step={0.5} value={weight} onChange={(e) => setWeight(+e.target.value)} /></div>
                  <div><Label htmlFor="src" className="mt-0">Video source</Label>
                    <Select id="src" aria-label="Video source" value={source} onChange={(e) => setSource(e.target.value)}><option value="camera">Camera</option><option value="demo">Demo clip</option></Select></div>
                </div>
                {source === "demo" && <><Label htmlFor="fault">Demo form</Label><Select id="fault" aria-label="Demo form" value={fault} onChange={(e) => setFault(e.target.value)}>{FAULTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</Select></>}
                <div className="mt-3 flex flex-wrap items-center gap-4 text-sm text-ink-2">
                  <label className="flex items-center gap-2"><input type="checkbox" checked={sound} onChange={(e) => { setSound(e.target.checked); try { localStorage.setItem("kinetic.sound", e.target.checked ? "1" : "0"); } catch { /* blocked */ } }} />
                    {sound ? <Volume2 className="h-4 w-4" aria-hidden /> : <VolumeX className="h-4 w-4" aria-hidden />}Beep per rep</label>
                  <label className="flex items-center gap-2" title="Saves the landmarks sent to the engine (no video) as a .jsonl file for replay and evaluation">
                    <input type="checkbox" checked={recordOn} onChange={(e) => setRecordOn(e.target.checked)} />Record landmarks</label>
                </div>
                <Button variant="primary" size="lg" className="mt-5 w-full" onClick={start} disabled={!exerciseId} title={!exerciseId ? "Choose an exercise first" : undefined}>Start</Button>
                <p className="mt-2 text-center text-xs text-mute">Target {item.reps} reps · video stays on your device</p>
              </>) : (<>
                <p className="text-sm"><b>Manual tracking.</b> Reliable AI analysis isn&apos;t available for this exercise yet - log your set here.</p>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div><Label htmlFor="rp" className="mt-0">reps</Label><Input id="rp" type="number" min={0} value={reps} onChange={(e) => setReps(+e.target.value)} /></div>
                  <div><Label htmlFor="kg" className="mt-0">kg</Label><Input id="kg" type="number" min={0} step={0.5} value={weight} onChange={(e) => setWeight(+e.target.value)} /></div>
                </div>
                <Button variant="primary" size="lg" className="mt-5 w-full" disabled={busy} onClick={end}>Log set</Button>
              </>)}
            </div>
          </div>)}

        {/* ---------------- the set ---------------- */}
        {running && !summary && !askManual && (<>
          <PhaseGauge value={s?.progress ?? 0} target={target} personal={!!s?.personal_range} />
          <div className="absolute right-4 top-3 flex max-w-[16rem] flex-col items-end gap-2 text-sm">
            {s && (s.range_capture ? (
              <div className="flex flex-col items-end gap-2 rounded-2xl bg-black/60 p-3 text-right backdrop-blur" role="status">
                <span>{live.rangeMsg?.text || "Do 2-3 slow reps through your full comfortable range."}</span>
                <span className="flex gap-2"><Button size="sm" variant="ghost" className="text-ink" onClick={() => live.sendRange("cancel")}>Cancel</Button>
                  <Button size="sm" variant="primary" onClick={() => live.sendRange("finish")}>Done</Button></span>
              </div>
            ) : (
              <Button size="sm" variant={s.personal_range ? "ghost" : "outline"} className={s.personal_range ? "text-mute" : "border-white/30 bg-black/40 text-ink"} onClick={() => live.sendRange("start")}
                title="Calibrate counting to your own range of motion for this exercise">{s.personal_range ? "Recapture my range" : "Capture my range"}</Button>))}
            {live.rangeMsg && !s?.range_capture && <span className={`rounded-full px-3 py-1 text-right ${live.rangeMsg.ok ? "bg-good text-black" : "bg-warn text-black"}`} role="status">{live.rangeMsg.text}</span>}
          </div>
          {/* one blocking message at a time, most important first */}
          {(() => {
            const msg = s && s.quality.message && (!s.quality.ok || s.tracking !== "ok") ? s.quality.message : s && !s.view_ok ? s.view_hint : null;
            return msg ? <div className="absolute left-1/2 top-3 max-w-[min(90%,28rem)] -translate-x-1/2 rounded-2xl bg-warn px-4 py-2 text-center text-sm font-semibold text-black" role="alert">{msg}</div> : null;
          })()}
          <div className="pointer-events-none absolute bottom-4 right-4 text-right md:bottom-6 md:right-8">
            <motion.div key={s?.reps ?? 0} initial={{ scale: 1.12 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 400, damping: 18 }}
              className="live-count drop-shadow-[0_2px_10px_rgba(0,0,0,0.85)]" data-testid="rep-count" aria-hidden>{s?.reps ?? 0}</motion.div>
            <div className="text-lg font-semibold text-ink-2">of {item.reps}{s?.partial_reps ? <span className="text-mute"> · {s.partial_reps} partial</span> : null}</div>
          </div>
          <div className="absolute bottom-4 left-12 flex max-w-[calc(100%-14rem)] flex-col items-start gap-2 md:left-16" aria-live="polite">
            <AnimatePresence mode="popLayout">
              {live.rec && live.rec.action !== "wait" && (
                <motion.div key="rec" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-wrap items-center gap-2 rounded-2xl border border-white/20 bg-black/60 px-4 py-2 text-sm font-semibold backdrop-blur">
                  <span>{live.rec.message}</span>
                  {live.rec.action === "confirm" && live.rec.top?.members.map((m) => (
                    <button key={m} type="button" className="rounded-full bg-brand px-3 py-1 text-brand-ink"
                      onClick={() => { setExerciseId(m); live.sendExercise(m, weight); }}>{exercises.find((e) => e.id === m)?.name ?? m.replace(/_/g, " ")}</button>))}
                </motion.div>)}
              {(warning ?? positive) && (
                <motion.span key={(warning ?? positive)!.message} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                  className={cn("flex items-center gap-2 rounded-2xl border bg-black/60 px-4 py-2 text-base font-semibold backdrop-blur", warning ? "border-warn" : "border-white/20")}>
                  {warning ? <AlertTriangle className="h-5 w-5 text-warn" aria-hidden /> : <Check className="h-5 w-5 text-good" aria-hidden />}{(warning ?? positive)!.message}</motion.span>)}
            </AnimatePresence>
            {live.notice && <span className="rounded-full bg-white/15 px-4 py-1.5 text-sm font-semibold backdrop-blur">{live.notice}</span>}
          </div>
        </>)}

        {/* ---------------- after the set ---------------- */}
        {(summary || askManual) && (
          <div className="absolute inset-0 grid place-items-center overflow-y-auto bg-black/50 p-4 backdrop-blur-sm">
            <div className="w-full max-w-lg rounded-3xl border border-line bg-surface p-5 shadow-2xl" role="region" aria-label="Set summary">
              {summary ? (<>
                <p className="text-sm font-semibold uppercase tracking-widest text-mute">Set complete</p>
                <div className="mt-2 flex items-end gap-6">
                  <div><div className="live-count !text-7xl">{summary.reps}</div><div className="text-sm text-mute">reps{summary.partial_reps ? ` · ${summary.partial_reps} partial` : ""}</div></div>
                  {summary.avg_form_score != null && <div><div className="font-display text-4xl font-black tabular-nums">{summary.avg_form_score}</div><div className="text-sm text-mute">form checks passed</div></div>}
                </div>
                <RepStrip reps={summary.reps_detail} compact />
                {topIssues(summary.reps_detail).slice(0, 3).map(([m, n]) => <p key={m} className="mt-1 text-sm text-warn">{m} <span className="text-mute">({n}×)</span></p>)}
                <Button variant="primary" size="lg" className="mt-5 w-full" onClick={finish}>Continue</Button>
              </>) : (<>
                <p className="font-semibold text-warn" role="alert">{askManual === "empty" ? "No reps were detected." : "Couldn't save automatically."} Log it manually?</p>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div><Label htmlFor="rp2" className="mt-0">reps</Label><Input id="rp2" type="number" min={0} value={reps} onChange={(e) => setReps(+e.target.value)} /></div>
                  <div><Label htmlFor="kg2" className="mt-0">kg</Label><Input id="kg2" type="number" min={0} step={0.5} value={weight} onChange={(e) => setWeight(+e.target.value)} /></div>
                </div>
                <div className="mt-5 flex gap-2"><Button variant="ghost" className="flex-1 text-ink" disabled={busy} onClick={finish}>Discard</Button>
                  <Button variant="primary" className="flex-1" disabled={busy} onClick={end}>Log set</Button></div>
              </>)}
            </div>
          </div>)}
      </div>

      <footer className="flex items-center justify-end gap-3 px-4 py-3" data-testid="live-controls">
        {running && !summary && !askManual && <Button variant="primary" size="lg" disabled={busy} onClick={end}>End set</Button>}
      </footer>
      {ask.dialog}
    </>);
}
