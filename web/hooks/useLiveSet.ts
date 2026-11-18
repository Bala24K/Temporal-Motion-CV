"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth, wsUrl } from "@/lib/api";
import type { Landmarks, LiveState, Recognition } from "@/lib/types";

export interface SetSummary { exercise: string; reps: number; partial_reps: number; avg_form_score: number | null; avg_rom: number | null }
export interface SavedMsg { type: "set_saved"; saved: boolean; reason?: "end_set" | "switch"; summary?: SetSummary; message?: string }

/** One line per message sent to the engine; replay with `python -m ml.replay <file>`. */
interface RecordedFrame { t: number; landmarks: Landmarks }

/** One WebSocket per set. Frames are fire-and-forget (the server keeps only the latest); control messages are
 *  queued losslessly on the server. */
export function useLiveSet() {
  const ws = useRef<WebSocket | null>(null);
  const lastSend = useRef(0);
  const e2e = useRef(0);
  const ending = useRef(false);
  const savedCb = useRef<((m: SavedMsg) => void) | null>(null);
  const recording = useRef<RecordedFrame[] | null>(null);
  const exerciseRef = useRef<string>("");
  const [state, setState] = useState<LiveState | null>(null);
  const [rec, setRec] = useState<Recognition | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [lost, setLost] = useState(false);

  const flash = (msg: string, ms = 3000) => { setNotice(msg); setTimeout(() => setNotice((n) => (n === msg ? null : n)), ms); };

  const connect = useCallback((exercise: string, targetReps: number, sessionId: number) => {
    const token = useAuth.getState().token ?? "";
    exerciseRef.current = exercise; ending.current = false; setLost(false);
    const s = new WebSocket(wsUrl(`/ws/live?exercise=${exercise}&target_reps=${targetReps}&session_id=${sessionId}&token=${encodeURIComponent(token)}`));
    ws.current = s;
    s.onopen = () => setOpen(true);
    s.onclose = () => {
      setOpen(false);
      // an unexpected close mid-set means the set was not saved: say so instead of failing silently
      if (!ending.current && ws.current === s) { setLost(true); setNotice("Connection lost - this set was not saved. Log it manually or restart the set."); }
    };
    s.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "state") { e2e.current = performance.now() - lastSend.current; setState(m); setRec(null); }
      else if (m.type === "recognition") setRec(m);
      else if (m.type === "exercise_selected") {
        exerciseRef.current = m.exercise;
        setState((prev) => prev ? { ...prev, exercise: m.exercise, reps: 0, partial_reps: 0, progress: 0, feedback: [] } : null);
        setRec(null);
        flash(`Switched to ${m.exercise_name || m.exercise}`, 2500);
      }
      else if (m.type === "setup") flash(m.message, 6000);       // camera/distance differs from calibration, or not calibrated
      else if (m.type === "target_reached") setNotice("Target reached - end the set when you're ready.");
      else if (m.type === "unsupported" || m.type === "error") setNotice(m.message);
      else if (m.type === "set_saved") {
        if (m.reason === "switch") flash(m.saved ? `Previous set saved (${m.summary?.reps ?? 0} reps)` : "Previous set had nothing to save", 3500);
        else savedCb.current?.(m);
      }
    };
  }, []);

  /** Switch exercise mid-set. The server saves the set in progress first, so counted reps are never lost. */
  const sendExercise = useCallback((exercise: string, weightKg = 0) => {
    if (ws.current?.readyState === 1) ws.current.send(JSON.stringify({ type: "select_exercise", exercise, weight_kg: weightKg }));
  }, []);

  const sendFrame = useCallback((lm: Landmarks, t: number) => {
    recording.current?.push({ t, landmarks: lm });
    if (ws.current?.readyState === 1) { lastSend.current = performance.now(); ws.current.send(JSON.stringify({ type: "frame", t, landmarks: lm })); }
  }, []);

  const sendStats = useCallback((cameraFps: number, poseMs: number) => {
    if (ws.current?.readyState === 1) ws.current.send(JSON.stringify({ type: "client_stats", camera_fps: cameraFps, pose_ms: poseMs, e2e_ms: e2e.current }));
  }, []);

  const endSet = useCallback((weightKg: number) => new Promise<SavedMsg | null>((resolve) => {
    const s = ws.current;
    if (!s || s.readyState !== 1) return resolve(null);
    ending.current = true;
    savedCb.current = resolve;
    s.send(JSON.stringify({ type: "end_set", weight_kg: weightKg }));
    setTimeout(() => resolve(null), 5000);
  }), []);

  /** Landmark recorder (no video): start/stop, then download as JSON Lines for the replay harness. */
  const startRecording = useCallback(() => { recording.current = []; }, []);
  const stopRecording = useCallback((meta: Record<string, unknown> = {}) => {
    const rows = recording.current; recording.current = null;
    if (!rows?.length) return 0;
    const header = JSON.stringify({ type: "meta", exercise: exerciseRef.current, recorded_at: new Date().toISOString(), user_agent: navigator.userAgent, ...meta });
    const blob = new Blob([[header, ...rows.map((r) => JSON.stringify({ type: "frame", ...r }))].join("\n") + "\n"], { type: "application/x-ndjson" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `recording-${exerciseRef.current || "set"}-${Date.now()}.jsonl`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    return rows.length;
  }, []);

  const close = useCallback(() => { ending.current = true; try { ws.current?.close(); } catch { /* already closed */ } ws.current = null; }, []);
  useEffect(() => close, [close]);
  return { state, rec, notice, open, lost, connect, sendFrame, sendStats, sendExercise, endSet, close, startRecording, stopRecording };
}
