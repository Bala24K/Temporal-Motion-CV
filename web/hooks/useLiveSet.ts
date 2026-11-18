"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, useAuth, wsUrl } from "@/lib/api";
import type { ClientMessage, ServerMessage, SetSavedOut } from "@/lib/protocol";
import type { Landmarks, LiveState, Recognition } from "@/lib/types";

export type SavedMsg = SetSavedOut;
export interface SetSummary { exercise: string; reps: number; partial_reps: number; avg_form_score: number | null; avg_rom: number | null }

/** One line per message sent to the engine; replay with `python -m ml.replay <file>`. */
interface RecordedFrame { t: number; landmarks: Landmarks }

const RETRY_MS = [500, 1000, 2000, 4000, 8000, 15000, 15000, 15000];   // ~60 s total; the server parks a set for 120 s
const STATE_HZ = 15;

/** One WebSocket per set (protocol: backend/protocol.py, types generated into lib/protocol.ts).
 *  - auth token goes in the first message, never the URL;
 *  - the set has a client-generated key: if the connection drops, we reconnect with the same key and the server
 *    resumes the set with every rep counted so far; if we can't reconnect, End set finishes it over REST;
 *  - frames are fire-and-forget (server keeps the latest); control messages are lossless. */
export function useLiveSet() {
  const ws = useRef<WebSocket | null>(null);
  const params = useRef<{ exercise: string; targetReps: number; sessionId: number; setKey: string } | null>(null);
  const lastSend = useRef(0);
  const e2e = useRef(0);
  const ending = useRef(false);
  const attempt = useRef(0);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const savedCb = useRef<((m: SavedMsg | null) => void) | null>(null);
  const recording = useRef<RecordedFrame[] | null>(null);
  const exerciseRef = useRef<string>("");
  const pendingState = useRef<LiveState | null>(null);
  const raf = useRef<number | null>(null);
  const [state, setState] = useState<LiveState | null>(null);
  const [rec, setRec] = useState<Recognition | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [lost, setLost] = useState(false);
  const [rangeMsg, setRangeMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const flash = (msg: string, ms = 3000) => { setNotice(msg); setTimeout(() => setNotice((n) => (n === msg ? null : n)), ms); };
  const send = (m: ClientMessage) => { if (ws.current?.readyState === 1) { ws.current.send(JSON.stringify(m)); return true; } return false; };

  /** At most one React render per animation frame, however fast state messages arrive. */
  const pushState = (s: LiveState) => {
    pendingState.current = s;
    if (raf.current == null) raf.current = requestAnimationFrame(() => { raf.current = null; if (pendingState.current) setState(pendingState.current); });
  };

  const onMessage = (m: ServerMessage) => {
    switch (m.type) {
      case "ready":
        attempt.current = 0; setReconnecting(false); setLost(false);
        if (m.resumed) flash(`Reconnected - ${m.reps ?? 0} rep${m.reps === 1 ? "" : "s"} kept`, 3500);
        else setNotice((n) => (n?.startsWith("Connection lost") ? null : n));
        break;
      case "state": e2e.current = performance.now() - lastSend.current; pushState(m); setRec(null); break;
      case "recognition": setRec(m); break;
      case "exercise_selected":
        exerciseRef.current = m.exercise;
        setState((prev) => prev ? { ...prev, exercise: m.exercise, reps: 0, partial_reps: 0, progress: 0, feedback: [] } : null);
        setRec(null); flash(`Switched to ${m.exercise_name || m.exercise}`, 2500);
        break;
      case "target_reached": setNotice("Target reached - end the set when you're ready."); break;
      case "unsupported": case "error": setNotice(m.message); break;
      case "setup": if (m.message) flash(m.message, 6000); break;
      case "range_capture": setRangeMsg(m.status === "started" ? { ok: true, text: m.instruction ?? "" } : null); break;
      case "range_captured":
        setRangeMsg(m.ok ? { ok: true, text: `Range saved: ${Math.round(m.rest ?? 0)}° to ${Math.round(m.peak ?? 0)}°${m.clamped ? " (limited to a safe range)" : ""}` } : { ok: false, text: m.message ?? "Range capture failed" });
        setTimeout(() => setRangeMsg((r) => (r && r.text.startsWith("Range saved") ? null : r)), 5000);
        break;
      case "set_saved":
        if (m.reason === "switch") flash(m.saved ? `Previous set saved (${(m.summary as SetSummary | undefined)?.reps ?? 0} reps)` : "Previous set had nothing to save", 3500);
        else savedCb.current?.(m);
        break;
    }
  };

  const openSocket = () => {
    const p = params.current; if (!p) return;
    const q = new URLSearchParams({ exercise: p.exercise, target_reps: String(p.targetReps), session_id: String(p.sessionId), set_key: p.setKey, state_hz: String(STATE_HZ) });
    const s = new WebSocket(wsUrl(`/ws/live?${q}`));
    ws.current = s;
    s.onopen = () => { setOpen(true); s.send(JSON.stringify({ type: "auth", token: useAuth.getState().token ?? "" } satisfies ClientMessage)); };
    s.onmessage = (ev) => onMessage(JSON.parse(ev.data) as ServerMessage);
    s.onclose = (ev) => {
      if (ws.current !== s) return;
      setOpen(false);
      if (ending.current || ev.code === 4401 || ev.code === 4404) return;
      // dropped mid-set: the server parked the set; reconnect with the same key to resume it
      if (attempt.current < RETRY_MS.length) {
        setReconnecting(true); setNotice("Connection lost - reconnecting…");
        retryTimer.current = setTimeout(openSocket, RETRY_MS[attempt.current++]);
      } else {
        setReconnecting(false); setLost(true);
        setNotice("Couldn't reconnect. Press End set to save what was counted.");
      }
    };
  };

  const connect = useCallback((exercise: string, targetReps: number, sessionId: number) => {
    exerciseRef.current = exercise; ending.current = false; attempt.current = 0; setLost(false); setReconnecting(false);
    const setKey = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    params.current = { exercise, targetReps, sessionId, setKey };
    openSocket();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Switch exercise mid-set. The server saves the set in progress first, so counted reps are never lost. */
  const sendExercise = useCallback((exercise: string, weightKg = 0) => { send({ type: "select_exercise", exercise, weight_kg: weightKg }); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sendFrame = useCallback((lm: Landmarks, t: number) => {
    recording.current?.push({ t, landmarks: lm });
    if (send({ type: "frame", t, landmarks: lm })) lastSend.current = performance.now();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sendStats = useCallback((cameraFps: number, poseMs: number) => {
    send({ type: "client_stats", camera_fps: cameraFps, pose_ms: poseMs, e2e_ms: e2e.current });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Finish the set. Over the socket when it's up; otherwise (or if no reply in 5 s) over REST from the parked set. */
  const endSet = useCallback((weightKg: number) => new Promise<SavedMsg | null>((resolve) => {
    ending.current = true;
    if (retryTimer.current) clearTimeout(retryTimer.current);
    const key = params.current?.setKey;
    const viaRest = async () => {
      try { ws.current?.close(); } catch { /* closed */ }
      if (!key) return resolve(null);
      // give the server a moment to notice the drop and park the set
      for (const wait of [0, 400, 1200]) {
        await new Promise((r) => setTimeout(r, wait));
        try { return resolve(await api<SavedMsg>(`/api/live/sets/${key}/end`, { body: { weight_kg: weightKg } })); } catch { /* not parked yet */ }
      }
      resolve(null);
    };
    let done = false;
    savedCb.current = (m) => { if (!done) { done = true; resolve(m); } };
    if (!send({ type: "end_set", weight_kg: weightKg })) { done = true; return void viaRest(); }
    setTimeout(() => { if (!done) { done = true; void viaRest(); } }, 5000);
  }), []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Personal range capture: start, do 2-3 slow full reps, finish (or cancel). Applied immediately and remembered. */
  const sendRange = useCallback((action: "start" | "finish" | "cancel") => { send({ type: "range_capture", action }); }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  const close = useCallback(() => {
    ending.current = true;
    if (retryTimer.current) clearTimeout(retryTimer.current);
    if (raf.current != null) cancelAnimationFrame(raf.current);
    try { ws.current?.close(); } catch { /* already closed */ }
    ws.current = null;
  }, []);
  useEffect(() => close, [close]);
  /** Test hook: drop the socket the way a network failure would (used by the E2E reconnect test). */
  useEffect(() => { (window as unknown as { __kineticDropSocket?: () => void }).__kineticDropSocket = () => ws.current?.close(4000, "test drop"); }, []);
  return { state, rec, notice, open, reconnecting, lost, rangeMsg, connect, sendFrame, sendStats, sendExercise, sendRange, endSet, close, startRecording, stopRecording };
}
