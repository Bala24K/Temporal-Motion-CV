import { api } from "./api";
import type { Landmarks } from "./types";

export const MP: Record<string, number> = { nose: 0, left_shoulder: 11, right_shoulder: 12, left_elbow: 13, right_elbow: 14, left_wrist: 15, right_wrist: 16,
  left_hip: 23, right_hip: 24, left_knee: 25, right_knee: 26, left_ankle: 27, right_ankle: 28 };
export const BONES: [string, string][] = [["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"], ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"], ["left_hip", "right_hip"], ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"]];
/** Key carrying [width, height] of the source image (contract with motion_engine/skeleton.py IMAGE_KEY). */
export const IMAGE_KEY = "_image";

// Pinned versions: an unpinned "latest" model can change behaviour under us without any code change.
const TASKS_VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

export interface PoseStats { cameraFps: number; poseMs: number; width: number; height: number; delegate: string; cameraLabel: string }
/** World landmarks (approximate 3D, metres, hip-centred) travel as "w:<joint>": [X, Y, Z]. */
export const WORLD_PREFIX = "w:";
export interface PoseSource { stop: () => void; grab?: () => Landmarks | null; stats: () => PoseStats }
export type OnFrame = (lm: Landmarks | null, tSeconds: number) => void;

/** Draw landmarks inside the letterboxed video rectangle (object-contain), so the skeleton lines up with the picture. */
export function drawSkeleton(cv: HTMLCanvasElement, lm: Landmarks | null, aspect = 4 / 3) {
  const dpr = window.devicePixelRatio || 1, cw = cv.clientWidth, ch = cv.clientHeight;
  if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); }
  const c = cv.getContext("2d"); if (!c) return;
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, cw, ch); if (!lm) return;
  const rw = cw / ch > aspect ? ch * aspect : cw, rh = cw / ch > aspect ? ch : cw / aspect, ox = (cw - rw) / 2, oy = (ch - rh) / 2;
  const P = (k: string): [number, number] | null => { const v = lm[k]; return v && v[2] > 0.3 ? [ox + v[0] * rw, oy + v[1] * rh] : null; };
  c.lineWidth = 4; c.lineCap = "round"; c.strokeStyle = "rgba(111,146,255,.85)";
  for (const [a, b] of BONES) { const p = P(a), q = P(b); if (p && q) { c.beginPath(); c.moveTo(...p); c.lineTo(...q); c.stroke(); } }
  c.fillStyle = "#fff";
  for (const k of Object.keys(MP)) { const p = P(k); if (p) { c.beginPath(); c.arc(p[0], p[1], 4, 0, 7); c.fill(); } }
}

class Meter { private n = 0; private t0 = performance.now(); private ms = 0; fps = 0;
  tick(poseMs: number) { this.n++; this.ms = this.ms * 0.8 + poseMs * 0.2; const dt = performance.now() - this.t0; if (dt > 1000) { this.fps = (this.n * 1000) / dt; this.n = 0; this.t0 = performance.now(); } }
  get poseMs() { return this.ms; } }

/** Server-generated synthetic clip streamed in real time through the same path a camera uses.
 *  Demo frames already carry `_image: [1, 1]` (synthetic coordinates are isotropic). */
export async function startDemo(exercise: string, reps: number, fault: string, onFrame: OnFrame): Promise<PoseSource> {
  const d = await api<{ frames: Landmarks[]; timestamps: number[] }>(`/api/demo/sequence?exercise=${exercise === "auto" ? "squat" : exercise}&reps=${Math.max(1, reps)}${fault ? `&fault=${fault}` : ""}`);
  let i = 0, stopped = false; const t0 = performance.now(), m = new Meter();
  const tick = () => {
    if (stopped || i >= d.frames.length) return;
    const due = (performance.now() - t0) / 1000;
    while (i < d.frames.length && d.timestamps[i] <= due) { onFrame(d.frames[i], d.timestamps[i]); m.tick(0); i++; }
    requestAnimationFrame(tick);
  };
  tick();
  return { stop: () => { stopped = true; }, stats: () => ({ cameraFps: m.fps, poseMs: 0, width: 1, height: 1, delegate: "demo", cameraLabel: "demo" }) };
}

/** Standing demo body for the calibration wizard when no camera is available. */
export async function demoStage(stage: string): Promise<Landmarks[]> {
  return (await api<{ frames: Landmarks[] }>(`/api/demo/standing?n=40&stage=${stage}`)).frames;
}

let landmarker: any = null, delegate = "none";
async function getLandmarker() {
  if (landmarker) return landmarker;
  const V: any = await import(/* webpackIgnore: true */ `${TASKS_VISION}/vision_bundle.mjs`);
  const fs = await V.FilesetResolver.forVisionTasks(`${TASKS_VISION}/wasm`);
  for (const d of ["GPU", "CPU"]) {            // GPU delegate can fail on some drivers/browsers: fall back instead of breaking
    try {
      landmarker = await V.PoseLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: POSE_MODEL, delegate: d }, runningMode: "VIDEO", numPoses: 1 });
      delegate = d; return landmarker;
    } catch (e) { if (d === "CPU") throw e; }
  }
  return landmarker;
}

export async function startCamera(video: HTMLVideoElement, onFrame: OnFrame): Promise<PoseSource> {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: "user" }, audio: false });
  video.srcObject = stream; video.hidden = false; await video.play();
  const lmk = await getLandmarker();
  // The camera decides the real resolution (constraints are only ideals) - read it, never assume it.
  const cameraLabel = stream.getVideoTracks()[0]?.label || "camera";
  const toLm = (p: any[], w?: any[]): Landmarks => {
    const out: Landmarks = Object.fromEntries(Object.entries(MP).map(([k, i]) => [k, [p[i].x, p[i].y, p[i].visibility ?? 1]]));
    if (w) for (const [k, i] of Object.entries(MP)) out[WORLD_PREFIX + k] = [w[i].x, w[i].y, w[i].z];
    out[IMAGE_KEY] = [video.videoWidth, video.videoHeight];
    return out;
  };
  let stopped = false, lastMs = -1, latest: Landmarks | null = null; const m = new Meter();
  const rvfc = typeof (video as any).requestVideoFrameCallback === "function";
  const process = (captureMs: number) => {
    // MediaPipe VIDEO mode needs strictly increasing timestamps
    const ts = captureMs > lastMs ? captureMs : lastMs + 0.001; lastMs = ts;
    const t0 = performance.now();
    const res = lmk.detectForVideo(video, ts), p = res.landmarks?.[0];
    m.tick(performance.now() - t0);
    latest = p ? toLm(p, res.worldLandmarks?.[0]) : null;
    onFrame(latest, ts / 1000);
  };
  const loopRvfc = (now: number, meta: any) => {
    if (stopped) return;
    // capture time when the browser knows it (camera streams), else presentation time - both precede inference jitter
    process(typeof meta?.captureTime === "number" ? meta.captureTime : now);
    (video as any).requestVideoFrameCallback(loopRvfc);
  };
  let lastVideoTime = -1;
  const loopRaf = () => {
    if (stopped) return;
    if (video.currentTime !== lastVideoTime) { lastVideoTime = video.currentTime; process(performance.now()); }
    requestAnimationFrame(loopRaf);
  };
  if (rvfc) (video as any).requestVideoFrameCallback(loopRvfc); else loopRaf();
  return { stop: () => { stopped = true; stream.getTracks().forEach((t) => t.stop()); },
    grab: () => latest,                          // reuse the loop's result: a second detectForVideo doubled the cost
    stats: () => ({ cameraFps: m.fps, poseMs: m.poseMs, width: video.videoWidth, height: video.videoHeight, delegate, cameraLabel }) };
}
