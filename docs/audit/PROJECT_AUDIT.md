# Temporal Motion Analyzer — Project Audit

Audit date: 2026-11-18. Scope: the whole `TemporalMotionAnalyzer/` folder. No application code was changed.

**Path aliases used in all audit documents**

| Alias | Folder | What it is |
| --- | --- | --- |
| `RT/` | `Real-Time Human Motion Intelligence & Analysis System/` | **Current product** ("Kinetic"): FastAPI + Next.js + numpy `motion_engine`. Last edited 2026-11-06. |
| `V1/` | `Ver-1/` | Previous generation: desktop OpenCV app, QEVD pseudo-label classifiers, per-exercise state machines, biomechanics. `V1/app.py` last edited 2026-11-05. |
| `MU/` | `motion-upgrade/` | ST-GCN research code. An identical copy lives at `RT/research/motion_upgrade/` (verified with `diff -rq`). `motion-upgrade.zip` is the same code zipped. |
| — | `analysis/` | QEVD benchmark summary and confusion matrices (2026-09-19). |
| — | `QEVD-FIT-300k-Part-1/` | 76,000 `.mp4` clips (640×480 and 640×360, 30 fps, about 5 s each). No exercise labels and no subject IDs. |

**How to read the evidence tags**

- **[Reproduced]**: I ran it during this audit. Scripts are in `docs/audit-evidence/`.
- **[Measured]**: a number I measured during this audit, on this machine.
- **[Inspected]**: concluded from reading the code, not executed.
- **[Historical]**: a number taken from an existing report in the repo. It was not re-measured, and it is not validated accuracy.
- **[Hypothesis]**: plausible but unconfirmed. It needs a real-camera recording to settle.

---

## 1. Executive summary

1. **Three generations of the product sit side by side.** RT is the newest and most complete. V1 and MU are kept for reference. No one shared core library connects them: RT copied the 15-node skeleton idea and re-implemented everything in numpy.
2. **Every automated test passes, and every test uses synthetic or mocked data.** RT: 163 passed in 68 s. V1: 255 passed in 35 s [Reproduced]. No test has ever used a landmark sequence from a real camera. RT's own README says so: "The camera path is unverified… never run".
3. **On a real landscape webcam, calibration fails because of a coordinate bug, not a modelling limit.** The browser sends MediaPipe coordinates normalized separately by width and by height (`x/W`, `y/H`). The engine then treats them as if they were isotropic, so horizontal distances shrink by `H/W` (0.75 at 4:3, 0.5625 at 16:9). The front-pose gate needs shoulder-width / torso-length ≥ 0.55. On 179 real QEVD frames at 4:3 (full body, arms down), **0% pass with the coordinates the app uses, and 97% pass after aspect correction** [Reproduced, `pose_study.py`].
4. **Calibration barely affects analysis even when it succeeds.** Every rep and form signal in the nine exercise definitions is an angle, and angles do not depend on scale. The profile changes only (a) a torso scale that no current signal uses and (b) yaw compensation for two frontal-plane exercises. Rep thresholds (`rest`/`peak`) are textbook constants. `RepCounter.set_range()` exists but nothing calls it.
5. **Auto-detect cannot be reached from the main flow, and when it is reached it often stops at a dead end.** "Quick set (auto-detect)" starts a bicep-curl session, because `"auto"` is mapped to `"bicep_curl"` (`RT/web/components/live-workout.tsx:62`). Two of the seven recognition groups (curl/hammer curl, lateral/front raise) can never be told apart and always return `confirm`, and the UI has no confirm control. The learned ONNX recognizer is not shipped, so `rules` mode is always the one in use.
6. **Reps are not counted when the user doesn't return almost all the way to the textbook rest pose.** A curl that re-extends only to 150° counts 1 of 5 reps. At 140° it counts 0 of 5. A press that bottoms out at 110° instead of 90° counts 0 of 5 [Reproduced]. Real users commonly move like this.
7. **The UI the backend serves is stale.** `RT/web/out` was built on 2026-10-03. The Nov 6 source changes to exercise switching are not in it (no `select_exercise` string in the bundle).
8. **Historical classification results are agreement with a heuristic, not accuracy.** The QEVD labels come from a three-threshold rule in which `lateral_raise` is the fallback class for anything else. The "manual label review" with 83.6% agreement was **simulated by code** (`V1/recognition/label_review_tool.py:80-120`). No human looked at those clips. The best model (Random Forest, 64% accuracy, macro-F1 0.44) barely beats always predicting "squat" (59% of the test set). The ST-GCN scored 43% test accuracy against the same rule [Historical].

**Smallest reliable vertical slice.** Browser camera → aspect-corrected landmarks → guided calibration with quality score → *user-selected* exercise (squat side-view, bicep curl front-view) → personal range capture → rep counting with tracking-loss handling → save set → visible history. Auto-detect, nutrition, the model registry, the 3D body model and biomechanics are all deferred (see `REBUILD_ROADMAP.md`).

---

## 2. Languages, frameworks and dependencies

| Layer | RT (current) | V1 | MU |
| --- | --- | --- | --- |
| Language | Python 3.12+ (tested here on 3.13.2), TypeScript 5.7 | Python | Python |
| Backend | FastAPI, Starlette WebSocket, SQLAlchemy 2 (SQLite default, Postgres in compose), Pydantic 2, PyJWT, OpenTelemetry API/SDK | FastAPI inference API (`V1/api/`) | FastAPI (`MU/api/`) |
| Numerics/ML | numpy, onnxruntime (serving), onnx (export), optional PyTorch (`ml/torch_models.py`), scikit-learn/scipy (dev) | numpy, scikit-learn, XGBoost, joblib, PyTorch (1D-CNN) | PyTorch ST-GCN + transformer |
| Pose | **Browser** `@mediapipe/tasks-vision@0.10.14` from jsDelivr, `pose_landmarker_lite` (`latest`), GPU delegate | Python MediaPipe Tasks, `models/pose_landmarker.task` (lite, 5.8 MB) | Same as V1 |
| Frontend | Next.js 15 (static export), React 19, Tailwind 3, TanStack Query 5, Zustand 5, Framer Motion 12, React Hook Form + Zod, Recharts 2, three.js + R3F, lucide-react, cva | OpenCV window overlay (`V1/ui/dashboard.py`) | — |
| Infra | Dockerfile (multi-stage), docker-compose (+observability), GitHub Actions, Prometheus/Grafana/OTel configs, DVC stub | Dockerfile, compose | Dockerfile, compose |

The installed environment has torch 2.14 CPU and mediapipe 1.0.1. Note: `RT/docs/STACK.md` says "190+ pytest tests", but the suite has 163. `V1/README.md` says 216 tests, while the repo has 255 and `FINAL_VALIDATION_REPORT.md` says 218.

---

## 3. Architecture (RT)

```
Browser (web/)                                        Server (backend/)
──────────────                                        ─────────────────
getUserMedia 640×480 (ideal) ─► <video> (CSS mirrored)
MediaPipe PoseLandmarker lite, VIDEO mode, main thread
  toLm(): 13 landmarks → {name: [x/W, y/H, visibility]}   ◄── no aspect, no timestamp from camera
  drawSkeleton() on overlay <canvas>
  sendFrame() every processed frame ──WebSocket /ws/live?token=…──► reader(): ALL messages → LatestSlot (1 slot)
                                                                     consumer: LiveSession.push(frame, t)
                                                                       QualityTracker (13 or 8 joints, vis≥0.5, 2% margin)
                                                                       frame_to_array → normalize(profile, yaw if frontal)
                                                                       signals (primitives) → RepCounter (EMA α=0.5, hysteresis)
                                                                       per completed rep: evaluate_rep (form rules) + Baseline
                                                                     ◄── {"type":"state", reps, phase, progress, feedback…} per frame
                                                                     auto mode: buffer 60–180 frames, recognize() every 1 s (sync)
end_set ──────────────────────────────────────────────────────────► save_set + store_baseline (SQLite/Postgres)

REST: /api/auth, /api/profile (calibration), /api/exercises, /api/workouts, /api/sessions, /api/progress,
      /api/nutrition, /api/admin (registry, requests, dataset ingest), /api/demo/*, /api/metrics, /api/health
Static: web/out mounted at "/" (stale build), legacy SPA at /legacy
```

### 3.1 Entry points

| Entry point | Purpose | State |
| --- | --- | --- |
| `uvicorn backend.main:app` (`RT/backend/main.py`) | API, WebSocket, serves `web/out` | Runs; serves the stale UI |
| `cd RT/web && npm run dev` | Next.js dev server (port 3000, needs `NEXT_PUBLIC_API_URL`) | Source is current; not run during this audit |
| `python -m ml.train_recognizer out/recognizer.onnx` | Trains the synthetic ONNX recognizer | Produces a model trained only on synthetic data |
| `python -m ml.train_encoder` | PyTorch SupCon encoder | Trained on a 54-clip synthetic dataset (`RT/reports/encoder_metrics.json`) |
| `python scripts/e2e_web.py URL` | Playwright E2E | Needs Chromium. Not run here |
| `python V1/app.py` | V1 desktop OpenCV app (camera, 4 exercises, AUTO classifier) | Not run here (no camera) |
| `python V1/webcam_recognizer.py`, `V1/api/main.py` | Older demo / inference API | Superseded |
| `python -m motion.live --profile` (MU) | Threaded capture/pose/inference profiler | Superseded |

### 3.2 Runtime execution flow (the real path)

1. `/live/?workout=ID` or `?quick=1` → `RT/web/app/live/page.tsx`: `POST /api/sessions` creates a session and builds a queue of sets. In Quick set mode the queue item is `exercise_id: "auto"`.
2. `LiveWorkout` → `SetScreen` (`live-workout.tsx:56`). The initial `exerciseId` is the item's id, **but `"auto"` becomes `"bicep_curl"`** (line 62).
3. **Start** → `useLiveSet.connect(exerciseId, …)` opens the WebSocket, then `startCamera()` (`RT/web/lib/pose.ts:51`) or `startDemo()`.
4. Each video frame: `detectForVideo` runs on the main thread, then `drawSkeleton`, then `sendFrame` (JSON, about 1 KB).
5. Server `live()` (`RT/backend/routers/live.py:54`): authenticates from the `?token=` query, loads the profile and baseline, and builds a `LiveSession`. It processes the latest frame and replies with `state` after each one.
6. **End set** → `end_set` → `_finish` → `save_set(source="ai")`. The set is saved only if `reps or partial_reps` is non-zero **and** `session_id` is set.
7. Session finish → `POST /api/sessions/{id}/finish` → summary screen.

### 3.3 Landmark data structures and coordinate conventions

| Stage | Structure | Coordinates | Notes |
| --- | --- | --- | --- |
| Browser `toLm` | `{name: [x, y, visibility]}` for 13 MediaPipe landmarks (0, 11–16, 23–28) | `x∈[0,1]` of width, `y∈[0,1]` of height, y down | `z`, `presence` and **`worldLandmarks` are discarded**. Image width/height are not sent. |
| `frame_to_array` (`RT/motion_engine/skeleton.py:32`) | `(15, 3)`: 13 real joints + `hip_center` + `shoulder_center` (midpoints) | Same | Joints below visibility 0.3 → NaN xy. A midpoint is NaN if either side is NaN. |
| `normalize` (`skeleton.py:167`) | `(T, 15, 3)` | Hip-centred, divided by torso length. When calibrated, this is the profile's fixed torso length from calibration time. | Optional x / cos(yaw) for frontal-plane exercises. |
| Primitives (`RT/motion_engine/primitives.py`) | `(T,)` signals | Degrees (angles) or torso units | All rep/form signals in the 9 definitions are angles. |
| Synthetic generator (`RT/motion_engine/synthetic.py`) | Same dict format | **Isotropic** (square-pixel) | This is why tests never see the aspect bug. |
| V1 `PoseFrame` (`V1/pose/landmarks.py`) | dataclass with x, y, z, visibility | Same normalized convention, 1280×720 capture | V1 normalization is also not aspect-corrected [Inspected]. |

**Left/right convention.** MediaPipe labels landmarks by the subject's anatomical side. The video and canvas are mirrored together with CSS (`scaleX(-1)`), so the overlay lines up with the picture. The engine never mirrors coordinates. This part is consistent [Inspected].

### 3.4 Persistence and user profiles

SQLite `RT/fitness.db` by default (`RT/backend/config.py:8`). Tables: `users`, `body_profiles` (JSON blob), `baselines`, `workouts`, `workout_items`, `workout_sessions`, `sets`, `reps`, `unsupported_requests`, `custom_exercises`, `model_versions`, `food_logs`.

Local DB contents, read-only [Measured]: 2 users, 3 workout sessions, **0 sets**, 0 baselines. The one body profile is exactly the synthetic demo body (torso 0.22, shoulder 0.13, `cm_per_torso` 0). Whatever happened in this database, no real-camera calibration or AI-counted set was ever saved.

The calibration profile stores no camera metadata (resolution, aspect, device, distance or framing), so it cannot be invalidated when the camera moves.

### 3.5 Frontend state management and API integration

- Server state: TanStack Query (`queryKey`s per page). Auth: Zustand store (`RT/web/lib/api.ts`) with the JWT in `localStorage`. The WebSocket token goes in the URL query string.
- Live set state: local React state in `useLiveSet` (`RT/web/hooks/useLiveSet.ts`). `setState` runs on every WebSocket message (about 30 Hz), which re-renders the whole `SetScreen` [Inspected].
- The supported exercise list is **hard-coded** in `live-workout.tsx:15` instead of coming from `/api/exercises`, so custom definitions added through the admin console never appear in the live screen.
- Calibration instructions are duplicated in `calibration.tsx:61` and `RT/motion_engine/calibration.py:27`.

---

## 4. Component status by subsystem (summary; details in the topic documents)

| Subsystem | Status | Primary document |
| --- | --- | --- |
| Camera acquisition (browser) | Written, never exercised by the authors; no aspect/timestamp metadata | `POSE_CALIBRATION_AUDIT.md` |
| Pose estimation | MediaPipe lite, defaults, main thread; no smoothing apart from MediaPipe's own | `POSE_CALIBRATION_AUDIT.md` |
| Calibration | **Broken on real landscape cameras**; works on demo body | `POSE_CALIBRATION_AUDIT.md` |
| Feature extraction (primitives) | Correct on isotropic input; distorted on real input | `POSE_CALIBRATION_AUDIT.md`, `REP_COUNTING_AUDIT.md` |
| Exercise classification | Rules-only in production; auto unreachable / dead-ends; V1 learned models trained on pseudo-labels | `CLASSIFICATION_AUDIT.md` |
| Repetition counting | Works on clean synthetic; misses reps on incomplete return; no tracking-loss handling | `REP_COUNTING_AUDIT.md` |
| Form analysis | Interpretable rule evidence (good design); thresholds unvalidated; false positives from geometry | `REP_COUNTING_AUDIT.md` |
| Workout/session management | API sound; history UI missing; edit UI missing | `FEATURE_INVENTORY.md` |
| Frontend | Functional, generic; stale build served | `FRONTEND_DESIGN_AUDIT.md` |
| Tests | Many, all synthetic | `TEST_AND_EVALUATION_PLAN.md` |

---

## 5. Existing tests and evaluation datasets

| Suite | Result [Reproduced] | What it proves | What it cannot prove |
| --- | --- | --- | --- |
| `RT/tests` (11 files, 163 tests) | 163 passed, 68 s | Engine logic on synthetic poses, API flows, calibration staging logic, uncertainty maths, data-pipeline invariants, infra configs, ONNX parity | Anything about real cameras, real bodies, real lighting or occlusion |
| `V1/tests` (40 files) | 255 passed, 35 s | State machines on hand-built angle traces, biomechanics maths, leakage guards, API | Real rep accuracy, classifier validity |
| `MU/tests` | Not run (superseded copy) | — | — |
| `RT/scripts/e2e_web.py` (Playwright) | Not run here (no Chromium) | UI wiring with demo clips | Camera path |

| Dataset | Labels | Subject IDs | Usable for |
| --- | --- | --- | --- |
| QEVD-FIT-300k Part 1 (76,000 clips) | **None shipped**; V1 assigned heuristic labels to 257 clips | None | Pose-robustness studies, unlabeled pre-training, *candidate* clips for human annotation |
| `V1/data/qevd_pose_cache.pkl` | Heuristic labels | None | Re-running historical baselines only |
| `RT/data/dataset/` | Empty (`clips/`, `versions/` empty) | — | Intended store for human-labelled data |
| Synthetic generator | Exact by construction | Seeds/scales stand in for "subjects" | Unit tests and pipeline plumbing only |

---

## 6. Historical results, restated honestly

None of these numbers are validated exercise-recognition accuracy. All of them measure agreement with a deterministic kinematic rule (`V1/recognition/qevd_dataset.py:84-91`):
`knee_rom > 20° → squat`, else `wrist_above_ratio > 0.2 → shoulder_press`, else `elbow_rom > 25° → bicep_curl`, **else `lateral_raise`**. So the `lateral_raise` class means "anything else".

| Source | Claim | What it actually is |
| --- | --- | --- |
| `V1/README.md`, `V1/analysis/FINAL_VALIDATION_REPORT.md` | RF 75.0 ± 10.5% (5 seeds, 12 test clips/seed); LR 66.7% in README vs 78.3% in report | Agreement with the rule on a *re-balanced* 100-clip subset. The two documents disagree. 12 test clips per seed. |
| `analysis/qevd_benchmark_summary.json` | RF 64.3% acc / 0.435 macro-F1; XGB 62.2%; 1D-CNN 53.0%; MLP 58.9%; ST-GCN 51.4% | Agreement with the rule on an imbalanced set. The summed confusion matrix shows "squat" is 110/185 = **59.5%**, so RF is about 5 points above always predicting the majority class, and the 1D-CNN and ST-GCN are below it. |
| `MU/models/motion_stgcn_results.json` | ST-GCN test accuracy 43.2%, macro-F1 0.30, bicep F1 0.0; learned rep count mean 0.0 | 37 test clips, rule labels (`trusted_label_fraction: 0.0`). `release_ready: false`. |
| `FINAL_VALIDATION_REPORT.md §4` | "Manual label review … heuristic agreement rate 83.6%" | **Not manual.** `run_batch_simulation_review()` marks a clip "correct" when it clears slightly relaxed versions of the same thresholds. This measures nothing about label validity. |
| `RT/README.md` | OOD flagged 7/7 novel movements, conformal coverage 0.91, encoder 77.8% subject-disjoint | Synthetic generator data; 9 test clips. Mechanism checks only, as the README itself says. |
| `RT/README.md` | Lateral raise ROM at 50° yaw: 0.72 uncorrected vs 0.915 corrected | Synthetic orthographic projection. Valid as a unit test of the formula. |

---

## 7. Performance and reliability

### 7.1 Measurements taken during this audit

| Quantity | Value | Conditions |
| --- | --- | --- |
| MediaPipe Pose **lite** inference | p50 **14.2 ms**, p95 26.6 ms (n = 3,595 frames) [Measured] | Python Tasks API, CPU/XNNPACK, this machine, 640×480/640×360 QEVD frames |
| MediaPipe detection rate | 93.5% of frames [Measured] | 40 QEVD clips × 90 frames |
| Landmark visibility | Median 0.97; median of per-frame minimum 0.51 [Measured] | Python API returns real visibility values. Browser values not verified. |
| Frames with all 13 landmarks at vis ≥ 0.5 inside a 2% margin | 43.8% [Measured] | QEVD framing (many crop the feet or hands) |
| `LiveSession.push` (server, per frame) | p50 0.18 ms, p95 0.41 ms [Measured] | Synthetic squat, this machine |
| `recognize()` (rules, 180 frames, 9 definitions) | p50 13.6 ms, p95 15.8 ms [Measured] | Runs **synchronously on the asyncio event loop**, once per second per auto-mode connection |
| Browser camera FPS, browser pose latency, end-to-end latency, browser memory | **Not measured.** No camera in this environment, and the authors also never ran the camera path. | The client reports `client_pose_ms`/`client_e2e_ms` to `/api/metrics`, but no data exists yet. |
| CPU/GPU usage, long-session stability | **Not measured** | — |

### 7.2 Blocking operations, stale frames, leaks, races

| # | Issue | Location | Effect | Evidence |
| --- | --- | --- | --- | --- |
| P1 | **Control messages share the latest-wins frame slot.** `select_exercise` followed by any frame before the consumer runs: the select is silently dropped. | `RT/backend/routers/live.py:101-104`, `RT/motion_engine/runtime.py` | Exercise switch is ignored at random; UI shows "Switched to…" only if the server processes it | [Reproduced] consumer received `frame`, `dropped=1` |
| P2 | Pose inference runs on the **main thread** in `requestVideoFrameCallback` | `RT/web/lib/pose.ts:62-71` | Inference and React rendering compete; jank under load | [Inspected] |
| P3 | **Duplicate inference during calibration.** `grab()` calls `detectForVideo` while the loop also runs it | `pose.ts:74`, `calibration.tsx:35` | Twice the GPU/CPU load during calibration; possible timestamp collisions on the shared landmarker | [Inspected] |
| P4 | Timestamps are processing time (`performance.now()`), not capture time (`requestVideoFrameCallback` metadata `mediaTime`) | `pose.ts:68` | Tempo and rep duration include inference jitter | [Inspected] |
| P5 | One HTTP POST per calibration frame (`stage-check`), every 300 ms | `calibration.tsx:36` | Slow capture; fails offline. Frames are spread over 6+ s and not checked for stillness. | [Inspected] |
| P6 | Rule recognition (about 14 ms) runs synchronously on the event loop | `live.py:148` | Blocks every connection on that worker for about 14 ms/s per auto-mode user | [Measured] |
| P7 | Server sends one `state` message per processed frame; client `setState` per message | `live.py:163`, `useLiveSet.ts:28` | 30 Hz React re-renders including Framer Motion | [Inspected] |
| P8 | JWT in the WebSocket query string | `useLiveSet.ts:22` | Tokens appear in proxy and access logs (the app's own middleware avoids logging the query) | [Inspected] |
| P9 | No reconnect or resume on WebSocket drop; `onclose` only sets `open=false` | `useLiveSet.ts:25` | Mid-set disconnect loses the set silently | [Inspected] |
| P10 | V1 camera `read()` returns the latest frame without a frame id, so the main loop can run the detector on the same frame twice; detector timestamps come from wall-clock `time.time()` | `V1/camera/sources.py:126`, `V1/pose/detector.py:56` | Wasted CPU; non-monotonic risk if the clock changes | [Inspected] |
| P11 | Unbounded growth: none found on the server (`deque` and `history` are capped; metrics capped at 500). `LiveSession.reps` grows per set, which is fine. | — | — | [Inspected] |

---

## 8. Scientific validity versus software correctness

| Question | Software correctness | Scientific validity |
| --- | --- | --- |
| Is the rep counter implemented as designed? | Yes, on its design inputs [Reproduced] | Unknown. No human-counted reference set exists. |
| Is calibration logic implemented as designed? | Yes on isotropic input; **wrong** on real camera input (coordinate contract bug) | Body ratios from 2D image landmarks are approximate proportions, not anatomy. No metric claim is supportable without a known-size reference or a calibrated camera. |
| Does auto-recognition work? | Decision logic is coherent; product wiring is broken | Never measured on real labelled data |
| Are form warnings correct? | Rules evaluate as written | Thresholds are uncited guesses. No expert-reviewed examples exist. |
| Are the QEVD numbers accuracy? | The pipelines run | **No.** They measure agreement with a heuristic, and the "manual review" was simulated. |

---

## 9. Retain, repair, replace, defer

| Decision | Components |
| --- | --- |
| **Retain** (sound design, keep) | `ExerciseDefinition`-as-data + validating registry (`RT/exercise_system`); primitive vocabulary; interpretable `Evidence` form model; explicit `Recognition` statuses incl. `ambiguous`/`unsupported`; equivalence groups (honest about indistinguishable exercises); latest-wins idea for *frames*; dataset store with consent, content hashing, subject-disjoint splits and leakage assertion (`RT/ml/data`); model registry gate that refuses non-human labels; synthetic generator *as a unit-test tool*; V1 `RepStateMachine` ideas (time-based dwell, tracking-loss "degraded" state, re-anchoring) |
| **Repair** | Browser→engine coordinate contract (aspect, timestamps, world landmarks, metadata); calibration protocol and persistence; `RepCounter` (personal range, tracking loss, rest drift); WebSocket protocol (separate control channel, acks, reconnect); live UI wiring (auto, confirm, exercise list from API); stale build; history and workout-edit UI |
| **Replace** | Calibration front gate based on a fixed ratio threshold; front-view sagittal definitions (squat-from-front, curl-from-front without depth); V1 forced-choice classifier and its pseudo-label training set; "calibrated display score" cap (`V1/recognition/fitness_pipeline.py:139`), which is cosmetic |
| **Defer** | Auto-detect in the main flow; learned recognizer/encoder; ST-GCN; nutrition; admin model registry/Label Studio/DVC/MLflow; OIDC/PKCE; observability stack; 3D body model viewer; V1 biomechanics (inverse dynamics, muscle optimisation and muscle estimation are unvalidated); RTMPose/YOLO adapters |

---

## 10. Ten most serious defects

| # | Defect | Severity | Evidence |
| --- | --- | --- | --- |
| 1 | Landmarks are anisotropic (x/W, y/H) but treated as isotropic everywhere: calibration gate fails on landscape cameras and angles are distorted (true 45° reads 36.9° at 4:3, 29.4° at 16:9) | Critical | [Reproduced] real footage + synthetic |
| 2 | Calibration does not personalize analysis: thresholds are constants, `set_range()` is never called, and the profile affects only an unused scale plus yaw | Critical (product promise) | [Inspected] |
| 3 | Rep completion requires return to within 15% of the textbook rest pose; incomplete re-extension is silently uncounted (0/5 reps at 140° curl rest, 110° press bottom) | Critical | [Reproduced] |
| 4 | "Quick set (auto-detect)" starts a bicep-curl session (`"auto"` → `"bicep_curl"`) | High | [Inspected] `live-workout.tsx:62` |
| 5 | Auto-recognition `confirm` has no UI action; 2 of 7 groups always confirm | High | [Reproduced] + [Inspected] |
| 6 | `select_exercise` can be silently dropped by the latest-wins slot; switching also discards the reps already counted | High | [Reproduced] |
| 7 | The FastAPI server serves a stale build (`web/out`, 2026-10-03) without the Nov 6 changes | High | [Inspected] |
| 8 | Staged calibration returns `"calibrated": true` while the stored profile can fail `BodyProfile.calibrated` (front-only: quality 0.50 < threshold), so the UI and engine disagree | Medium-High | [Reproduced] |
| 9 | Sagittal exercises are advertised for front view (squat, curls) and both-side joints are required in side view, so foreshortening and occlusion go unmanaged. Squat at 16:9 raises false `rom` + `tempo` warnings on a clean synthetic set. | High | [Reproduced] + [Inspected] |
| 10 | Historical evaluation presented as validation: simulated "manual review", README/report numbers disagree, models at or below majority baseline | High (credibility) | [Inspected] |

---

## 11. Reproducing the audit evidence

Scripts are in `docs/audit-evidence/`. Neither script writes to the project.

```bash
cd "Real-Time Human Motion Intelligence & Analysis System"
python ../docs/audit-evidence/verify_rt.py          # synthetic reproductions (sections 1-7)
python ../docs/audit-evidence/pose_study.py         # needs mediapipe + V1/models/pose_landmarker.task + QEVD clips
python -m pytest -q                                 # 163 passed
cd ../Ver-1 && python -m pytest -q                  # 255 passed
```
