# Kinetic — movement intelligence platform

Camera-based rep counting, form feedback, personal calibration, workouts, progress and nutrition. Built on one idea:
**exercises are definitions, movement understanding is infrastructure, models are replaceable.**

```
browser camera ─► pose (MediaPipe, on-device) ─► landmarks ──WebSocket──► motion_engine
                                                                          skeleton + BodyProfile (calibrated, yaw-compensated)
                                                                          primitives ─► ExerciseDefinition ─► rep / phase / form
                                                                          recognition: rules ⟂ learned (ONNX)  → agree? analyse : ask
                                                                          uncertainty: Mahalanobis OOD + conformal sets
backend (FastAPI) ─ workouts · progress · nutrition · admin/model registry · dataset ingest      web/ (Next.js) ─ the product UI
ml/ ─ dataset store · subject-disjoint splits · Label Studio · PyTorch encoder (SupCon) · ONNX export · run tracking
```

## Run
```bash
pip install -r requirements-dev.txt
python -m pytest -q                                     # 163 tests (torch tests auto-skip without torch)
ADMIN_EMAILS=you@example.com uvicorn backend.main:app   # http://localhost:8000  (serves web/out; old SPA at /legacy/)
# rebuild the UI:  cd web && npm ci && npm run build      dev: npm run dev (NEXT_PUBLIC_API_URL=http://localhost:8000)
# full stack + observability: SECRET_KEY=... docker compose -f docker-compose.yml -f docker-compose.observability.yml up --build
python scripts/e2e_web.py http://localhost:8000          # browser test (needs `playwright install chromium`)
```
No camera? Choose **Demo clip** in the live screen (injectable faults: shallow, too fast, elbow drift, torso lean) or **Demo body** in calibration.
Optional: `python -m ml.train_recognizer out/recognizer.onnx` then `RECOGNIZER_ONNX=out/recognizer.onnx RECOGNIZER_MODE=consensus`.

## Landmark frame contract (since M1)
Pose models return x as a fraction of image *width* and y as a fraction of *height*. Every frame sent to the engine
must therefore carry `"_image": [width, height]`; `frame_to_array` converts to isotropic units exactly once, and the
API rejects frames without it (`motion_engine/contract.py`). Before this, a 45-degree elbow read 29 degrees on a 16:9
camera and calibration rejected real people facing a landscape webcam (see `../docs/POSE_CALIBRATION_AUDIT.md`).
Synthetic/demo frames declare `[1, 1]`.

- **Recorder:** tick "Record landmarks" on the live screen to download the exact frames the engine received (no video).
- **Replay:** `python -m ml.replay recording.jsonl [--exercise squat] [--expect-reps 10]` - deterministic, no camera.
- **Stale UI check:** `/api/health` reports `ui.stale` when `web/out` was built from older sources than `web/`.
- **Windows:** the `&` in this folder's name breaks npm's `.bin` shims; run tools via node:
  `node node_modules/typescript/bin/tsc --noEmit` and `node node_modules/next/dist/bin/next build`.

## What is in the repo
| Path | Purpose |
| --- | --- |
| `motion_engine/` | numpy engine: skeleton/profile, **staged calibration**, primitives, rep counter, form evidence, recognition, **uncertainty** (OOD/conformal/prototypes), **pose-estimator interface + benchmark**, ONNX recognizer, latest-wins runtime |
| `exercise_system/` | `ExerciseDefinition` schema (incl. `motion_plane`), validating registry, 9 JSON definitions |
| `backend/` | auth (local + OIDC), library, unsupported-exercise pipeline, calibration, workouts/sessions, progress (e1RM, PRs, trends, goals), **nutrition**, admin (requests, registry + lineage + gate, dataset ingest jobs, Label Studio import), `/ws/live`, OTel, metrics |
| `nutrition/` | energy *ranges* (MET), Mifflin–St Jeor targets with a safety floor, food table |
| `ml/` | `data/` (schemas, versioned store, splits + leakage guard, report, ingestion, Label Studio), `torch_models.py` + `train_encoder.py` (PyTorch), `train_recognizer.py` (ONNX baseline), `benchmark_pose.py`, `tracking.py` |
| `web/` | Next.js 15 + TypeScript + Tailwind app: Home, Workout builder, **Live**, Exercises, Progress, Nutrition, Profile (calibration wizard + 3D body model), Console |
| `infrastructure/` | Prometheus, alerts, Grafana dashboard, OTel collector |
| `research/motion_upgrade/` | your original ST-GCN code, untouched |

## What changed in this iteration, and the evidence
- **Calibration is now a real onboarding flow** (front → side → arms-up), validated per stage, with height-based physical scale, a plausibility score,
  and a 3D view of *your* proportions vs average. *(The first iteration only had a front-pose torso scale and rep baselines.)*
- **Camera-angle compensation** for frontal-plane exercises. Measured: at a 50° camera yaw, a lateral raise read 0.72 ROM uncorrected (false "raise higher")
  vs 0.915 corrected (0.916 straight-on); an uncorrected shoulder press **falsely passed** a shallow rep. Sagittal exercises are *not* corrected — their
  foreshortening needs depth, which 2D cannot give.
- **The confidently-wrong recognizer is fixed at the decision rule, not by threshold tuning.** Softmax scored a Y-raise and jumping jack at 1.00 as lateral/front raise.
  Mahalanobis OOD flagged 7/7 hand-built novel movements while rejecting 2.8% of held-out in-distribution windows; conformal coverage 0.91 (target 0.90). Regression tests lock this in.
- **Data pipeline**: content-hashed dataset versions (relabelling = new version), consent enforced, raw video deleted after landmark extraction, subject-disjoint splits with a leakage assertion,
  dataset report that says when a dataset is *not* trustworthy, Label Studio pre-annotation (engine proposes reps) and import.
- **PyTorch encoder** (adaptive-adjacency ST-GCN → temporal transformer → unit embedding, SupCon). Trains; exported ONNX matches torch to 7e-8.
- **Registry**: lineage (commit, config, pose model, feature version, evaluation dataset), stricter gate (OOD AUROC ≥ 0.90, conformal coverage ≥ 0.85, ≤ 10% in-distribution rejection, human-verified labels,
  no regression), promotion record, `why` endpoint, register-from-run. Additive schema migration keeps existing databases working.
- **Measured**: server frame processing 0.54 ms p50 / 0.76 ms p95 (744 frames, localhost); send→feedback 4 ms p50.

## Honest limits — read before trusting any number
1. **There is still no real data.** Every threshold, the OOD detector, the conformal calibration and the encoder were fit or checked on the *synthetic generator*
   (no foreshortening, occlusion, clothing, lighting). The 100% synthetic accuracy and the OOD results show the **mechanism** works; they say nothing about real footage.
   The registry gate refuses to promote anything without human-verified labels, so this can't be shipped by accident.
2. **The camera path is unverified.** Browser MediaPipe (`web/lib/pose.ts`) was written but never run — no camera or CDN here. Demo mode exercises everything downstream of landmarks.
3. **Adapters not run**: RTMPose, YOLO-Pose, MediaPipe-on-video ingestion, Label Studio/DVC/MLflow servers, Postgres, Docker build, the Grafana/OTel containers, the browser SSO redirect flow.
   Pure parts (landmark conversion, benchmark, converters, config) are tested.
4. **2D limits**: a forward hinge and a sideways bend look alike from the front; deadlift-family recognition needs the side view. Front-view curls are foreshortened.
5. **Nutrition and calories** are formula estimates shown as ranges, general guidance only — computer vision cannot measure energy expenditure.
6. **Auth** is prototype-grade locally (stdlib JWT + PBKDF2). OIDC bearer verification is implemented and tested; put an IdP in front before real users.
7. `web/out` is committed to the zip for convenience; it is a build artifact.

## Next, in order
1. Collect subject-disjoint, human-annotated clips (`ml.data.cli ingest` → Label Studio → `ls-import`) until `/api/admin/dataset/report` says `gate_ready`.
2. Run `python -m ml.benchmark_pose` on that footage before changing pose models.
3. Retrain the encoder on it; compare against the ONNX baseline through the gate; try consensus → learned in `shadow` mode first.
4. Re-tune definitions on real footage; add a real "none of the above" negative class.
5. Then: PKCE login, WebRTC only if server-side video is ever needed, deployment IaC once there is a target.
See `docs/STACK.md` for the full adopted/deferred table with triggers.
