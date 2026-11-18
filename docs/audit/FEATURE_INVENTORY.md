# Feature Inventory

Aliases (`RT/`, `V1/`, `MU/`) and evidence tags are defined in `PROJECT_AUDIT.md`.

**Status values:** Verified working · Partially working · Broken · Mocked or hardcoded · UI-only · Not implemented.
"Verified working" means verified by passing automated tests on this machine or by reproduction. A feature never qualifies just because UI for it exists. Each row also states its **evidence**, because "verified" here is never "verified with a real camera". No feature has been verified with a real camera.

---

## 1. Summary table (RT, the current product)

| # | Feature | Status | Evidence | Primary code |
| --- | --- | --- | --- | --- |
| F1 | Registration / login / logout (local JWT + PBKDF2) | Verified working | `tests/test_api.py::test_auth_flow_and_protection` | `RT/backend/routers/auth.py`, `security.py` |
| F2 | Account deletion (erases data) | Verified working | `test_account_deletion_removes_data` | `auth.py:93` |
| F3 | User goal + weekly target | Verified working (API); UI wired | API tests; UI inspected | `auth.py:78`, `web/app/profile/page.tsx` |
| F4 | Body metrics (height for calibration; sex/age/weight/activity for nutrition) | Partially working | Inspected | `profile.py`, `nutrition.py:26` |
| F5 | Camera setup / framing guidance | **Broken** | Reproduced (RC-1) + inspected | `web/lib/pose.ts`, `quality.py` |
| F6 | Staged body calibration (front/side/arms up) | **Broken** on real landscape cameras; works with Demo body | Reproduced on real QEVD frames | `calibration.py`, `calibration.tsx` |
| F7 | Calibration persistence | Partially working | Reproduced (calibrated-flag mismatch) | `profile.py:62-82`, `db.BodyProfileRow` |
| F8 | 2D skeleton overlay | Partially working | Inspected (geometry correct); never run with a camera | `pose.ts:14-25` |
| F9 | 3D body-proportion viewer | Partially working | Renders per authors' Playwright run (not re-run); needs height; distorted inputs | `components/skeleton-3d.tsx` |
| F10 | Exercise auto-detect | **Broken** (product flow) | Reproduced + inspected | `recognition.py`, `live.py:142-160`, `live-workout.tsx:62` |
| F11 | Learned recognizer (ONNX, OOD, conformal) | Not implemented in product (code exists, no model shipped, not enabled) | Inspected | `learned.py`, `recognizers.py` |
| F12 | Rep counting (manual exercise selection) | Partially working | Reproduced | `repetition.py`, `session.py` |
| F13 | Partial-rep detection | Partially working | Reproduced | `repetition.py:104-110` |
| F14 | Form analysis & feedback | Partially working | Reproduced (rules evaluate; false positives under distortion) | `form.py`, definitions |
| F15 | ROM metric | Partially working | Inspected | `session.summary`, `progress.py` |
| F16 | Left/right symmetry | Partially working | Inspected | `form.py:79-83` |
| F17 | Personal baselines & insights | Partially working | Inspected | `personalization.py`, `form.py:97` |
| F18 | Exercise library (9 definitions, filters, detail) | Verified working | `test_exercise_library_and_filters` | `exercises.py`, `web/app/exercises/page.tsx` |
| F19 | Unsupported-exercise check & request logging | Verified working | `test_unsupported_exercise_is_honest_and_logged_for_developers` | `exercises.py:75` |
| F20 | Custom exercise definitions via admin (no redeploy) | Partially working | API test passes; **never appear in live screen** (hard-coded list) | `admin.py:146`, `live-workout.tsx:15` |
| F21 | Workout builder | Partially working | Inspected | `web/app/workout/page.tsx`, `sessions.py` |
| F22 | Live workout screen | Partially working | Inspected | `live-workout.tsx`, `useLiveSet.ts`, `live.py` |
| F23 | Exercise switching during a set | **Broken** | Reproduced (race) + inspected; **absent from served build** | `live.py:122-136`, `live-workout.tsx:92-105` |
| F24 | Rest timer between sets | Partially working | Inspected only | `live-workout.tsx:46-54` |
| F25 | Manual set logging (unsupported exercises) | Verified working (API) | API tests | `sessions.py:166` |
| F26 | Set saving from AI analysis | Partially working | Inspected | `live.py:177-197` |
| F27 | Session summary screen | Partially working | Inspected | `web/app/live/page.tsx:32-38` |
| F28 | Workout history (list past sessions) | **Not implemented** (UI); API exists and is unused | Inspected | `sessions.py:190` (`GET /api/sessions`) |
| F29 | Workout editing | **Not implemented** (UI); API exists | Inspected | `sessions.py:78` (`PUT /api/workouts/{id}`) |
| F30 | Goals & streaks | Partially working | Inspected | `progress.py:219-262` |
| F31 | Progress analytics (volume, reps, e1RM, trends, muscle split) | Partially working | Inspected | `progress.py`, `web/app/progress/page.tsx` |
| F32 | Offline clip analysis | Not implemented (UI); API exists | Inspected | `sessions.py:227` |
| F33 | Demo mode (synthetic clip / demo body) | Verified working (synthetic only) | Engine tests | `live.py:30-51`, `synthetic.py` |
| F34 | Nutrition (targets, food log, energy ranges) | Verified working (API) | `test_platform.py` nutrition tests | `nutrition/`, `nutrition.py` |
| F35 | Admin console: requests triage, model registry, gate, lineage | Partially working | API tests; promotion gate needs data that doesn't exist | `admin.py`, `web/app/admin/page.tsx` |
| F36 | Dataset ingest (video → landmarks, server-side) | **Broken by default** (needs `MEDIAPIPE_POSE_MODEL`; not run) | Inspected | `admin.py:305`, `motion_engine/pose.py:63` |
| F37 | Label Studio bridge, DVC, run tracking | Partially working (converters tested; servers never run) | Tests | `ml/data/labelstudio.py`, `dvc.yaml`, `ml/tracking.py` |
| F38 | OIDC bearer verification | Verified working (unit level); no browser SSO flow | `test_platform.py` OIDC test | `backend/oidc.py` |
| F39 | Metrics / telemetry | Verified working (in-process); Grafana/OTel never run | `test_infra.py` | `state.Metrics`, `telemetry.py`, `infrastructure/` |
| F40 | Dark/light theme toggle | Partially working | Inspected (icon logic shows both icons) | `components/shell.tsx:26` |
| F41 | PWA manifest | UI-only (manifest only, no service worker) | Inspected | `web/public/manifest.webmanifest` |
| F42 | Legacy SPA at `/legacy` | Partially working, superseded | Inspected (calls v1 `/api/profile/calibrate`) | `RT/frontend/index.html` |
| F43 | Hard-coded live exercise list | Mocked or hardcoded | Inspected | `live-workout.tsx:15-26` |
| F44 | Seeded model registry rows | Mocked or hardcoded (honestly labelled) | Inspected | `main.py:28-50` |

### V1 / MU features still in the repo

| # | Feature | Status | Notes |
| --- | --- | --- | --- |
| V1-a | Desktop OpenCV app (setup screen, keyboard controls, JSON export, recording) | Partially working | Not run here (no camera). Forced-choice classifier; defaults to bicep curl (`V1/recognition/fitness_pipeline.py:116`). |
| V1-b | AUTO exercise classifier (RF bundle) | Broken (scientifically) | Pseudo-labels, train/inference window mismatch, no reject option (`CLASSIFICATION_AUDIT.md`) |
| V1-c | "Calibrated" confidence display | Mocked or hardcoded | `min(0.96, score)` cap (`fitness_pipeline.py:139`) is cosmetic, not calibration |
| V1-d | Manual label review | Mocked or hardcoded | Simulated by code (`V1/recognition/label_review_tool.py:80-120`) |
| V1-e | Biomechanics: inverse dynamics, moment arms, muscle optimisation, muscle activation estimate | Partially working (maths tested); scientifically unvalidated | No EMG or force-plate data; defer |
| V1-f | Threaded camera with latest-frame preference | Partially working | No frame id, so duplicate frames get processed (P10) |
| V1-g | V1 FastAPI inference API | Partially working | Superseded by RT |
| MU-a | ST-GCN + transformer training/ONNX export | Partially working (research) | Below majority baseline on pseudo-labels |
| MU-b | Async capture/pose/inference runtime + profiler | Partially working | Superseded by browser inference |

---

## 2. Details for broken and partially working features

Each entry gives reproduction steps, root cause and dependencies.

### F5 Camera setup / framing guidance — Broken
- **Repro:** open the Live or Calibration screen on a laptop webcam (640×480 or 720p) and stand facing the camera, arms down. The calibration message stays at "Turn to face the camera." Offline: `python docs/audit-evidence/pose_study.py` gives a 0% front-gate pass at 4:3 with the app's coordinates, 97% after aspect correction.
- **Root cause:** `pose.ts:60` sends x/W and y/H with no image size. The engine treats them as isotropic. The framing check is all-or-nothing over 13 joints, and messages are generic.
- **Dependencies:** coordinate contract (Roadmap M1).

### F6 Staged calibration — Broken on real cameras
- **Repro:** as F5. The side stage also needs all 13 joints at vis ≥ 0.5, which conflicts with far-side occlusion [Hypothesis].
- **Root cause:** RC-1 to RC-4 in `POSE_CALIBRATION_AUDIT.md`.
- **Dependencies:** M1, then the new protocol (M2).

### F7 Calibration persistence — Partial
- **Repro:** run `verify_rt.py §1`. A front-only calibration of the clean synthetic body gets quality 0.50 and `profile.calibrated=False`, while `POST /calibrate/staged` returns `calibrated: true`.
- **Root cause:** quality is multiplied by coverage, the endpoint hard-codes `True`, and no camera fingerprint or invalidation exists.
- **Dependencies:** M2 data model (`body_calibration`, `camera_setup`, `exercise_range`).

### F8 2D skeleton overlay — Partial
- **Repro:** Live → Demo clip. The overlay draws 12 bones with white dots.
- **Root cause / gaps:** no confidence colouring, no smoothing, nose not connected, no indication of which joints the active exercise uses or which rule failed. Visibility threshold 0.3 for drawing vs 0.5 for gating, so the overlay can look fine while gating says "not visible".
- **Dependencies:** M1 (metadata), M5 design.

### F9 3D body-proportion viewer — Partial
- **Repro:** Profile → calibrate with Demo body **and** enter a height. The 3D figure appears only if `calibrated && segments_cm && height`.
- **Root cause:** a static T-pose built from segment centimetres derived from self-reported height and distorted 2D ratios. It presents relative proportions as centimetres to 0.1 cm.
- **Dependencies:** defer, or recast as "proportions vs average (ratios)".

### F10 Exercise auto-detect — Broken
- **Repro 1:** Home → "Quick set" → Start. The header shows "Bicep Curl", because `"auto"` is mapped to `"bicep_curl"` (`live-workout.tsx:62`).
- **Repro 2:** pick "Auto-detect (experimental)" in the dropdown before Start and do curls. The message says "These exercises look identical… : bicep curl / hammer curl" and nothing more happens (`verify_rt.py §5`).
- **Repro 3:** sway side to side. It locks onto Romanian deadlift (`verify_rt.py §8`).
- **Root cause:** UI mapping bug; no confirm handler; permanent lock; weak unknown rejection; geometry bug upstream.
- **Dependencies:** M1, then S1–S5 architecture (Roadmap M7, deferred).

### F12 / F13 Rep counting and partials — Partial
- **Repro:** `verify_rt.py §4`. Curl re-extending to 150° → 1/5 counted; 140° → 0/5. Press bottom 110° → 0/5.
- **Root cause:** constant rest/peak; strict return to within 15% of rest; silent 12 s abandon; no tracking-loss state; no dwell.
- **Dependencies:** M1, M3 (personal range + counter v2).

### F14 Form analysis — Partial
- **Repro:** `verify_rt.py §9`. A clean squat at 16:9 fails `rom` and `tempo` (form 56).
- **Root cause:** geometry distortion; thresholds unvalidated; evaluated only after a counted rep; rules report "passed" even when their landmarks weren't visible; 2D symmetry in side views.
- **Dependencies:** M1, M3, expert-reviewed examples (Test plan §5).

### F15 ROM metric — Partial
- **Root cause:** reported as a fraction of the *textbook* range (`peak_progress`). The Progress page labels it "of target range of motion". It isn't personal and isn't an angle in degrees.
- **Dependencies:** M3 personal range; report degrees with the view stated.

### F16 Symmetry — Partial
- **Root cause:** mean |L−R| of 2D angles. Invalid in side views, distorted in front views of sagittal motion. No minimum-visibility requirement.
- **Dependencies:** M3 (`evaluable_if`), front-view-only use.

### F17 Personal baselines — Partial
- **Root cause:** an EWMA of peak_progress, duration and form score, used only for insight text after 5 reps. It doesn't adapt thresholds. Stored only when a set is saved.
- **Dependencies:** replace with `exercise_range` (M3).

### F20 Custom exercise definitions — Partial
- **Repro:** POST a valid definition to `/api/admin/exercises`. It appears in `/api/exercises` but not in the live screen's exercise dropdown.
- **Root cause:** `SUPPORTED_EXERCISES` is hard-coded in `live-workout.tsx:15-26`.
- **Dependencies:** M4 (fetch the list from the API).

### F21 Workout builder — Partial
- **Repro:** Workout page. You can create and delete workouts, but you can't edit one, and there's no rest-seconds field (it defaults to 60). Free-text exercise names are resolved by fuzzy match (cutoff 0.82) without showing which exercise was matched.
- **Root cause:** UI lacks edit (the API `PUT` exists) and the rest field. Matching feedback appears only as a "manual" badge.
- **Dependencies:** M4.

### F22 Live workout screen — Partial
- **Repro:** start the server with `uvicorn backend.main:app` and open `/live/?quick=1`. The UI served is the **2026-10-03 build** (no `select_exercise` in the `web/out` bundle).
- **Root cause:** stale committed build artifact; Quick set bug (F10); the exercise select appears twice (header and footer); the "confirm" state has no action; a WebSocket drop isn't detected or recovered (P9); 30 Hz re-renders (P7).
- **Dependencies:** M0 (build hygiene), M4.

### F23 Exercise switching — Broken
- **Repro:** `verify_rt.py §6`. `select_exercise` followed by a frame before the consumer runs → the consumer only sees the frame (`dropped=1`). When the switch does land, the reps already counted are discarded, not saved.
- **Root cause:** control messages share the latest-wins frame slot (`live.py:101-104`); `start()` replaces the session.
- **Dependencies:** M4 (protocol: separate control queue, acks, close-and-save on switch).

### F24 Rest timer — Partial
- **Root cause / gaps:** works as code. The countdown has `aria-live="off"` and the screen gives no audio or haptic cue. It can't be paused or extended.
- **Dependencies:** M5.

### F26 Set saving from AI analysis — Partial
- **Repro:** run an AI set with 0 counted reps and 0 partials, then End set. `set_saved: false` and nothing is stored or logged. The user sees nothing unless the 5 s timeout resolves `null`.
- **Root cause:** `_finish` saves only when `reps or partial_reps`. A dropped WebSocket loses the set.
- **Dependencies:** M4.

### F27 Session summary — Partial
- **Root cause:** shows sets, reps, volume and form. There's no per-rep breakdown or evidence, although per-rep evidence is stored in `reps`.
- **Dependencies:** M5/M6.

### F30 Goals — Partial
- **Root cause:** the goal is a free enum plus a weekly target. Headlines are text templates. Goals have no targets, deadlines or measured progress, apart from consistency and volume change.
- **Dependencies:** defer beyond the vertical slice.

### F31 Progress analytics — Partial
- **Root cause:** volume, reps and e1RM (Epley) are computed correctly from logged data. "Form consistency" and "ROM" rest on unvalidated measurements and are shown without caveats. The local DB holds no AI sets at all.
- **Dependencies:** M3 (trustworthy measurements); label AI-derived metrics as experimental.

### F35 Admin console — Partial
- **Root cause:** functional over the API, but its central purpose (promoting models trained on human-labelled data) can't be exercised because no such data exists.
- **Dependencies:** defer.

### F36 Dataset ingest — Broken by default
- **Repro:** `POST /api/admin/dataset/ingest` without `MEDIAPIPE_POSE_MODEL` set → `PoseUnavailable` inside the job.
- **Root cause:** environment-dependent; never run by the authors.
- **Dependencies:** defer; replace with a client-side recorder (Test plan §2).

### F40 Theme toggle — Partial
- **Root cause:** the Sun/Moon icons use `dark:block`, but Tailwind's `darkMode` isn't configured as `class`/`[data-theme]`, so the icon state doesn't follow the manual toggle [Inspected].
- **Dependencies:** M5.

---

## 3. Keep / repair / replace / defer by feature

| Retain as-is | Repair | Replace | Defer / hide |
| --- | --- | --- | --- |
| F1, F2, F3, F18, F19, F25, F33 (as test tool), F38, F39 (in-process) | F5, F6, F7, F8, F12–F17, F20–F24, F26–F28, F29, F31 | V1-b, V1-c, V1-d; F9 (recast as proportions); F17 (→ personal range) | F10/F11 (auto-detect, learned models), F30 advanced goals, F32, F34 nutrition, F35–F37 (MLOps), F41, F42 legacy SPA, V1-e biomechanics, MU-a/MU-b |
