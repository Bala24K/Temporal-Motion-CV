# Rebuild Roadmap

Aliases and evidence tags are defined in `PROJECT_AUDIT.md`. "Target" means a proposed acceptance threshold, not a measured result.

**Strategy:** repair the RT stack in place. Don't rewrite. RT's architecture (exercise definitions as data, primitives, evidence-based form rules, explicit recognition statuses, data store with consent and subject-disjoint splits) is sound. Its failures come from the input contract, missing personalization, session wiring and a lack of real data, not from its structure. V1 and MU become read-only references.

---

## Smallest end-to-end vertical slice (end of M4)

> Laptop or phone camera → aspect-correct landmarks with metadata → guided setup and calibration with a quality score → user **selects** Squat (side view) or Bicep Curl → 2-rep personal range capture → reliable rep counting with tracking-loss states → set saved (reliably, even across a WebSocket drop) → visible in History and Progress.

No auto-detect, no learned models, no nutrition, no 3D.

---

## Milestone overview

```
M1 Trustworthy input ──► M2 Calibration v2 ──► M3 Counter v2 (squat, curl) ──► M4 Session & product wiring ──► [VERTICAL SLICE]
        │                                              ▲                               │
        └──── recorder ──► data collection (continuous) ┘                               ▼
                                                                         M5 Premium frontend ──► M6 Press + raise, form validation
                                                                                                   │
                                                              (gated on labelled data) M7 Auto-detect S1–S5 ──► M8 Deferred features
```

| Milestone | Depends on | Relative size |
| --- | --- | --- |
| M1 Trustworthy input | — | M |
| M2 Calibration v2 | M1 | M |
| M3 Rep counter v2 + personal range (squat, curl) | M1, M2 (range capture uses setup fingerprint), recorded data from M1 | L |
| M4 Session protocol & product wiring | M1 | M |
| M5 Premium frontend | M2, M3, M4 (real outputs to design around) | L |
| M6 Shoulder press + lateral raise; form-rule validation | M3, expert-labelled data | M |
| M7 Auto-detect (S1–S5) | M3, M6, human-labelled subject-disjoint dataset | L (gated) |
| M8 Deferred features | Product decision | — |

---

## M1 — Trustworthy input (first implementation milestone)

**Goal:** every number the engine computes comes from correctly scaled, timestamped and attributable landmarks. The worst user-visible wiring bugs stop. Real data collection can begin.

**Scope**
1. **Coordinate contract** (`RT/web/lib/pose.ts`, `RT/motion_engine/skeleton.py`): send `image.w/h`, `mirrored`, `frame_idx`, `t_capture_ms` (from `requestVideoFrameCallback` metadata), model id/version, visibility **and** presence, and 3D world landmarks. Convert to isotropic coordinates once on the server boundary (`x·W/H`). Reject frames without metadata. Framing checks keep using the raw [0,1] values.
2. **Pin and self-host** MediaPipe wasm and model (exact version). Fall back to the CPU delegate.
3. **Landmark recorder and replay harness** (`TEST_AND_EVALUATION_PLAN.md §2.1–2.3`). No video by default.
4. **Hotfixes:**
   - Quick set: don't map `"auto"` to `"bicep_curl"`. Until M7, Quick set opens an exercise picker (`RT/web/components/live-workout.tsx:62`).
   - Control messages (`select_exercise`, `end_set`) bypass the latest-wins frame slot and go to an ordered control queue (`RT/backend/routers/live.py:101-104`).
   - Calibration response `calibrated` = the stored profile's real state (`RT/backend/routers/profile.py:79`).
   - Stale build: stop serving the committed `web/out`; build in CI/Docker; expose the build commit in `/api/health` and the bundle.
5. **Audit regressions** added as tests (failing ones marked `xfail` with a link to the milestone that fixes them).

**Acceptance criteria (all measurable)**

| # | Criterion | How measured |
| --- | --- | --- |
| A1 | Aspect invariance: one synthetic pose rendered at 4:3, 16:9, 3:4 and 9:16 gives every primitive angle within **0.5°** and every ratio within **1%** of isotropic | New unit test |
| A2 | The synthetic standing body passes the front calibration stage at **all four** aspect ratios (today: fails at 4:3 and 16:9) | `verify_rt.py §1` as a test |
| A3 | On the fixed QEVD frame set from `pose_study.py` (cached landmarks committed as a fixture), the front-gate pass rate with the new contract is **≥ 90%** for both 4:3 and 16:9 (today: 0% / 25%; with correction 97% / 94%) | Fixture test |
| A4 | Joint-angle reproduction: true 45° reads 45.0 ± 0.5° at 16:9 (today: 29.4°) | `verify_rt.py §3` as a test |
| A5 | No control message is ever dropped: 1,000 randomized interleavings of frames and control messages → every control message is processed exactly once, in order | Async unit test (today: dropped, `verify_rt.py §6`) |
| A6 | Quick set never starts a bicep-curl session unless the user picks it | Component test + Playwright |
| A7 | `/api/health.build` equals the commit hash embedded in the served bundle | CI check |
| A8 | Calibration endpoint status equals `GET /api/profile.calibrated` for front-only, 2-stage and 3-stage calibrations | API test |
| A9 | Replaying a recording through `LiveSession` reproduces the live session's per-frame payloads **exactly** | Replay determinism test |
| A10 | Frames missing `image.w/h` or `t_capture_ms` are rejected with an error message | API/WebSocket test |
| A11 | Recorder used to capture **≥ 12 sets from ≥ 3 people** (squat side, curl), double-annotated rep counts | Data inventory report |
| A12 | Existing suites still pass (RT 163, V1 255 or explicitly retired) | CI |

**Regression tests introduced:** A1–A10, plus `verify_rt.py §1, §3, §6` converted to tests.

**Status (2026-11-18): implemented, except where noted.** Tests: `RT/tests/test_m1_contract.py` (RT suite: 204 passed + 3 strict `xfail` rep-counting regressions that M3 must turn green). Playwright E2E (`RT/scripts/e2e_web.py`) passes against the real server with 0 console errors.

| # | State | Notes |
| --- | --- | --- |
| A1, A2, A4 | Done | Aspect invariance across 4:3, 16:9, 3:4 and 9:16 |
| A3 | Done | `RT/tests/fixtures/qevd_front_frames.json` (241 real frames, landmarks only); front gate set to 0.50 isotropic |
| A5 | Done | `LatestSlot.put_control`: lossless FIFO for control messages |
| A6 | Done | Quick set requires an exercise choice; verified in E2E. Auto-detect lock now shows in the selector; `confirm` has candidate chips |
| A7 | Done, adapted | No git here, so the build records a hash of the UI sources (`build-info.json`) and `/api/health.ui.stale` compares it |
| A8 | Done | `BodyProfile.stability` separates "usable" from overall quality; the endpoint reports the real state |
| A9 | Done | `ml/replay.py` reproduces live payloads exactly (lock-step test) |
| A10 | Done | WebSocket and REST reject frames without `_image` |
| A11 | **Not done** | Needs people: record at least 12 sets from at least 3 subjects with the new recorder, and have two people count reps for each |
| A12 | Done | Existing tests updated for the contract; V1 untouched |
| Scope 1, 3D landmarks | **Deferred to M2** | World landmarks are not sent yet |
| Scope 2, self-hosting | **Partial** | Model and wasm versions pinned; CPU fallback added; assets still loaded from the CDN |
| Extra | Done | Exercise switch saves the set in progress; WebSocket drop shows "not saved" and switches to manual logging with the counted reps prefilled; exercise list loaded from the API; theme toggle fixed; calibration no longer runs inference twice |

---

## M2 — Calibration v2

**Goal:** a real person on a laptop or phone calibrates in under a minute, gets a quality score they can trust, and is told precisely what to fix when it fails. The calibration is tied to the camera setup.

**Scope:** protocol in `POSE_CALIBRATION_AUDIT.md §6`: setup checklist, geometric facing test (not depending on the user's own proportions), per-joint visibility gating per stage, contiguous 1.5 s stillness window, ratios (2D + 3D cross-check), quality components, setup fingerprint and invalidation, `body_calibration` + `camera_setup` tables, specific failure messages, local stage checks (no HTTP per frame), and removal of the duplicate `grab()` inference. Replace "arms up" with "arms forward/T"; make "side" optional and used only for sagittal-view setup checks.

**Acceptance criteria**
- Synthetic: shoulder/torso 0.55–0.95 bodies pass the front stage when facing; side stage passes with the far arm at visibility 0.2; injected jitter and drift trigger "hold still" / "camera moved".
- Real (recorded, ≥ 12 subjects): success ≤ 60 s in **≥ 90%** of attempts (target); ratio CV across 3 sessions **≤ 5%** (target); laptop vs phone ratio difference **≤ 5%** (target); failure message names the correct joint or issue in **≥ 90%** of induced failures (target).
- Fingerprint change (distance > 15%, device, aspect) is detected **before the next set** in ≥ 95% of induced changes (target).
- UI shows centimetres only with height entered and an explicit ± range. Default display is ratios.

**Regression tests:** gate-robustness suite; fingerprint invalidation suite; calibrated-status consistency (from M1).

**Status (2026-11-18): software implemented; the real-recording criteria are still open.** Tests: `RT/tests/test_m2_calibration.py` (28 tests). Full RT suite: 232 passed + 3 strict `xfail` (M3). Browser E2E passes with the new three-stage wizard.

| Item | State | Notes |
| --- | --- | --- |
| Geometric facing test | Done | 3D shoulder/hip yaw from world landmarks (`"w:<joint>"` keys sent by `pose.ts`); isotropic ratio only as a fallback. A narrow-shouldered body that the 2D ratio refuses passes with 3D. |
| Per-joint gating per stage | Done | Side stage needs only the visible side; it passes with the far limbs at visibility 0.2 |
| Contiguous-window stillness | Done | `assess_window`: jitter and sustained drift over ~1.5 s; one request per window (`POST /api/profile/calibrate/assess`), replacing one per frame |
| Specific failure messages | Done | Names the joint and the fix (e.g. "Your left ankle is out of frame - step back."); a timeout reports the most frequent problem |
| Arms-out (T) pose | Done | Replaces arms-up for arm lengths; arms-up still accepted. Legs come from the front pose; the side view is never used for lengths |
| Quality components and status | Done | quality = min(stability, confidence, coverage); status good/usable/retry; plausibility and 2D/3D disagreement only warn |
| Setup fingerprint and invalidation | Done | Stored in the profile JSON (aspect, image, body height fraction, hip centre, ankle roll, camera label). The live socket checks the first 30 frames and sends `setup`: camera_changed / distance_changed / uncalibrated |
| Persistence | Done, simplified | Inside the existing `body_profiles` JSON rather than new `body_calibration`/`camera_setup` tables; no migration needed. History of setups not kept yet |
| Centimetres | Unchanged | Still shown with a height entered; no ± range displayed yet |
| Real-recording targets (success ≤ 60 s for ≥ 90% of attempts, ratio CV ≤ 5%, device independence, message accuracy) | **Not measured** | Needs recorded subjects. Thresholds marked PROVISIONAL in `motion_engine/calibration.py` (yaw limits, window visibility, jitter, drift, distance) must be tuned on those recordings |

---

## M3 — Rep counter v2 with personal range (squat side, curl)

**Goal:** count reps people actually do.

**Scope:** `REP_COUNTING_AUDIT.md §4`: `ExerciseDefinition` v2 (views, per-view required landmarks, `side: visible`, `population_range`, `personal_range`, start/peak/complete with dwell, tracking states, `evaluable_if`), personal range capture (2–3 reps; stored in `exercise_range`), relaxed completion (velocity reversal + next descent), a `degraded`/`lost` tracking model ported from V1's `RepStateMachine` ideas, time-based (not per-frame) smoothing, a One Euro landmark filter, form rules only marked passed when evaluable. Remove `front` from the squat's supported views. Curl defaults to the side/45° view.

**Acceptance criteria**
- Synthetic regressions (all currently failing): curl re-extending to 150°/140° → **5/5**; press bottom 110° → 5/5 (enabling press tests early); 13 s hold → 1 rep with correct duration; 15 vs 30 fps → identical counts; 1 s tracking gap → counted; 4 s gap → abandoned with state surfaced.
- On the human-counted golden set (squat side + curl, ≥ 12 subjects): mean absolute count error **≤ 0.3 reps/set**; **≥ 90%** exact sets and **≥ 98%** within ±1; false reps **≤ 2%**; missed **≤ 3%** (all targets).
- A clean synthetic set gives zero form warnings at every aspect ratio (today a 16:9 squat fails `rom`+`tempo`).

**Regression tests:** synthetic scenario suite; golden-set replay gate in CI.

**Status (2026-11-18): software implemented; the golden-set targets are still open.** Tests: `RT/tests/test_m3_counter.py` (29 tests). The three audit regressions that were strict `xfail` in M1 now pass as ordinary tests. Browser E2E passes, including personal range capture.

| Item | State | Notes |
| --- | --- | --- |
| Counter v2 | Done | `motion_engine/repetition.py`: time-based EMA and dwell. A rep completes on a relaxed return (< 0.35), a reversal into the next rep, or a settle (paused part-way down). An open rep is closed at end of set (`flush`). Timeouts are counted as `abandoned`, never silently dropped |
| Incomplete re-extension | Done | Curl back to 150/140/125 deg and press from 110 deg: 5/5 (were 1/5, 0/5, 0/5) |
| Frame-rate independence | Done | Same counts at 15, 24, 30 and 60 fps for squat, curl, press and raise |
| Tracking states | Done | ok -> degraded (rep kept open) -> lost after 3 s (rep abandoned, smoothing re-anchored); shown in the UI |
| Personal range | Done | `motion_engine/personal_range.py`: 2+ full cycles, robust extremes, clamped to `population_range` (default ±25%), keyed to the definition version. Captured live ("Capture my range") or via `POST /api/profile/range`; stored in `baselines` and applied to later sets. A front-view curl reaching only ~110 deg: 0/5 without a range, 5/5 with one |
| Visible side | Done | `laterality: visible_side` plus `side: "visible"` primitive; side-on quality gating needs only one clear side; midpoints fall back to the visible joint. A side-on squat with the far leg hidden, or wrongly guessed, still counts 5/5 |
| Definitions v2 | Done (squat, bicep/hammer curl) | Squat: side/45deg only, knee-symmetry rule removed (not judgeable side-on). Curl: 45deg preferred; symmetry only in front/45deg views |
| Form rules | Done | P95/P5 instead of max/min; `evaluated: false` ("not checked") when the rule's joints weren't visible for 80% of the rep or the view is wrong; excluded from the score |
| View guidance | Done, advisory | Per-set view estimate (3D yaw, else shoulder ratio). Shows a hint such as "turn side-on" but does not block counting until the classifier is tuned on recordings |
| One Euro landmark filter | **Not done** | The time-based EMA on the rep signal covers counting; landmark-level filtering waits for recordings to tune it |
| Golden-set targets (MAE <= 0.3 reps/set, >= 90% exact sets, false reps <= 2%, missed reps <= 3%) | **Not measured** | Needs human-counted recordings (`python -m ml.replay` is the harness). Counter thresholds (start/complete/reversal/settle, tau, lost_after, range clamp, MIN_AMPLITUDE 0.35) are provisional until then |
| Performance | Measured | `LiveSession.push` p50 0.41 ms / p95 1.2 ms (synthetic, this machine), up from 0.18 ms; within the 5 ms budget |

---

## M4 — Session protocol and product wiring

**Goal:** the slice works end to end, reliably.

**Scope**
- Typed WebSocket protocol (JSON Schema from Pydantic → zod on the client); control channel with acks; frames latest-wins; state messages throttled to ≤ 15 Hz.
- Exercise change = close and save the current set, then start a new one; between sets only in the UI.
- Reconnect with session resume (server keeps the `LiveSession` for N seconds keyed by a set id); the client buffers `end_set` until acknowledged.
- Token sent in the first WebSocket message, not the URL.
- Recognition off the event loop (`asyncio.to_thread`) for when M7 returns.
- Exercise list from `/api/exercises` (custom definitions appear).
- History page (`GET /api/sessions`, `/api/sessions/{id}` with per-rep evidence); workout edit (`PUT /api/workouts/{id}`) and rest-seconds field.
- Saving zero-rep AI sets: ask "Save as 0 reps / log manually / discard".

**Acceptance criteria**
- Kill the WebSocket mid-set (Playwright): the set resumes or is saved with all counted reps in **100%** of 20 trials.
- Switching exercise saves the previous set in 100% of trials; no reps lost.
- An exercise added through the admin API appears in the live picker without a redeploy.
- History shows every finished session with per-rep details; edit round-trips.
- Server event loop: no task over 10 ms during a 10-minute 2-client run.

**Regression tests:** protocol contract tests (server ↔ client schemas), reconnect E2E, switch-and-save E2E.

**→ Vertical slice complete.**

---

## M5 — Premium frontend

**Goal:** a camera-first sports-performance experience built on M2–M4 outputs (`FRONTEND_DESIGN_AUDIT.md §3`).

**Scope:** tokens and type scale; dark-first theme; navigation (bottom tabs / left rail; admin and nutrition out of the main nav); live HUD (counter, phase gauge vs personal range, cue stack, tracking badge, setup checklist, set summary with per-rep evidence); skeleton overlay v2 (confidence opacity, active-joint emphasis, rule pulses, midline); pose inference in a Web Worker; library with camera-requirement emphasis and setup illustrations; analytics with experimental labels on AI metrics; accessibility fixes (live region and audio for reps, focus management, accessible dialogs); error and empty states everywhere.

**Acceptance criteria**
- axe-core: 0 serious or critical violations on all routes; AA contrast on every token pair.
- Playwright screenshots at 375, 768, 1280 and 1920 widths: no horizontal overflow; during a set, at most 2 controls visible.
- Rep count announced through a live region (assistive-tech test) and optional audio cue.
- 30-minute fake-camera soak: browser heap growth < 10%; camera fps stays ≥ 24 on the reference laptop (target); end-to-end p95 ≤ 150 ms (target).
- Lighthouse performance ≥ 90 on app routes (target).

**Regression tests:** component tests (Vitest + RTL), visual snapshots, accessibility scan in CI.

---

## M6 — Shoulder press, lateral raise; form-rule validation

**Scope:** v2 definitions for press (front) and lateral raise (front); expert-labelled form data (`TEST_AND_EVALUATION_PLAN.md §2.4`); per-rule precision and false-warning rate; rules that miss their target are downgraded to `info` or disabled.

**Acceptance criteria:** rep-count targets as M3 for both exercises; per-rule warning precision **≥ 0.8** and **≤ 5** false warnings per 100 clean reps (targets); every warning carries measured value, threshold, view and frames.

---

## M7 — Auto-detect (gated)

**Entry gate:** a human-labelled, subject-disjoint dataset meeting `TEST_AND_EVALUATION_PLAN.md §2.4` (≥ 20 subjects, negative class ≥ 25%), with the dataset report saying `gate_ready`.

**Scope:** S1 movement detection, S2 family, S3 specific exercise (view-aware), S4 phase (= the rep engine), S5 rejection (`CLASSIFICATION_AUDIT.md §5`); confirm chips; re-recognition; compare against logistic regression and the rules before any deep model.

**Acceptance criteria (targets):** macro-F1 ≥ 0.90 at ≤ 10% abstention; negative-movement false acceptance ≤ 5% (today a side-bend is accepted as RDL [Reproduced]); decision ≤ 3 s after the first rep; ≤ 1 label flip per 5 minutes.

---

## M8 — Deferred (explicitly out of scope until the slice is validated)

| Feature | Reason to defer | Revisit when |
| --- | --- | --- |
| Learned recognizer / encoder / ST-GCN | No labelled data; historical models at or below the majority baseline on pseudo-labels | M7 entry gate met and rules + logistic regression plateau |
| Nutrition | Outside movement analysis; adds surface area | Core slice validated and user research asks for it |
| Admin model registry, Label Studio, DVC, MLflow, dataset ingest | Infrastructure for data that doesn't exist yet | Annotation project starts (keep the code, hide the UI) |
| 3D body-model viewer | Shows distorted proportions as cm; little user value | Recast as a ratios card in M5 |
| V1 biomechanics (inverse dynamics, muscle estimation) | Scientifically unvalidated | Validated against EMG or force data |
| RTMPose / YOLO-Pose | No benchmark on real footage | Benchmark on the golden set shows gains |
| OIDC/PKCE, observability stack, Postgres, Kubernetes | Not needed for a single-user validation slice | Multi-user deployment target exists |
| Goals beyond weekly target | Depends on trustworthy metrics | After M6 |
| Front raise, RDL, overhead triceps extension, goblet squat as a separate exercise | View and hinge ambiguity (see the rep audit) | Side-view support proven in M3 |

---

## Repository hygiene (runs alongside M1)

- Initialise git at the repository root (only `Ver-1/` is a git repo today). Move `Ver-1/`, `motion-upgrade/` and `motion-upgrade.zip` to `archive/`, or delete `motion-upgrade/` (it duplicates `RT/research/motion_upgrade/`).
- Keep `QEVD-FIT-300k-Part-1/` (76k videos) and the `.pkl`/`.joblib` artifacts out of version control (DVC or external storage).
- Remove `RT/fitness.db` from the tree. Seed a dev DB through a script.
- Fix the documentation claims the audit found inconsistent: test counts (STACK.md "190+", V1 README "216"); V1 README vs report accuracy figures; the "manual review" wording.
