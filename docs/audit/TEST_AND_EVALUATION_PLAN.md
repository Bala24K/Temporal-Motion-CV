# Test and Evaluation Plan

Aliases and evidence tags are defined in `PROJECT_AUDIT.md`.

> **Convention used in this document.** **Measured** = observed during this audit. **Target (proposed)** = an acceptance threshold to adopt. A target is not a result. No real-world accuracy, rep-count or form result exists for this project today.

---

## 1. Current state

| Item | Measured |
| --- | --- |
| RT unit/integration tests | 163 passed (68 s). Synthetic or mocked inputs only. |
| V1 tests | 255 passed (35 s). Synthetic angle traces and maths only. |
| Browser E2E (`RT/scripts/e2e_web.py`) | Exists; not run in this audit; uses demo clips only |
| Real landmark sequences in any test | **0** |
| Human-labelled exercise clips | **0** (QEVD Part 1 has no labels; V1 "manual review" was simulated) |
| Human-counted reference sets | **0** |
| Expert-reviewed form examples | **0** |
| Frontend component tests | **0** (no Vitest/Jest/RTL in `package.json`) |

The tests prove the code does what its authors designed on idealised input. They can't catch interface-contract bugs (aspect ratio), domain shift (real people), or product wiring bugs that only show up in a browser.

---

## 2. Evaluation infrastructure (build once, used by everything below)

### 2.1 Landmark recorder (client-side, privacy-preserving)

A dev-mode toggle in the live and calibration screens records exactly what the engine receives: the per-frame message from the new coordinate contract (`t_capture_ms`, frame index, W/H, mirrored, model id, 2D + 3D landmarks with visibility/presence), plus device info. **No video.** The recording is saved as `recording.jsonl.gz` with a sidecar `meta.json`:
```json
{"recording_id": "r_000123", "subject_id": "s_017", "consent": "research_v1", "exercise": "squat",
 "view": "side", "camera": {"device": "laptop-720p", "w": 1280, "h": 720, "height_m": 0.9, "distance_m": 3.0},
 "conditions": {"lighting": "indoor_daylight", "clothing": "fitted", "occlusion": "none"},
 "annotations": "annotations.json"}
```
Video is recorded **only** with explicit consent, and only so that annotators can label. It is kept separate from landmarks and deleted after annotation (`RT/ml/data/store.py` already enforces consent and deletion; reuse it).

### 2.2 Annotation schema

```json
{"labels": {"exercise": "squat", "view": "side"},
 "reps": [{"start_ms": 1200, "bottom_ms": 2100, "end_ms": 3050, "counted": true, "partial": false}],
 "events": [{"type": "tracking_loss", "start_ms": 8000, "end_ms": 9100}],
 "form": [{"rep": 3, "rule": "depth", "verdict": "fail", "reviewer": "coach_02", "note": "above parallel"}],
 "annotators": ["a_01", "a_02"], "agreement": {"rep_count": 1.0, "rep_boundaries_iou": 0.86}}
```
Rep counts are annotated independently by **two** people. Disagreements are adjudicated. Inter-annotator agreement is reported for every label type.

### 2.3 Replay harness

`python -m eval.replay --recordings data/eval/ --engine HEAD --report out/eval.json` feeds recordings through `LiveSession` (and calibration and recognition) at the recorded timestamps, including dropped frames. It is deterministic, CI-runnable, and needs no camera. The **golden set** (a frozen, versioned subset) is the regression suite.

### 2.4 Minimum data collection (proposed)

| Purpose | Minimum | Composition |
| --- | --- | --- |
| Calibration study | 12 subjects × 3 sessions × 2 devices | Varied height (≥ 150 to ≥ 190 cm), build, sex; laptop 16:9 + phone portrait |
| Rep counting (4 MVP exercises) | 12 subjects × 4 exercises × 3 sets × 8–12 reps ≈ 1,400 reps | Include deliberate partial reps, pauses, incomplete re-extension, walking out of frame |
| Classification (when auto returns) | ≥ 20 subjects; ≥ 40 clips per class; plus a **negative class** (other exercises, walking, stretching, idle, fidgeting) ≥ 25% of the data | Subject IDs required |
| Form | ≥ 30 reps per rule per exercise with expert labels, ≥ 30% faulty | Two qualified reviewers (e.g. certified coaches) |

QEVD Part 1 can supply **candidate** clips for annotation (pose robustness, varied people), but it has no subject IDs, so it can't support subject-independent evaluation. Treat any QEVD-based split as video-disjoint only.

---

## 3. Pose and calibration

### 3.1 Automated (synthetic and contract) tests — add now

| Test | Assertion |
| --- | --- |
| Aspect invariance | The same synthetic pose rendered for 4:3, 16:9, 3:4 and 9:16 → after the boundary conversion, every primitive's angle agrees within 0.5° and every ratio within 1% |
| Coordinate contract | Missing W/H → message rejected; mirrored flag respected; timestamps strictly increasing |
| Calibration gate robustness | Synthetic bodies with shoulder/torso 0.55–0.95 (isotropic) all pass the front stage when facing |
| Calibrated-state consistency | The API response `status` equals what the engine reads from the stored profile |
| Fingerprint invalidation | Distance change > 15%, aspect change and device change each trigger the specified re-validation |
| Stability gate | Injected jitter above the threshold → stage refuses with "Hold still"; drift → "camera moved" |
| Per-joint gating | Side stage succeeds with the far arm at visibility 0.2 |

### 3.2 Real-recording evaluation (proposed targets)

| Condition | Protocol | Metric | Target (proposed) |
| --- | --- | --- | --- |
| Different body proportions | 12+ subjects | Calibration success within 60 s | ≥ 90% of attempts |
| Repeatability | 3 sessions per subject, same setup | CV of each ratio across sessions | ≤ 5% |
| Device/aspect independence | Same subject, laptop 16:9 vs phone 9:16 | Ratio difference | ≤ 5% |
| Camera distance | 2.0 / 3.0 / 4.0 m | Ratio difference vs 3.0 m | ≤ 5%; rep-angle bias ≤ 3° |
| Lighting | Bright / normal / dim / backlit | Median visibility; stage success | Report; dim/backlit must produce a specific lighting hint, not a generic failure |
| Occlusion | Furniture blocking the lower legs; baggy clothing | Correct failure message naming the joint | ≥ 90% correct messages |
| Partial body visibility | Feet out of frame | Calibration refuses; upper-body exercises still allowed | 100% |
| Camera movement | Move the camera mid-session | Detected before the next set | ≥ 95% |
| Confidence degradation | Synthetic visibility decay on recordings | Graceful state change (degraded → lost) | No rep counted during "lost" |
| Pose model choice | lite vs full vs heavy on the golden set | Joint jitter, detection rate, angle error vs annotated keyframes, latency | Choose the smallest model meeting the rep targets |

The **metric-measurement claim stays disabled** unless a study with a known-size reference shows segment error within a stated bound. Until then, show ratios only.

---

## 4. Classification (when auto-detect is re-introduced, Roadmap M7)

- **Splits:** subject-disjoint train/val/test, enforced by the existing leakage assertion (`RT/ml/data/split.py`). Report per-subject variance.
- **Metrics:** confusion matrix over classes **plus `unknown`**; per-class precision, recall and F1; macro-F1; abstention rate; **selective accuracy** (accuracy on non-abstained windows); OOD AUROC on the negative class; expected calibration error.
- **Baselines to beat:** majority class; the current rule recognizer (C1); logistic regression on the 35 interpretable features. Deep models only if they beat logistic regression on a pre-registered metric by more than the subject-variance CI.
- **Temporal stability:** label flips per minute on continuous recordings; time-to-correct-decision from movement onset.
- **Unknown rejection:** recall of `unknown` on held-out negative movements (incl. side-bend, thruster, upright row, which are currently accepted or confused [Reproduced]).
- **Targets (proposed):** macro-F1 ≥ 0.90 on recognisable groups at ≤ 10% abstention; false acceptance of negative movements ≤ 5%; decision within 3 s of the first rep; ≤ 1 flip per 5 minutes.

---

## 5. Repetition counting

| Metric | Definition |
| --- | --- |
| Absolute count error | \|predicted counted reps − human count\| per set |
| Exact-set rate | % of sets with error = 0 (and % within ±1) |
| Missed reps | Human reps with no predicted rep overlapping the annotated [start, end] |
| False reps | Predicted reps with no overlapping human rep |
| Partial classification | Precision/recall of `partial` vs annotation |
| Boundary accuracy | Median \|t_pred − t_human\| for rep end |
| Recovery | Reps counted correctly after an annotated tracking loss |

Scenario coverage (in the golden set and synthetic tests): full reps; incomplete re-extension (curl 140–155°, press 100–115° bottom); deep and shallow; slow (≥ 6 s) and fast (≤ 1 s) reps; pauses at top and bottom; walking out of frame mid-set; one-arm alternating curls; switching exercise between sets; different frame rates (15/24/30 fps) and dropped frames (10–30%).

**Regression tests to add now (synthetic, from this audit's reproductions):**
- Curl re-extending only to 150° and 140° → 5/5 counted (currently 1/5 and 0/5).
- Press bottoming out at 110° → 5/5 (currently 0/5).
- Hold of 13 s at the bottom → 1 rep with duration including the descent.
- Same clip at 15 fps and 30 fps → identical counts.
- Tracking loss of 1 s mid-rep → rep counted; 4 s → rep abandoned and the state surfaced.

**Targets (proposed, on the human-counted golden set):** mean absolute count error ≤ 0.3 reps/set; ≥ 90% of sets exact and ≥ 98% within ±1; false reps ≤ 2%; missed reps ≤ 3%; partial-rep F1 ≥ 0.8.

---

## 6. Form analysis

- **Data:** expert-labelled reps per rule (§2.4). Each label is pass, fail or "not judgeable from this view".
- **Metrics per rule:** precision and recall of the warning; false-warning rate per 100 clean reps (the main driver of user trust); `not_evaluated` rate.
- **Explainability:** every warning carries the measured value, threshold, signal name, the frames it came from, and the view. A test asserts that no warning lacks these fields.
- **Targets (proposed):** per-rule warning precision ≥ 0.8; false warnings ≤ 5 per 100 clean reps. A rule that misses either target ships as "info" severity or is disabled.
- **Regression:** a clean synthetic set gives zero warnings at every supported aspect ratio (currently a squat at 16:9 fails `rom`+`tempo` [Reproduced]).

---

## 7. Frontend

| Layer | Tooling (justification) | Scope |
| --- | --- | --- |
| Component tests | **Vitest + React Testing Library** (new dev dependencies; there is no component-test tooling today) | `RepCounter`, `PhaseGauge`, `CueStack`, `SetupChecklist`, `ErrorState`; protocol reducer for the live state machine |
| Integration tests | Vitest with a mocked WebSocket (protocol fixtures from server JSON Schema) | Exercise switch acked and set saved; `confirm` chips; reconnect; calibration status consistency |
| Browser E2E | **Playwright** (already used by `scripts/e2e_web.py`) with Chromium's `--use-file-for-fake-video-capture` fed a recorded `.y4m` clip | Real camera path incl. MediaPipe in the browser; Quick set; full workout; history |
| Responsive layout | Playwright screenshots at 375×812, 768×1024, 1280×800, 1920×1080; axe-core accessibility scan | No horizontal overflow; controls reachable; AA contrast |
| Long-session stability | Playwright: 30-minute fake-camera session | Heap growth < 10%; no rising dropped-frame rate; WebSocket stays open; FPS stable |
| Build hygiene | CI check: served `web/out` built from the current commit (embed the commit hash in `/api/health` and the bundle) | Prevents stale UI |

---

## 8. Performance (measure, then set budgets)

| Quantity | How | Budget (proposed) |
| --- | --- | --- |
| Camera FPS (delivered) | `requestVideoFrameCallback` counts | ≥ 24 fps on the target laptop |
| Pose latency (browser) | Worker timing per frame | p95 ≤ 40 ms (lite/full chosen accordingly) |
| Engine latency (server) | `frame_process` metric (measured 0.18 ms p50 here) | p95 ≤ 5 ms |
| End-to-end (capture → feedback rendered) | Capture timestamp to paint | p95 ≤ 150 ms |
| Frame drops | `dropped_frames` / total | ≤ 5% |
| Event-loop blocking | Recognition moved to a thread | No single task > 10 ms on the loop |
| Memory (browser) | 30-minute soak | Growth < 10% |

---

## 9. CI gates

1. Unit, contract and synthetic regression tests (fast, every push).
2. Golden-set replay: count-error and false-warning metrics must not regress beyond the CI of the last release.
3. Frontend typecheck, component and integration tests.
4. Playwright E2E with a fake camera (nightly and before release).
5. Release gate: §3–§6 targets met on the held-out human set; numbers published with dataset version, subject count and confidence intervals.
