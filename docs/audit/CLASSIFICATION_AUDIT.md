# Exercise Classification Audit

Aliases (`RT/`, `V1/`, `MU/`) and evidence tags are defined in `PROJECT_AUDIT.md`.

---

## 1. Classification paths that exist

| # | Path | Where | Used live? | Training data |
| --- | --- | --- | --- | --- |
| C1 | **Definition-scoring rules** | `RT/motion_engine/recognition.py` | **Yes.** RT auto mode, `RECOGNIZER_MODE=rules` (default) | None (hand-set constants) |
| C2 | ONNX MLP + Mahalanobis OOD + conformal sets | `RT/motion_engine/learned.py`, `RT/ml/train_recognizer.py` | **No.** No `.onnx` exists in the repo (`RT/out/` and `RT/models/` are absent), so `RecognizerService` falls back to `rules` (`RT/backend/recognizers.py:22-30`) | Synthetic generator only |
| C3 | PyTorch ST-GCN→transformer encoder (SupCon) | `RT/ml/torch_models.py`, `train_encoder.py` | No | 54 synthetic clips (`RT/reports/encoder_metrics.json`) |
| C4 | V1 tabular classifiers (RF / XGB / LR / MLP / 1D-CNN) | `V1/recognition/*`, bundle `V1/models/exercise_recognizer_bundle.joblib` | Yes, in the **V1 desktop app** (`V1/app.py` → `FitnessCVEngine`) | QEVD clips with **heuristic pseudo-labels** |
| C5 | ST-GCN + transformer, exercise + phase heads | `MU/motion/*`, `MU/models/motion_stgcn.pt` | No. Seeded as a `candidate` row in RT's model registry (`RT/backend/main.py:39-49`) | 247 QEVD clips, pseudo-labels, phase labels derived from kinematics |

---

## 2. Trace of the production path (C1)

1. `live.py:142-160`: with `exercise=auto`, frames are appended to `history` (last 180 kept). Once there are ≥ 60 frames, `state.recognizers.recognize(frames)` runs at most once per second, **synchronously on the event loop** (≈ 14 ms [Measured]).
2. `recognize()` (`recognition.py:91`):
   - `fill_gaps` (gaps ≤ 6 frames) → `normalize` **without the user's profile** (per-frame torso scale, no yaw compensation).
   - For each `supported` definition, `score_definition`:
     - computes the definition's rep signal → progress = (smooth(sig) − rest)/(peak − rest);
     - range = max − min progress; amplitude fit = √range when range ≤ 1, Gaussian decay above 1;
     - runs a full `RepCounter` over the window: periodic = 1.0 if at least one rep or partial was seen, else 0.6;
     - multiplies by a posture penalty `exp(−(violation/scale)²)` per posture constraint.
   - If every definition's range is under 0.30, the result is `no_movement`.
   - Definitions are grouped by `equivalence_key()` (same signal, rest, peak and posture). Each group's score is the max of its members' scores.
   - Softmax over group scores at temperature 0.12; `conf = score × probability`.
   - Decision: `top.score < 0.45` → `unsupported`; `conf < 0.55` → `ambiguous`; group has more than one member → `ambiguous` ("look identical"); otherwise `recognized`.
3. In `live.py`, `analyze` → `start(registry.get(rec.top.exercise))`, replay the history into the new session (so early reps count), then **lock permanently**. `confirm` and `fallback` only send the message to the client.
4. Client (`live-workout.tsx:125-127`) shows `rec.message` plus the member names. **There is no button to confirm a candidate.** The user has to know to pick an exercise from the dropdown, which sends `select_exercise`, and that message can be dropped (P1 in `PROJECT_AUDIT.md`).

**And in practice auto mode is never reached from the main entry.** "Quick set (auto-detect)" creates an item with `exercise_id: "auto"`, but `SetScreen` initialises `exerciseId` to `"bicep_curl"` whenever the id is `"auto"` (`live-workout.tsx:62`) and connects with that. Auto mode only starts if the user picks "Auto-detect (experimental)" from the dropdown before pressing Start.

---

## 3. Answers to the audit questions

| Question | Finding | Evidence |
| --- | --- | --- |
| Which features are actually calculated? | C1: per-definition progress range, amplitude fit, rep-cycle presence, posture penalty (angles only). C2: 7 channels (elbow, knee, hip angle, arm elevation, trunk lean, wrist height, L–R elbow asymmetry) × 5 stats = 35. C4: per-frame landmarks, angles, Δ, Δ², symmetry, ROM → mean/std/min/max/last/ROM (≈870 dims). | [Inspected] |
| Is normalization consistent between training and inference? | C2: same `window_features()` code in both, **but** training inputs are isotropic synthetic and live inputs are anisotropic real landmarks. That is a guaranteed distribution shift [Inspected; real-input behaviour not measured]. C4: **inconsistent.** Training used `frame_step=4` over whole clips (≈7.5 fps effective, `V1/recognition/qevd_dataset.py:95`); live uses every frame at 30 fps over a 45-frame (1.5 s) window (`V1/app.py:64`). Per-frame velocity and acceleration (the feature group the V1 ablation found most useful, +16.7 points) are about 4× smaller live, and the window covers a different time span. | [Inspected] |
| Identical feature ordering? | C2: yes (`feature_names()` is deterministic, plus a `FEATURE_VERSION` check at load). C4: the same function produces the features and the bundle stores `feature_config`, but there is **no version guard**. Any edit to the feature code silently changes the columns. | [Inspected] |
| Are temporal windows correct? | C1: 2–6 s windows, fine for rep cycles. Lock-on is permanent, so a later change of exercise is never detected. C4: 1.5 s live vs whole clip in training (mismatch). | [Inspected] |
| Do missing landmarks corrupt classification? | C1: `fill_gaps` keeps gaps over 6 frames as NaN, but `smooth()` then **linearly interpolates across every NaN regardless of gap length** (`primitives.py:135-136`), inventing smooth motion across long occlusions. C2: an all-NaN channel becomes `zeros(1)`, which looks like "no movement" on that channel. C4: `is_valid_person_pose(min_visibility=0.05)` (`fitness_pipeline.py:69`) accepts almost anything. | [Inspected] |
| Is exercise classification confused with phase detection? | C1 uses a rep counter (phase machinery) as a periodicity feature. Coupled, but harmless. C5 trains a phase head on phases derived from the same kinematic rules: its phase agreement with the rule is 0.34 and its learned rep count is 0.0 for every clip [Historical]. Neither path separates "is the person exercising", "which family", "which exercise" and "which phase". | [Inspected] |
| Do static thresholds conflict with learned models? | `consensus` mode commits only when the rules and ONNX agree on the group, which is a sensible guard. But both encode the same synthetic assumptions, so their agreement is not independent evidence. Only C1 runs in production. | [Inspected] |
| Is the model actually loaded and used live? | RT: **no learned model is loaded** (no ONNX file; default mode `rules`). V1: yes, the RF bundle is loaded and used. | [Inspected] |
| Does prediction smoothing add latency or stale classifications? | C1: first decision after ≥ 2 s, then every 1 s; replaying history avoids losing early reps. No smoothing across decisions; the first `analyze` wins forever. C4: per-frame prediction with a 5-vote window (3 agreeing, average ≥ 0.55), so changes are fast, **but while consensus builds the analyzer defaults to `bicep_curl` and counts curl reps** (`fitness_pipeline.py:116-117`). Switching the active exercise does **not reset** the analyzers, so stale state carries over. | [Inspected] |
| Are unknown movements forced into a known class? | C4: **always.** 4 classes, no reject option, and the `lateral_raise` training class is the rule's catch-all "everything else". C1: partly rejects. A Y-raise → `unsupported`, but a **side-bend/sway is accepted as Romanian deadlift** (`analyze`, conf 0.61); a thruster, an upright-row-like movement and a half curl → curl group (`confirm`); a small arm swing → raise group (`confirm`). | [Reproduced] `verify_rt.py §8` |

### 3.1 Groups that can never be told apart (C1)

`equivalence_key()` puts identical definitions together, which is honest: the system says so instead of guessing [Reproduced §5, §7]:
- `bicep_curl` ≡ `hammer_curl` (grip isn't visible from 13 landmarks);
- `lateral_raise` ≡ `front_raise` (same signal, range and posture in 2D; the plane of motion needs 3D or a side view).

So auto mode on a clean curl or raise always ends in `confirm`, and `confirm` has no UI action.

### 3.2 Model artifacts, class mappings and splits (historical)

| Artifact | Classes | Split | Reported | Honest reading |
| --- | --- | --- | --- | --- |
| `V1/models/exercise_classifier.joblib` (2.0 MB), `qevd_exercise_classifier.joblib`, `exercise_recognizer_bundle.joblib` | bicep_curl, squat, shoulder_press, lateral_raise | Video-disjoint 70/15/15; **no subject IDs exist**, so it cannot be subject-disjoint | RF 75 ± 10.5% (README) / 64.3% (summary JSON) | Agreement with a 3-threshold rule. The test sets are 12 clips per seed (README) or 37 clips (summary). Majority baseline on the imbalanced set ≈ 59.5%. |
| `MU/models/motion_stgcn.pt` | same 4 | 173/37/37, video-disjoint | test acc 0.432, macro-F1 0.30, bicep F1 0.0 | Below the majority baseline; `trusted_label_fraction 0.0`; `release_ready false` |
| `RT` synthetic ONNX / encoder | 7 groups | Seeds/scales as pseudo-subjects | 100% / 77.8% (9 test clips) | Pipeline checks only |

The "manual label review" (55 clips, 83.6% agreement) is simulated by `run_batch_simulation_review()` (`V1/recognition/label_review_tool.py:80-120`) and must not be cited as human verification.

---

## 4. Root causes of classification failure

1. **No ground truth.** No human-verified exercise labels exist anywhere: QEVD Part 1 comes without labels, V1 labels are a rule, the "review" was simulated, and RT trains on a generator. No classifier accuracy can be claimed, and no threshold has been tuned on real people.
2. **Geometry contract bug upstream.** Anisotropic coordinates distort every angle feature (see `POSE_CALIBRATION_AUDIT.md` RC-1). Rules tuned on isotropic synthetic data meet distorted real data.
3. **2D views can't separate several of the advertised exercises.** Curl and hammer curl, lateral and front raise from the front, hinge and side bend from the front. The definitions also list views (squat from front, curls from front) where the discriminating motion is foreshortened.
4. **Weak unknown rejection in the production path.** The amplitude-and-posture score accepts any movement that moves one rule's signal through a plausible range (side bend → RDL). The OOD/conformal machinery that addresses this exists only in the undeployed learned path.
5. **Train/inference mismatch in V1** (frame rate and window), together with a garbage class (`lateral_raise` = "other") and forced choice.
6. **Product wiring.** Auto is unreachable from Quick set; `confirm` has no handler; the lock is permanent with no re-recognition; the hard-coded exercise list goes out of sync with the registry.
7. **One mechanism carries several jobs.** Detecting movement, choosing the exercise, segmenting phases and rejecting unknowns are all folded into one score, so a failure in one looks like a wrong exercise.

---

## 5. Recommended architecture

Five separate stages, each with its own output, confidence and failure mode. **For the first milestones the user selects the exercise. Auto-detect comes back only once stage 3 has measured, human-labelled performance.**

```
landmarks (isotropic 2D + 3D world, per-joint confidence, t_capture)
   │
   ├─ S1 Movement detection ── "is the person moving rhythmically?"   (always on)
   │     motion energy of joint velocities over 2 s; periodicity via autocorrelation
   │     output: idle | moving | not_visible
   │
   ├─ S2 Exercise-family classification ── which joint system drives the motion
   │     families: elbow_flexion, knee_hip_flexion (squat), hip_hinge, shoulder_elevation, overhead_press
   │     rules on channel activity ratios (existing embedding.py descriptor is a good start)
   │     output: family + margin, or "unknown_family"
   │
   ├─ S3 Specific-exercise classification ── within a family, only where the view allows it
   │     needs: required view satisfied (from the setup fingerprint), posture constraints
   │     indistinguishable groups stay groups → ask the user ONCE, remember per workout item
   │
   ├─ S4 Movement-phase recognition ── per exercise, from the ExerciseDefinition
   │     progress signal against personal range → rest / toward_peak / peak / away / rest
   │     (this is the rep engine, not the classifier)
   │
   └─ S5 Unknown/ambiguous rejection ── runs alongside S2/S3
         abstain if: S1 not 'moving' | family margin < m | descriptor far from every known
         prototype (distance threshold set on real held-out data) | view not suitable
         output: unsupported → manual tracking, never a forced label
```

**Guidance**
- Keep the `Recognition` contract (`recognized | ambiguous | no_movement | unsupported` and the actions) and the equivalence groups. Both are sound.
- In the product, `confirm` must show the candidate group as tappable chips. The choice is stored on the workout item so the question isn't asked again.
- Allow re-recognition: if S2's family changes with high margin for 2 consecutive decisions, suggest a switch instead of silently keeping the lock.
- Move `recognize()` off the event loop (`asyncio.to_thread`).
- Don't train or adopt any learned classifier (RF, ST-GCN, encoder) until there's a human-labelled, subject-disjoint dataset. The minimum is set in `TEST_AND_EVALUATION_PLAN.md §3`. When there is one, the first model to compare against the rules is a small one (logistic regression or gradient boosting on the 35 interpretable features), not the ST-GCN, because the evidence so far gives no reason to expect more complexity to help.
- Retire C4 and its pseudo-label training set from anything user-facing.
