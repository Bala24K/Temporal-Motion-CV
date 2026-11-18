# Repetition Counting and Movement Analysis Audit

Aliases (`RT/`, `V1/`, `MU/`) and evidence tags are defined in `PROJECT_AUDIT.md`.

---

## 1. Counters that exist

| Counter | Where | Drives |
| --- | --- | --- |
| **`RepCounter`**, one generic progress-based machine for all 9 definitions | `RT/motion_engine/repetition.py` | RT live sets, offline `/api/analyze`, recognition periodicity |
| `RepStateMachine`, a two-threshold Schmitt trigger with time-based dwell, stuck timeout and tracking-loss "degraded" state | `V1/exercises/base.py` | V1 analyzers: `bicep_curl.py`, `squat.py`, `shoulder_press.py`, `lateral_raise.py` |
| Phase derivation / rep segmentation | `MU/motion/phase.py` | Labels for the ST-GCN phase head (research) |

### 1.1 RT `RepCounter` as implemented

- Input: the definition's `rep.signal` (an angle) per frame. Frames run only when `QualityTracker` says `ok` (`RT/motion_engine/session.py:50-51`).
- EMA smoothing, α = 0.5 **per frame** (`repetition.py:74`).
- `progress = (ema − rest) / (peak − rest)`, using the definition's constant `rest`/`peak`.
- Start a rep when progress > **0.20**. Finish when progress < **0.15** (`ENTER`/`EXIT`, line 15).
- On finish: peak progress < `partial_progress` (0.30) or duration < 0.15 s → discarded as noise. Peak ≥ `count_progress` (0.55, RDL 0.55) and duration ≥ 0.3 s → counted. Otherwise → partial (not counted).
- An active rep lasting more than `max_rep_seconds` (12 s) is abandoned silently. It restarts immediately if progress is still above 0.20.
- Phase = concentric/eccentric by the sign of the frame-to-frame progress change.

### 1.2 Per-exercise definitions (RT)

| Exercise | Rep signal | rest → peak | Counted when progress ≥ 0.55, i.e. signal passes | **Rep only completes after returning past** (progress < 0.15) | Preferred view | Motion plane |
| --- | --- | --- | --- | --- | --- | --- |
| squat / goblet_squat | knee angle (hip-knee-ankle), mean L/R | 172 → 85 | ≤ 124° | **≥ 159°** | side | sagittal |
| bicep_curl / hammer_curl | elbow angle, mean L/R | 170 → 45 | ≤ 101° | **≥ 151°** | **front** | sagittal |
| shoulder_press | elbow angle, mean L/R | 90 → 172 | ≥ 135° | **≤ 102°** | front | frontal |
| lateral_raise / front_raise | upper-arm angle from vertical | 10 → 90 | ≥ 54° | ≤ 22° | front / side | frontal / sagittal |
| overhead_triceps_extension | elbow angle | 40 → 168 | ≥ 110° | ≤ 59° | side | sagittal |
| romanian_deadlift | hip angle (shoulder-hip-knee) | 172 → 100 | ≤ 132° | ≥ 161° | side | sagittal |

---

## 2. Findings by topic

| Topic | Finding | Evidence |
| --- | --- | --- |
| **State transitions** | Two states (inactive and active), with phase inferred per frame. No explicit rest, bottom or top states and no dwell requirement. A single noisy frame across 0.20 starts a rep; across 0.15 ends it. | [Inspected] |
| **Angle calculations** | `_angle` is correct for 2D isotropic vectors (`RT/motion_engine/primitives.py:25-30`). On real input it gets anisotropic coordinates: true 45° → 29.4° at 16:9. `side: "mean"` averages left and right **even when one side is occluded or hallucinated** (side view). `nanmean` silently falls back to one side if the other is NaN, so the signal source changes mid-rep. | [Reproduced] `verify_rt.py §3` |
| **ROM thresholds** | Textbook constants. Not personalised (the `set_range` hook is never called), not view-dependent, and never fitted to real data. Front-view curl and front-view squat can't physically reach the projected peak for many people [Hypothesis: confirm with recorded front-view sets]. | [Inspected] |
| **Hysteresis** | 0.05 progress band (0.20 / 0.15), roughly 4–6° on a 125° range. Smaller than typical MediaPipe angle jitter at distance, so the EMA is what actually prevents chatter. | [Inspected] |
| **Return-to-rest requirement** | **The dominant miss mechanism.** A rep is counted only after progress drops below 0.15, i.e. back within about 15% of the textbook rest angle. Incomplete re-extension, which is common in real training, means the rep never completes. The 12 s abandon then discards it silently: no count, no partial, no message. | [Reproduced] curl re-extending to 160° → 5/5, **150° → 1/5, 140° → 0/5**; press bottom 100° → 5/5, **110° → 0/5** |
| **Temporal consistency / frame drops** | Timing comes from client timestamps (processing time). The EMA is per frame, so its time constant scales with the frame rate. Dropped frames (latest-wins) just make `dt` longer. There is no velocity plausibility check. | [Inspected] |
| **Tracking loss** | When quality is bad the frame is skipped and the active rep stays open indefinitely (up to 12 s). If tracking comes back at rest, the rep closes with the gap counted into its duration, which inflates tempo. No "tracking lost" state is surfaced to the counter. V1's `degraded` and re-anchor logic handles this better. | [Inspected] |
| **Partial repetitions** | Distinguished well when the user *does* return to rest (peak 0.30–0.55 → partial with a message). When the user doesn't return, the partial disappears. | [Reproduced] |
| **Duplicate repetitions** | No double counting found on synthetic data. Risk: no dwell, a narrow hysteresis band, and `nanmean` side switching could create a spurious crossing. Not reproducible without real jitter. | [Hypothesis] |
| **Movement interruptions** | A long hold mid-rep (13 s at squat bottom) is still counted once, but the abandon-and-restart throws away the descent, so `duration` and `toward_peak_seconds` cover only the post-restart part. The tempo evidence for that rep is wrong. | [Reproduced] (count 1, timing truncated) |
| **Incorrect exercise identification** | With a wrong exercise selected, the counter runs on the wrong joint. A squat while "bicep_curl" is selected (the Quick set bug) counts nothing, or counts arm movement. | [Inspected] |
| **State reset & exercise switching** | `select_exercise` builds a **new `LiveSession`**. Reps already counted are discarded (not saved) and the UI resets them to 0 (`useLiveSet.ts:31`). The switch message itself can be dropped (latest-wins race). V1 doesn't reset its analyzers on an automatic switch. | [Reproduced] race; [Inspected] reset |
| **View / plane mismatch** | Exercises are advertised for views where the rep signal is foreshortened (curl preferred view = front; squat lists front). In a side view, `required_joints` includes both sides at vis ≥ 0.5, which conflicts with occlusion. A squat at 16:9 without aspect correction fails `rom` and `tempo` on a clean set. | [Reproduced] `verify_rt.py §9` |

### 2.1 Form analysis (`RT/motion_engine/form.py`)

**Good, keep it:** every rule produces an `Evidence` record with measured value, threshold, pass/fail, severity and message. The form score is a transparent weighted share of rules passed. Personal insights are kept separate from pass/fail.

**Problems**
- Thresholds (`trunk_lean ≤ 12°`, `elbow drift range ≤ 18°`, `knee symmetry ≤ 12°`…) have no citation and no validation data.
- Rules are evaluated **only for counted reps**, so no feedback reaches the user during the rep, while it would still help.
- Stability/limit rules use raw `max − min` and `max`, which are sensitive to one bad frame. Robust percentiles (P5/P95) would not be.
- `symmetry` compares left and right angles in 2D, which is invalid in side views (far side hallucinated) and distorted in front views of sagittal motion.
- 2D trunk lean from the front mixes up a forward lean with a sideways bend.
- `tempo` uses timestamps that include inference jitter.

### 2.2 V1 counters (reference)

`RepStateMachine` adds what RT lacks: time-based dwell (0.08 s), an explicit armed cycle, a stuck timeout that suppresses re-confirmation, tracking-loss detection after 2 s with re-anchoring, and minimum ROM of 15°. Thresholds are also constants: curl 155/55, squat 165/100, press 160/90, raise 75/15. It has the same incomplete-return weakness: a curl must come back to ≥ 155°. Each analyzer picks one side ("auto" resolves once) instead of averaging. **Port the robustness ideas, not the code.**

---

## 3. Root causes of repetition-count errors

1. **Fixed textbook rest/peak with a strict return-to-rest rule.** People who don't fully re-extend never complete a rep, and the miss is silent.
2. **Distorted and foreshortened angle signals.** No aspect correction; 2D angles for out-of-plane motion; front-view sagittal exercises; left/right averaging in side views.
3. **No tracking-loss model.** Skipped frames leave reps hanging; no degraded state; timing gets inflated.
4. **No dwell or debounce, and a narrow hysteresis band.** Robustness depends on a frame-rate-dependent EMA.
5. **Session lifecycle bugs.** Switching exercise discards reps; the switch message can be dropped; Quick set runs the wrong exercise.
6. **No ground truth.** Nobody has ever compared a counter to human-counted real sets, so none of the above showed up in testing.

---

## 4. Recommended design

### 4.1 Reusable movement primitives (keep and extend `primitives.py`)

| Primitive | Status | Change |
| --- | --- | --- |
| `joint_angle(a,b,c, side)` | exists | Add a `source: "2d"|"3d"|"auto"` parameter. Add `side: "visible"` (the side with higher visibility, chosen once per set with hysteresis) instead of `mean` for side views. |
| `segment_vertical_angle` | exists | Same `source` option |
| `trunk_lean` | exists | Split into `trunk_flexion` (sagittal) and `trunk_lateral_bend` (frontal), with the valid view declared |
| `relative_height`, `joint_distance`, `joint_x_offset` | exist | Make sure they take scale-free (torso-normalized) inputs at **live** time, not the calibration-time torso length |
| `velocity` | Exists, but **would crash live**: `LiveSession` computes signals on 1-frame arrays and `np.gradient` raises `ValueError` on length 1 [Reproduced]. No current definition uses it, so the bug is latent. | Compute on a ring buffer of recent frames |
| new `range_progress(signal, personal_range)` | — | Progress against the user's captured range |
| new `stillness(window)` | — | Dwell and hold detection |
| new `visibility_ok(joints, min)` | — | Per-rule evaluability (a rule without data must report "not evaluated", never "passed") |

### 4.2 Declarative `ExerciseDefinition` (evolution of the existing schema)

```yaml
id: squat
version: 2
family: knee_hip_flexion
views:
  supported: [side, "45deg"]            # front removed: knee flexion is foreshortened
  required_view_check: {primitive: facing_angle, min_abs_deg: 60}
required_landmarks:                      # per view; visibility gates per joint
  side:  {joints: [hip, knee, ankle, shoulder], side: visible, min_visibility: 0.7}
signals:
  knee:  {primitive: joint_angle, joints: [hip, knee, ankle], side: visible, source: auto}
  trunk: {primitive: trunk_flexion, view: side}
phases: [rest, descent, bottom, ascent]  # descent = eccentric
rep:
  signal: knee
  population_range: {rest: [160, 180], peak: [70, 110]}   # allowed bounds for personal capture
  personal_range: required                                 # captured once per setup
  start:    {progress_gt: 0.25, dwell_s: 0.10}
  peak:     {progress_gte: 0.60}                           # min depth to count (of personal range)
  complete: {progress_lt: 0.35, dwell_s: 0.15}             # relaxed return; OR velocity reversal + next descent
  timing:   {min_rep_s: 0.6, max_rep_s: 15, stall_s: 20}
  partial:  {peak_progress: [0.30, 0.60]}
invalid_movement:
  - {when: {view_check: failed},           action: pause, message: "Turn side-on to the camera"}
  - {when: {landmarks_missing_s: 1.0},      action: hold_state, message: "We lost sight of your knees"}
  - {when: {landmarks_missing_s: 3.0},      action: abandon_rep}
form_checks:
  - {id: depth,    type: rom,       min: 0.85, severity: warn,  evaluable_if: {visible: [hip, knee, ankle]}}
  - {id: chest_up, type: limit,     signal: trunk, robust: p95, max: 45, severity: warn}
  - {id: tempo,    type: tempo,     min_phase_s: 0.6, severity: info}
confidence:
  min_landmark_visibility: 0.7
  min_view_confidence: 0.8
  min_calibration_quality: 0.6
```

Engine rules that go with it:
- **Completion without a full return.** A rep completes when progress falls below `complete.progress_lt` **or** when a velocity reversal at the bottom of the range is followed by the start of the next rep. Users who stop short are still counted, and their re-extension depth is reported as a form observation, not used as a counting condition.
- **Personal range capture.** Before the first set of an exercise (or after the setup changes), ask for 2–3 slow reps. `rest` = median of the top 10% of the signal, `peak` = median of the bottom 10%, clamped to `population_range`. Store it per (user, exercise, view, setup fingerprint).
- **Tracking states:** `tracking → degraded (missing < 3 s, hold state) → lost (abandon the rep, re-anchor on recovery)`, shown to the user.
- **Exercise switch** closes the current set (saving it) before starting a new one, and is acknowledged on a reliable control channel.
- **Rule evaluability:** a rule whose landmarks were not visible for at least 80% of the rep reports `not_evaluated`.

### 4.3 Smallest reliable initial exercise set

| Exercise | View | Why it is in | What it needs first |
| --- | --- | --- | --- |
| **Bodyweight / goblet squat** | side (45° allowed) | Big, unambiguous knee and hip excursion; full body visible; highest user value | Visible-side primitive, personal range, side-view landmark gating |
| **Bicep curl** (curl group: hammer and dumbbell curls counted the same) | side or 45° (front only with 3D source after evaluation) | Upper body only, so camera placement is easy | Visible-side elbow angle, relaxed completion |
| **Shoulder press** | front | Frontal-plane motion, well suited to 2D | Aspect fix, relaxed completion |
| **Lateral raise** | front | Frontal-plane motion | Aspect fix |

**Defer:** front raise (same as lateral raise from the front; needs the side view), Romanian deadlift (needs a side view and hinge-vs-bend separation; wrongly accepts a side bend today), overhead triceps extension (wrist often out of frame overhead), goblet squat as a *separate* exercise (count it as squat). Stage the vertical slice: **squat (side) first, then curl, then press and raise.**
