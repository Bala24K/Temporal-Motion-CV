# Project Status: Completeness & Accuracy Audit

Honest answer to "is this complete, and can accuracy be improved": **no**, and
the specific gap worth naming precisely is: **before this pass, there were
zero trained ML models anywhere in this codebase.** Every "confidence,"
"score," or "estimate" up to that point came from deterministic
physics/geometry/rules, validated against known math or synthetic
properties — not from anything trained on data and measured for accuracy.
This document catalogs every place a number that looks like "accuracy" or
"confidence" appears, states plainly what it is and isn't validated
against, and lays out what's actually achievable right now vs. what's
blocked on infrastructure this environment doesn't have (a webcam, real
recorded reps, EMG hardware, OpenSim).

Read this alongside the README's V1/V2/V3 sections — this file is the
cross-cutting summary; the README has the per-component detail.

---

## The core distinction this whole project rests on

Everything built so far falls into exactly one of two categories:

1. **Deterministic (physics, geometry, rules).** Validated by checking it
   against *known correct answers* — analytic derivatives, hand-derived
   static equilibrium, the textbook Lagrange condition for a constrained
   optimization, exact rep-counting behavior on hand-constructed sequences.
   This kind of thing can be proven correct without any real-world data,
   the same way you can prove a sorting algorithm correct without running
   it on real customer records. **161 automated tests currently do this.**

2. **Statistical/learned (ML).** Can only be validated by measuring
   performance against real, independently-collected examples — you cannot
   prove a classifier's real-world accuracy by construction, only by
   testing it against data it didn't see, ideally from people/situations
   it didn't see either (V1's own "split by person" warning).

**Category 2 had exactly one thing built in it before this pass: nothing.**
Everything ML-shaped in this codebase was explicitly deferred, every time,
for the same reason: this sandbox has no webcam, no recorded human
movement, no EMG data. Training a model on synthetic ramps and reporting
its "accuracy" would produce a number that looks reassuring and means
nothing.

What changed in this pass: **exercise classification** (distinguishing
curl vs. squat vs. press vs. raise by movement geometry) is a case where
the *ground truth labels are geometric facts that can be defined
correctly*, not subtle human judgment calls — a squat involves large
knee-angle excursion and a curl doesn't, by definition, not by
observation. That's different in kind from "is this rep's form good" or
"how much is the bicep activated," which require real human variability
to mean anything. So this pass adds the project's first actual trained
model — clearly scoped to what synthetic data can honestly support, with
the accuracy number labeled for exactly what it measures (see Models
section below).

---

## Component-by-component status

### V1 — exercise detection core

| Component | Status | Validated against |
|---|---|---|
| Rep state machine | Built, tested | Hand-constructed synthetic sequences (jitter, partial ROM, occlusion, stuck timeout, variable frame rate) |
| Geometry engine (angles) | Built, tested | Exact trigonometry |
| Bicep curl / squat / shoulder press / lateral raise analyzers | **All four now built** (only curl existed before this pass — M6 was incomplete) | Synthetic pose sequences per exercise |
| Camera capture, MediaPipe wrapper, live UI | Written, syntax-checked only | **Nothing — never run against a real webcam in this sandbox.** FPS, latency, and calibration UX are still unverified against real hardware. |
| Rep-counting threshold values | Guessed, not calibrated | Nothing — explicitly flagged as starting points since the V1 README's first version |

### V2 — measurement layer

| Component | Status | Validated against |
|---|---|---|
| Normalization, symmetry, DTW trajectory comparison, rep/set analysis, personal baseline | Built, tested | Synthetic data + analytic properties (e.g. scale/rotation invariance proven exactly) |
| Exercise-quality ML model, movement embeddings, anomaly detection | **Not built** | Needs real recorded reps, including deliberately bad ones (V2 §6-7) — nothing here can substitute for that |

### V3 — biomechanics

| Component | Status | Validated against |
|---|---|---|
| Kinematics, inverse dynamics, muscle-redundancy optimization | Built, tested | Exact analytic derivatives; hand-derived static equilibrium; the analytic Lagrange condition for the optimizer |
| Muscle force parameters, anthropometric regression | Built, explicitly generic | **Not subject-specific. Not validated against any real person.** |
| OpenSim, real 3D pose, teacher/student distillation, EMG validation | **Not built — not possible in this sandbox** | OpenSim needs conda/compiled binaries this environment can't install; EMG needs hardware |

### Recognition — exercise classification (new in this pass)

| Component | Status | Validated against |
|---|---|---|
| Feature extraction (`recognition/features.py`) | Built, tested | Exact hand-computed angles |
| Synthetic dataset generator | Built, tested, **explicitly labeled synthetic** | Internal consistency (reproducibility, separability by construction) — not real data |
| Random Forest classifier | **Trained.** 95.7% accuracy, macro F1 0.957, on synthetic held-out subjects | See "What the accuracy number means" below — **synthetic-only, not a real-world claim** |
| Evaluation metrics (`evaluation/`) | Built, tested | Hand-computed confusion matrix / precision / recall / F1 on a worked toy example |

---

## What's added in this pass, in detail

1. **`exercises/squat.py`, `exercises/shoulder_press.py`, `exercises/lateral_raise.py`**
   — completes V1's M6. Same test-first discipline as bicep curl, and it
   caught real bugs worth naming: the squat's original knee-valgus metric
   (comparing one knee to its own hip-ankle line) turned out to conflate
   normal forward-and-down knee travel with genuine inward collapse —
   any 2D camera view shows *some* apparent horizontal knee shift as a
   knee simply bends, so a single-leg metric flagged perfectly normal
   squats as "valgus." Fixed by comparing both knees to each other
   instead (inter-knee vs. inter-hip width ratio), which is also the
   anatomically correct way to detect it. The press's default
   "partial lockout" threshold was initially set *above* the exercise's
   own achievable ROM, which would have flagged every single rep as
   partial regardless of how complete it was — caught by a test that
   checked the *good* rep case explicitly, not just the bad one.

2. **A live-phase-display redesign in `exercises/base.py`.** Found while
   building the new exercises: the original phase label ("CURLING" vs.
   "LOWERING") had dead code — the two internal flags that were supposed
   to distinguish them are always set and cleared together, so one
   branch could never be reached. Fixing that surfaced a deeper design
   issue: the debounced state used for robust rep-*counting* is
   deliberately laggy (that's the point — don't let one noisy frame flip
   the count), which makes it the wrong signal for a *live* phase
   display that should update in real time. Rep counting and phase
   display now use different signals from the same state machine:
   counting uses the debounced `confirmed_zone`, display uses the raw,
   undebounced zone plus recent angle direction. Regression-tested with
   phase-label-specific assertions (checking the actual displayed
   string), not just rep-counting behavior, which wouldn't have caught
   either bug.

3. **`recognition/` — the project's first trained model.** A Random
   Forest classifier that distinguishes which of the four exercises is
   being performed, from a single frame's joint-angle feature vector.
   - **Training data is synthetic-by-construction**: each exercise's
     defining geometric pattern (which joints move, by how much) is
     encoded directly from the same angle definitions used in the
     rule-based analyzers above — not fabricated human variability.
   - **Evaluated with a proper train/test split**, following V1's own
     "split by synthetic subject, not by rep" principle to avoid leakage.
   - **Actual result on synthetic held-out data**: 95.7% overall accuracy
     (macro F1 0.957). Per-exercise: shoulder_press 99.8%/100.0%
     precision/recall (very distinct — sweeps both elbow and
     shoulder-elevation angle together); bicep_curl 89.9%/97.4%; squat
     94.8%/94.6%; lateral_raise 99.1%/90.8%.
   - **A genuine, informative confusion pattern, not noise**: the
     confusion matrix shows curl and lateral-raise getting mixed up with
     each other more than with squat or press. This makes geometric
     sense on inspection — at the "rest" extreme of either movement's
     cycle (arm near the body, low shoulder-elevation), a single frame
     from a curl and a single frame from a lateral raise can look nearly
     identical; the defining difference only shows up once the movement
     is underway. This is exactly why V1's own roadmap (§16) suggests
     moving from single-frame to temporal (multi-frame windowed)
     features next — this result is a concrete, data-backed reason to
     prioritize that, not just a theoretical suggestion anymore.
   - **The accuracy number measures exactly one thing**: how well the
     classifier recovers the geometric definitions it was given, on more
     synthetic examples generated the same way. It is **not** a claim
     about real-world accuracy on an actual camera feed with real
     tracking noise, real anatomical variation, or exercise variants
     (sumo squats, incline presses, etc.) the synthetic generator doesn't
     model. Run `python train_exercise_classifier.py` to reproduce.

4. **`evaluation/classification.py`, `evaluation/reps.py`** — the metrics
   infrastructure the V1/V2 docs specify (accuracy, precision, recall, F1,
   confusion matrix; rep-counting error) but that never existed before
   this pass. Tested against hand-computed small examples. This is what
   any future real-data evaluation will actually be measured with.

## What's next, honestly ordered by what's actually achievable

1. **Get this running against a real webcam.** Nothing else on this list
   matters until someone runs `app.py` on real hardware and reports back
   FPS, latency, and whether the rep-counting thresholds are even close
   to right for a real body. This is the single highest-value next step
   and the only one that was never blocked by "no data" — it's blocked by
   "no camera in this sandbox."
2. **Record real reps** (even just one person, one camera, a few
   exercises, some deliberately bad reps per V2 §7) — this unlocks
   real accuracy measurement for the classifier above, real threshold
   calibration, and is the prerequisite for literally everything else
   in V2's ML section and V3's teacher/student pipeline.
3. **Move the classifier from single-frame to windowed/temporal
   features** — this pass's curl/lateral-raise confusion at the "rest"
   extreme is a concrete, measured reason to do this next, not just a
   theoretical one. A short rolling window of frames (V1 §16) would let
   the classifier see the *shape* of the movement, not just one static
   pose, which should directly resolve the observed confusion.
4. **Wire V2's symmetry/DTW modules into live dual-arm tracking** — built
   and tested standalone, not yet connected to a running exercise
   analyzer. Shoulder press and lateral raise are the natural first
   candidates (both arms move together).
5. **Camera-angle sensitivity study** — flagged as a risk since the V1
   README and never quantified. Answerable without a real camera
   (synthesize 3D motion, project it from different simulated viewpoints,
   measure how much the recovered 2D angle deviates from ground truth).

---

# Structured body representation + multi-model comparison

A later request asked to move toward the "real vision" of the project:
real exercise video (naming QEVD, M3GYM as candidate datasets), a
structured geometric body representation (joints/limbs as first-class
objects, not a handful of hand-picked angles) as the intermediate
representation, sequences of that representation as the movement
signal, and a genuine multi-model ML comparison rather than committing
to Random Forest by default.

## The hard blocker, stated up front

**Real exercise-video datasets are not obtainable in this sandbox.**
Checked what QEVD and M3GYM actually are, so this is accurate rather than
a guess:

- **QEVD** (Qualcomm Exercise Video Dataset): 474+ hours of video across
  three subsets — QEVD-Fit-300K (~289k short clips, 148 exercises with
  variations including deliberately bad form and varied pacing),
  QEVD-Fit-Coach (149 full workout videos ~3.5 min each with coach-style
  feedback annotations), plus Q&A annotations. Hosted at
  `qualcomm.com/developer` / `developer.qualcomm.com`.
- **M3GYM** (CVPR 2025): a real-gym, multi-view (8 cameras), multi-person
  pose dataset — 82 sessions, 50+ subjects, 47 million frames, 2D/3D
  keypoints plus mesh and 500+ labeled actions with sports-expert
  completeness assessments. An academic dataset, typically requiring a
  request/access process from the authors' project page.

Neither host is reachable from this sandbox's network (restricted to
package registries — PyPI, npm, GitHub for code — not general web or
dataset-hosting access), and even if it were, downloading and processing
hundreds of hours of video is a different scale of task than this
environment is set up for. This is the same class of hard blocker as
OpenSim was in V3: **honestly out of reach here, not a matter of trying
harder.**

## What's built instead — dataset-agnostic, ready the moment real data exists

Everything below operates on `PoseFrame`/`PoseSequence` — the same types
`pose/detector.py`'s MediaPipe wrapper already produces from a real video
frame. **The only missing piece to run this on QEVD or M3GYM is the video
file → MediaPipe step, which V1's `pose/detector.py` already does.**
Nothing in this section's code needs to change to consume real data
instead of synthetic data — only `recognition/pose_sequence_synthesis.py`
(the synthetic generator) would be replaced by a real video loader.

- **`body_model/skeleton.py`** — the structured geometric representation
  itself. Every limb (9: torso + bilateral upper-arm/forearm/thigh/shin)
  gets a `Limb` object with position, length, and orientation, built from
  the skeleton's own connectivity graph (`LIMB_DEFINITIONS`), not
  hand-picked per exercise. Every joint angle (8: elbows, knees,
  shoulders, hips) is derived from which limbs share an endpoint
  (`JOINT_ANGLE_DEFINITIONS`), generically — the same code produces all 8
  regardless of which exercise is being performed. Cross-checked against
  the existing, independently-tested `angle_3pt` to make sure the
  structured representation doesn't quietly compute a different number
  for the same three points.

- **`body_model/sequence.py`** — the temporal layer: a rolling buffer of
  `SkeletonFrame`s with velocity/acceleration for every limb orientation
  and every joint angle (reusing `biomechanics/kinematics.py`, already
  validated against exact analytic derivatives in V3 — not reimplemented
  a third time), plus live left/right symmetry (orientation and velocity
  difference between counterpart limbs). Explicitly returns `None` rather
  than interpolating through a tracking gap mid-window.

- **`recognition/skeleton_features.py`** — the systematic feature vector
  (59 features: 9 limbs × 3 + 8 joints × 3 + 4 symmetry pairs × 2) read
  off the structured representation, replacing the earlier 8 hand-picked
  angles. The representation doesn't know what a bicep curl is; it just
  describes the body and how it's moving.

- **`recognition/pose_sequence_synthesis.py`** — since the structured/
  temporal representation needs actual per-frame landmark sequences (not
  just an isolated angle), this generates full-body synthetic pose
  sequences via forward kinematics, driven by the same "which joints move"
  geometric definitions as before. **Caught a real bug during
  development**: the leg segments were initially generated pointing "up"
  from the hip instead of "down" (an inverted orientation-convention
  error), which a numerical sanity check — printing actual joint angles
  and landmark y-coordinates rather than assuming the math was right —
  caught before it silently corrupted every downstream result. Same
  synthetic-vs-real caveats as `dataset_synthesis.py` apply; see that
  module's docstring.

- **`recognition/temporal_model.py`** — a small 1D-CNN classifier
  (PyTorch) operating on the *raw* per-frame sequence, following V1 §16's
  own suggested model ladder (Random Forest → XGBoost → MLP → 1D CNN →
  LSTM/GRU → Transformer, tried in that order rather than reaching for
  the most exotic architecture first). Unlike the tree/MLP models, this
  one learns temporal patterns itself rather than being handed
  pre-computed velocity/acceleration.

- **`recognition/model_comparison.py`, `compare_models.py`** — trains
  Random Forest, XGBoost, MLP, and the temporal CNN on **the exact same
  underlying synthetic windows** (verified by a test that checks the two
  dataset generators consume their shared random seed identically), so
  the comparison is apples-to-apples, and evaluates all four with the
  same `evaluation/classification.py` metrics used for the earlier
  classifier.

## Actual comparison results (synthetic data — see disclaimer above)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 98.3% | 0.983 |
| XGBoost | 97.6% | 0.976 |
| MLP | 97.5% | 0.975 |
| Temporal 1D-CNN | **100.0%** | **1.000** |

Two findings worth stating plainly, both surfaced by actually running the
comparison rather than assumed beforehand:

1. **The MLP initially scored only 67.1%** — a big enough gap from the
   other three (all 97%+) to be suspicious rather than "MLP is just
   worse here." Tested the hypothesis directly: retrained with a
   `StandardScaler` on the same train/test split, accuracy jumped to
   97.5%. Trees split on per-feature thresholds, so raw-scale angles and
   large-magnitude accelerations don't bother Random Forest or XGBoost;
   a neural net's gradient-based optimization is sensitive to input
   scale. Fixed in `model_comparison.py` rather than reported as a weak
   MLP result — an unscaled-neural-net bug isn't a finding about
   architecture, it's a bug.
2. **Bicep curl and lateral raise remain the hardest pair to
   distinguish for every model trained on final-frame features**
   (Random Forest/XGBoost/MLP all score lowest, ~0.95-0.97 F1, on
   exactly those two exercises) — consistent with the single-frame
   8-angle classifier's confusion pattern reported earlier in this
   document. That consistency across two different feature
   representations and three different model families makes this a
   credible finding, not a fluke: the ambiguity is real, it's at the
   "rest" extreme of either movement's cycle, and it's a property of
   the *information available in a single frame's kinematics*, not of
   any one model. **The temporal CNN, given the raw sequence directly
   instead of pre-computed velocity/acceleration, resolves it
   completely (1.000 F1 on both).** That's a specific, actionable
   result: for this confusion pair, giving the model the sequence
   itself works better than engineering derivative features by hand.

As with every accuracy number in this project: this measures recovery of
synthetic geometric definitions, not real-world performance. The
difference this time is that the code producing it is dataset-agnostic —
the moment real pose sequences exist (QEVD, M3GYM, or a handful of
self-recorded videos through V1's own `pose/detector.py`), this exact
pipeline is what would need to run against them, unmodified except for
the data source.

## What's next, for this section specifically

1. **A real video → this pipeline, at any scale.** Even a handful of
   self-recorded clips run through `pose/detector.py` would upgrade every
   number in this section from "recovers a synthetic definition" to "the
   first real measurement this project has ever produced." This matters
   more than getting QEVD/M3GYM access specifically.
2. **Try LSTM/GRU**, per V1 §16's own ladder — the 1D-CNN's strong result
   suggests the temporal signal is genuinely learnable; whether a
   recurrent architecture does better, worse, or just slower is an open,
   cheap-to-test question with the harness already in place.
3. **Extend `JOINT_ANGLE_DEFINITIONS`/`SYMMETRY_PAIRS`** to cover ankles/
   wrists if a future exercise needs them — the representation was built
   to be extended by editing the topology dictionaries in
   `body_model/skeleton.py`, not by writing new per-exercise angle code.
