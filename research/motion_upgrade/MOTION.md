# `motion/` — learned spatial-temporal movement recognition

The recognition core of the next version. Skeleton sequences stay in their
natural `(T, J, C)` shape, an ST-GCN with a temporal transformer learns over
the body's connectivity graph, and a deterministic kinematic branch explains
what the model recognises.

Nothing in `biomechanics/`, `body_model/`, `exercises/` or `evaluation/` is
replaced. Those become the interpretable branch, and `motion/fusion.py`
returns the same dict keys `FitnessCVEngine` did, so `ui/dashboard.py` and
`ui/overlay.py` need no changes.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime        # only for motion/export.py
python -m motion.model              # shape + gradient selfcheck, ~1 second
```

`graph`, `tensors`, `dataset`, `phase`, `augment` and `calibration` have no
torch dependency, so data preparation and evaluation run on machines without
it. `motion/model.py`, `train.py`, `recognizer.py` and `export.py` need torch.

## Layout

| Module | Torch | What it owns |
| --- | --- | --- |
| `graph.py` | no | 15-node skeleton tree, spatial-configuration adjacency, mirror permutation |
| `tensors.py` | no | PoseFrame → `(T, J, C)`, gap handling, joint/bone/motion streams |
| `phase.py` | no | Phase derivation, rep segmentation |
| `dataset.py` | no | Window building, label provenance, group-disjoint splits |
| `augment.py` | no | Skeleton-aware augmentation |
| `calibration.py` | no | Temperature scaling, ECE, Brier, reliability |
| `model.py` | yes | ST-GCN backbone, temporal transformer, dual heads |
| `train.py` | yes | Training, leakage and provenance guards, evaluation |
| `recognizer.py` | yes | Inference adapter for the existing engine |
| `export.py` | yes | ONNX export, verification, benchmarking |
| `runtime.py` | no | Async capture/pose/inference threads, latency instrumentation |
| `fusion.py` | no | Merges the learned and kinematic branches |
| `live.py` | no | Entry point and headless profiler |

## Quick start

```bash
# 1. where does the time actually go — no camera, no checkpoint, no torch
python -m motion.live --profile --seconds 20

# 2. train (development run on rule-derived labels)
python -m motion.train --epochs 60 --allow-derived-labels --out models/motion_stgcn.pt

# 3. run it live
python -m motion.live --checkpoint models/motion_stgcn.pt

# 4. only after profiling says the model is the bottleneck
python -m motion.export models/motion_stgcn.pt --out models/motion_stgcn.onnx
```

## The two guards

**Leakage.** `assert_no_group_leakage` runs on every training job, not only in
tests. A leaked split makes the number go *up*, so there is no failure signal
to notice. `test_leakage_assertion_actually_fires` constructs the mistake
deliberately to prove the check works.

**Label provenance.** Every sample carries `label_source`. The current QEVD
labels come from `recognition/qevd_dataset.py:categorize_exercise_from_sequence`,
a deterministic kinematic rule — so a model trained and tested on them is
scored on reproducing that rule, not on exercise recognition. `train.py` prints
a banner and refuses to write a `release_ready` checkpoint without
`--allow-derived-labels`. Phase metrics are named
`phase_frame_agreement_with_rule`, never "accuracy".

The cheapest fix by a wide margin is a `clip_id,subject_id` CSV:

```bash
python -m motion.train --subjects data/subjects.csv --epochs 60
```

Without it, splits are video-disjoint but not subject-disjoint, and the model
can score well by recognising the *person* rather than the exercise.

## Architecture notes

**One model, not three milestones.** ST-GCN and the transformer are built as a
single configurable network. The ablations are config flags —
`--no-transformer`, `--fusion late`, `--adjacency uniform` — which is a better
experiment than three separately-tuned codebases, and one model to train, tune,
export and integrate.

**15 nodes, not 33.** The pose layer tracks 13 landmarks. `hip_center` and
`shoulder_center` are added because without them the left and right halves of
the body are disconnected components, and no amount of graph convolution moves
information between an arm and a leg.

**Early stream fusion by default.** Joint, bone and motion streams are
concatenated on the channel axis and share one backbone. The usual
multi-stream result in the literature trains a separate network per stream and
ensembles the softmax outputs, which multiplies training and inference cost —
the wrong trade for a model running live alongside pose estimation.
`--fusion late` is implemented for the comparison.

**Adaptive adjacency starts at zero.** The learnable residual on the adjacency
is zero-initialised, so training begins as an exact anatomical ST-GCN and only
deviates where the data pays for it. Random initialisation throws away the
topology prior on step one.

**Phases are physiological, not directional.** Rising elbow flexion is the
concentric curl; rising knee flexion is the *eccentric* squat descent. The
`increasing_is_concentric` flag lives per-exercise in `EXERCISE_SIGNALS`. Rep
segmentation is direction-agnostic for the same reason — assuming
concentric-first counts zero reps for every squat while looking healthy on the
other three exercises.

## Runtime design

```
capture thread   ~30 fps    always the freshest frame
     |  latest-wins slot (no queue)
pose thread      ~15-30     MediaPipe, one frame at a time
     |  ring buffer + per-frame kinematics
inference thread ~5-10      the ST-GCN over a window
     |  latest-result slot
consumer (UI)    ~30 fps    never blocks
```

Every handoff holds exactly one item and overwrites it. A queue would absorb a
burst and then feed the pose stage second-old frames — the pipeline would
report a healthy fps while showing the past. Drops are counted, not hidden.

Measured on one CPU core with the mock camera: capture 0.1 ms, MediaPipe pose
11.6 ms p50 / 12.3 ms p95, both stages holding 28 fps with zero drops. Re-run
`--profile` on the target machine before optimising anything.

## Rep counting is reported twice

The state machine stays authoritative by default. It is causal — it decides at
the moment the rep completes, from the current frame — whereas the learned
segmenter sees a whole window and can revise the recent past, which on screen
means a counter that goes backwards.

`state()` exposes `rep_count`, `learned_rep_count`, `state_machine_rep_count`
and `rep_count_agreement`. That disagreement is the experiment. When it closes
on real data, set `FusionConfig(rep_source="learned")`.

## Confidence

The displayed number is the temperature-scaled probability fitted on the
validation split, not `min(0.96, raw_score)`. Temperature scaling cannot change
the argmax, so it can never cost accuracy — the reason to prefer it over
isotonic regression until the validation set reaches roughly a thousand
samples. `state()` carries `confidence_is_calibrated` so nothing downstream
re-applies the old display cap.

## FastAPI Inference Service & Containerization

A production-grade, headless REST API interface wrapping the ST-GCN model and hybrid movement engine. The service accepts normalized pose landmarks via HTTP JSON payloads, performs inference on CPU without camera or display dependencies, and returns calibrated predictions, abstention flags, phase breakdowns, and joint attention scores.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check returning service status, timestamp, model readiness, and version |
| `GET` | `/model` | Model metadata, architecture, classes, calibrated temperature, and abstention threshold |
| `POST` | `/predict` | Raw ST-GCN exercise classification with calibrated confidence and class probabilities |
| `POST` | `/analyze` | Full hybrid pipeline: neural classification, phase segmentation, rep tracking, and joint attention |

### Running Locally

```bash
# 1. Install API dependencies
pip install -r requirements-api.txt

# 2. Start the FastAPI ASGI server
export PYTHONPATH="../Ver-1:."  # or $env:PYTHONPATH="..\Ver-1;." on PowerShell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Test with the example client
python examples/api_client.py
```

### Running with Docker

```bash
# Build and run the containerized service
docker build -f motion-upgrade/Dockerfile -t motion-stgcn-api .
docker run -p 8000:8000 motion-stgcn-api

# Or using Docker Compose
docker compose -f motion-upgrade/docker-compose.yml up
```

### Tests

```bash
# Run all motion tests including the API test suite (180 tests)
pytest tests/ -v
```

