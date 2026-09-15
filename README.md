# Fitness CV — Real-Time Exercise Recognition & Biomechanical Analysis Engine

An open-source, local-first computer vision pipeline for real-time exercise classification, repetition counting, movement phase tracking, and biomechanical form analysis using webcam video feeds.

---

## ⚠️ Important Dataset & Labeling Methodology Notice

> [!IMPORTANT]
> **Heuristic Labeling Disclaimer**: The video clips from `QEVD-FIT-300k-Part-1` used in these experiments were not provided with human-verified ground-truth exercise labels. Instead, movement labels (`squat`, `shoulder_press`, `bicep_curl`, `lateral_raise`) were generated automatically via deterministic kinematic rules based on range of motion (knee excursion, elbow excursion, wrist elevation).
>
> The benchmark and ablation numbers reported below evaluate how effectively ML models recover these kinematic movement patterns from raw video sequences under strict video-level data isolation. They do not claim population-level real-world recognition accuracy without further authoritative human label validation.

---

## Key Features & Pipeline Architecture

```
Webcam / Video Input
        ↓
MediaPipe Pose Landmarker
        ↓
Landmark Normalization (Pelvis origin, torso scaling, body alignment)
        ↓
Modular Temporal Feature Extraction (Landmarks, Angles, Velocity ΔF, Acceleration Δ²F, Symmetry & ROM)
        ↓
Exercise Classifier (Random Forest / Logistic Regression / 1D-CNN)
        ↓
Repetition State Machine & Form Analyzer (Bicep Curl, Squat, Shoulder Press, Lateral Raise)
        ↓
Live Real-Time Screen Overlay (Exercise, Confidence %, Rep Count, Phase, Form Warnings, FPS/Latency)
```

1. **Zero Data-Leakage Video-Level Splitting**: Stratified 3-way split (70% Train / 15% Val / 15% Test) partitioned strictly by video identity. No frames, temporal windows, or derived samples from the same video cross split boundaries.
2. **Multi-Seed Statistical Evaluation**: Evaluates models across 5 deterministic random seeds (`42, 101, 2024, 7, 999`), reporting Mean $\pm$ Standard Deviation and 95% Confidence Intervals.
3. **Modular Feature Group Ablation**: Independently switchable feature groups (`raw_landmarks`, `joint_angles`, `velocity`, `acceleration`, `range_of_motion`, `symmetry`).
4. **Manual Label Quality Audit Tool**: Interactive and batch review workflow (`recognition/label_review_tool.py`) evaluating ambiguous clips near kinematic thresholds.
5. **Prediction Error Analysis**: Logs test predictions, confidences, and top confusion pairs (`recognition/error_analysis.py`).
6. **Unified Fitness-CV Engine & Reusable Inference API**: High-level `ExerciseRecognizer` and `FitnessCVEngine` APIs with complete model bundle serialization (`models/exercise_recognizer_bundle.joblib`).
7. **Live Webcam Demo (`webcam_recognizer.py`)**: 100% local CPU processing with real-time overlay, rep counting, form feedback, and FPS/latency tracking.

---

## Statistical Benchmark Results (5 Deterministic Seeds)

Evaluated on a balanced dataset of 100 clips (25 per class; 72 Train, 16 Val, 12 Test):

| Model | Accuracy (Mean $\pm$ Std) | 95% CI (Accuracy) | Macro F1 (Mean $\pm$ Std) | 95% CI (Macro F1) |
|---|---|---|---|---|
| **Random Forest** | **75.0% $\pm$ 10.5%** | **$\pm$ 9.2%** | **0.722 $\pm$ 0.127** | **$\pm$ 0.112** |
| **XGBoost** | **73.3% $\pm$ 16.2%** | **$\pm$ 14.2%** | **0.694 $\pm$ 0.195** | **$\pm$ 0.171** |
| **Temporal 1D-CNN** | **68.3% $\pm$ 6.2%** | **$\pm$ 5.5%** | **0.645 $\pm$ 0.093** | **$\pm$ 0.082** |
| **Logistic Regression** | **66.7% $\pm$ 9.1%** | **$\pm$ 8.0%** | **0.636 $\pm$ 0.104** | **$\pm$ 0.091** |
| **MLP Classifier** | **58.3% $\pm$ 7.5%** | **$\pm$ 6.5%** | **0.549 $\pm$ 0.070** | **$\pm$ 0.061** |
| **Random Baseline** | 15.0% $\pm$ 12.2% | $\pm$ 10.7% | 0.131 $\pm$ 0.127 | $\pm$ 0.111 |
| **Majority Baseline** | 25.0% $\pm$ 0.0% | $\pm$ 0.0% | 0.100 $\pm$ 0.000 | $\pm$ 0.000 |

> *Note on Statistical Confidence*: $\pm\text{Std}$ and $\pm\text{CI}$ values measure split-to-split variance across random train/val/test partitions on this dataset subset ($N=100$ clips, 12 test clips/seed). They measure evaluation sensitivity to video partitioning, not population-level ground-truth confidence.

---

## Feature Group Ablation Study

Evaluated Random Forest across 5 cumulative feature levels:

| Ablation Level | Feature Groups Included | Features | Test Accuracy | Macro F1 | Key Finding |
|---|---|---|---|---|---|
| **Level 1** | Raw Landmarks Only | 195 | 58.3% | 0.564 | Baseline landmark coordinates |
| **Level 2** | + Joint Angles | 235 | 58.3% | 0.564 | Joint orientation angles |
| **Level 3** | **+ Velocity ($\Delta F$)** | **470** | **75.0%** | **0.727** | **Peak Performance (+16.7% Acc, +0.163 F1)** |
| **Level 4** | + Acceleration ($\Delta^2 F$) | 705 | 66.7% | 0.627 | Higher derivative adds variance |
| **Level 5** | + Symmetry & ROM | 870 | 66.7% | 0.627 | Full feature matrix |

---

## Quick Start & Usage

### 1. Run Automated Unit Tests (216 Tests Passing)
```bash
python -m pytest
```

### 2. Run Reproducible Benchmark Suite
```bash
python benchmark_qevd.py --target-per-class 25 --save-plots
```

### 3. Run Label Quality & Ambiguity Audit
```bash
python -m recognition.label_audit
```

### 4. Run Manual Label Review Tool
```bash
python -m recognition.label_review_tool --batch-limit 55
```

### 5. Run Detailed Prediction Error Analysis
```bash
python -m recognition.error_analysis
```

### 6. Run Live Real-Time Webcam Demo
```bash
python webcam_recognizer.py --camera-id 0
```

---

## Reusable Python API

```python
from recognition.exercise_recognizer import ExerciseRecognizer
from recognition.fitness_pipeline import FitnessCVEngine

# 1. Exercise Recognition API
recognizer = ExerciseRecognizer.load_bundle("models/exercise_recognizer_bundle.joblib")
result = recognizer.predict_exercise("my_video.mp4")

print(f"Exercise: {result.exercise}")
print(f"Confidence: {result.confidence * 100:.1f}%")
print(f"Probabilities: {result.probabilities}")

# 2. Complete Fitness-CV Engine (Recognition + Rep Counting + Form Feedback)
engine = FitnessCVEngine(bundle_path="models/exercise_recognizer_bundle.joblib")
res = engine.process_frame_sequence(pose_frames)
print(f"Exercise: {res['exercise']}, Reps: {res['rep_count']}, Phase: {res['phase']}")
for fb in res["form_feedback"]:
    print(f"Form ({fb['status']}): {fb['message']}")
```

---

## Project Structure

```
fitness_cv/
├── benchmark_qevd.py                  # Benchmark runner, multi-seed stats, ablation & plots
├── webcam_recognizer.py               # Real-time webcam demo with live UI overlay
├── recognition/
│   ├── feature_engineering.py         # Modular temporal feature groups (landmarks, angles, vel, acc, sym, ROM)
│   ├── qevd_dataset.py                # Dataset loader, pose caching, zero-leakage 3-way video split
│   ├── exercise_recognizer.py         # Production inference API & model bundle serialization
│   ├── fitness_pipeline.py            # Unified FitnessCVEngine (Recognition + Rep Counter + Form Analyzer)
│   ├── label_audit.py                 # Kinematic threshold audit & ambiguity detection
│   ├── label_review_tool.py           # Manual review workflow & agreement calculator
│   └── error_analysis.py              # Prediction error logger & top confusion pair analyzer
├── exercises/
│   ├── base.py                        # RepStateMachine (two-threshold hysteresis, time debounce)
│   ├── bicep_curl.py                  # Bicep curl analyzer & form thresholds
│   ├── squat.py                       # Squat depth & knee valgus analyzer
│   ├── shoulder_press.py              # Overhead press analyzer
│   └── lateral_raise.py               # Arm elevation & shrug analyzer
├── models/
│   ├── pose_landmarker.task           # MediaPipe pose model
│   └── exercise_recognizer_bundle.joblib # Serialized production model bundle
├── analysis/                          # Confusion matrices, ablation plots, ambiguity reports
└── tests/                             # 216 unit tests covering all modules & pipeline components
```
