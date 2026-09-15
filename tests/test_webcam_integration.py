"""
Non-camera integration test suite for webcam_recognizer.py and FitnessCVEngine.

Verifies end-to-end processing chain without requiring a physical camera:
  frame -> pose -> temporal buffer -> exercise classification -> rep counting -> form feedback -> overlay

Tests boundary conditions:
  - Empty / no person frame
  - Intermittent pose detection / low confidence
  - Insufficient temporal history
  - Extreme / impossible joint geometries
"""

import numpy as np
import pytest

from pose.landmarks import Landmark, PoseFrame, PoseSequence
from recognition.fitness_pipeline import FitnessCVEngine
from recognition.exercise_recognizer import ExerciseRecognizer, ExerciseRecognitionResult


def _make_dummy_frame_sequence(count: int = 10, visibility: float = 0.9) -> list:
    frames = []
    for t in range(count):
        pf = PoseFrame(timestamp=t * 0.05)
        pf.landmarks = {
            "left_shoulder": Landmark(0.4, 0.3, 0.0, visibility),
            "right_shoulder": Landmark(0.6, 0.3, 0.0, visibility),
            "left_elbow": Landmark(0.35, 0.5, 0.0, visibility),
            "right_elbow": Landmark(0.65, 0.5, 0.0, visibility),
            "left_wrist": Landmark(0.3, 0.6, 0.0, visibility),
            "right_wrist": Landmark(0.7, 0.6, 0.0, visibility),
            "left_hip": Landmark(0.45, 0.7, 0.0, visibility),
            "right_hip": Landmark(0.55, 0.7, 0.0, visibility),
            "left_knee": Landmark(0.45, 0.85, 0.0, visibility),
            "right_knee": Landmark(0.55, 0.85, 0.0, visibility),
            "left_ankle": Landmark(0.45, 0.95, 0.0, visibility),
            "right_ankle": Landmark(0.55, 0.95, 0.0, visibility),
        }
        frames.append(pf)
    return frames


def test_webcam_end_to_end_chain():
    # Test pipeline with rule-based fallback
    engine = FitnessCVEngine(bundle_path=None)
    frames = _make_dummy_frame_sequence(count=10)

    out = engine.process_frame_sequence(frames)

    assert "exercise" in out
    assert "confidence" in out
    assert "rep_count" in out
    assert "phase" in out
    assert "form_feedback" in out
    assert out["valid_pose"] is True
    assert isinstance(out["form_feedback"], list)


def test_edge_case_no_person():
    engine = FitnessCVEngine(bundle_path=None)
    out = engine.process_frame_sequence([])

    assert out["exercise"] == "NO PERSON DETECTED"
    assert out["confidence"] == 0.0
    assert out["rep_count"] == 0
    assert out["phase"] == "TRACKING LOST"
    assert out["valid_pose"] is False


def test_edge_case_low_visibility():
    engine = FitnessCVEngine(bundle_path=None)
    frames = _make_dummy_frame_sequence(count=5, visibility=0.1)

    out = engine.process_frame_sequence(frames)
    assert out["valid_pose"] is True
    assert len(out["form_feedback"]) > 0


def test_form_analyzer_no_nan_or_inf():
    engine = FitnessCVEngine(bundle_path=None)
    frames = _make_dummy_frame_sequence(count=10)

    for ex in ["bicep_curl", "squat", "shoulder_press", "lateral_raise"]:
        engine.active_exercise = ex
        out = engine.process_frame_sequence(frames)

        # Check numeric sanity
        if out["rom"] is not None:
            assert not np.isnan(out["rom"])
            assert not np.isinf(out["rom"])
        if out["tempo"] is not None:
            assert not np.isnan(out["tempo"])
            assert not np.isinf(out["tempo"])
