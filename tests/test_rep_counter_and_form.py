"""
Unit tests for RepCounter, FormAnalyzer, and unified FitnessCVEngine.
Uses synthetic joint landmark trajectories to test rep state transitions and form warnings.
"""

import pytest

from pose.landmarks import Landmark, PoseFrame
from recognition.fitness_pipeline import FitnessCVEngine
from exercises.bicep_curl import BicepCurlAnalyzer


def _make_pose_frame_with_wrist(wx: float, wy: float, time_offset: float = 0.0) -> PoseFrame:
    """Synthesizes a PoseFrame with explicit left wrist position."""
    pf = PoseFrame(timestamp=time_offset)
    pf.landmarks = {
        "left_shoulder": Landmark(0.5, 0.3, 0.0, 0.9),
        "left_elbow": Landmark(0.5, 0.5, 0.0, 0.9),
        "left_wrist": Landmark(wx, wy, 0.0, 0.9),
        "right_shoulder": Landmark(0.7, 0.3, 0.0, 0.9),
        "right_elbow": Landmark(0.7, 0.5, 0.0, 0.9),
        "right_wrist": Landmark(0.7, 0.7, 0.0, 0.9),
        "left_hip": Landmark(0.5, 0.7, 0.0, 0.9),
        "right_hip": Landmark(0.7, 0.7, 0.0, 0.9),
        "left_knee": Landmark(0.5, 0.85, 0.0, 0.9),
        "right_knee": Landmark(0.7, 0.85, 0.0, 0.9),
        "left_ankle": Landmark(0.5, 0.95, 0.0, 0.9),
        "right_ankle": Landmark(0.7, 0.95, 0.0, 0.9),
    }
    return pf


def test_bicep_curl_rep_counting_trajectory():
    analyzer = BicepCurlAnalyzer(side="left")
    now = 0.0

    # 1. Arm Extended (wrist at y=0.7, angle ~ 180 deg) for dwell time (> 0.08s)
    for _ in range(6):
        now += 0.05
        pf = _make_pose_frame_with_wrist(0.5, 0.7, time_offset=now)
        analyzer.process(pf, now=now)

    assert analyzer.sm.rep_count == 0

    # 2. Curl upwards to contracted position (wrist at y=0.32, angle ~ 30 deg)
    for wy in [0.6, 0.5, 0.4, 0.32, 0.32, 0.32, 0.32]:
        now += 0.05
        pf = _make_pose_frame_with_wrist(0.5, wy, time_offset=now)
        analyzer.process(pf, now=now)

    assert analyzer.sm._armed is True

    # 3. Lower back down to extended position (wrist at y=0.7, angle ~ 180 deg)
    for wy in [0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7]:
        now += 0.05
        pf = _make_pose_frame_with_wrist(0.5, wy, time_offset=now)
        analyzer.process(pf, now=now)

    assert analyzer.sm.rep_count == 1


def test_fitness_cv_engine_pipeline():
    engine = FitnessCVEngine(bundle_path=None)
    engine.active_exercise = "bicep_curl"

    frames = [_make_pose_frame_with_wrist(0.5, 0.7, time_offset=t*0.05) for t in range(5)]
    out = engine.process_frame_sequence(frames)

    assert out["exercise"] == "Bicep Curl"
    assert out["rep_count"] == 0
    assert "form_feedback" in out
    assert out["valid_pose"] is True
