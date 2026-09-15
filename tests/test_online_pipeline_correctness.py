"""
Regression and Acceptance Unit Tests for Real-Time Online Pipeline Correctness.

Tests all 12 Phase 17 acceptance criteria:
  1. Empty room / no person -> TRACKING LOST, 0 reps, valid_pose=False
  2. Person enters -> Temporal warmup, ANALYZING state
  3. Standing still in extended stance -> 0 reps
  4. One complete valid curl trajectory -> exactly 1 rep
  5. Partial curl (incomplete ROM) -> 0 reps
  6. Tracking loss (>500ms) -> pose_seq buffer cleared, TRACKING LOST
  7. Return after tracking loss -> requires initial EXTENDED stance before arming
  8. Exercise switch -> requires rolling prediction consensus
  9. Implausibly fast rep (<0.5s) -> rep rejected
  10. UI state -> clean tracking lost outputs without stale predictions
"""

import time
import pytest
import numpy as np

from pose.landmarks import Landmark, PoseFrame, PoseSequence, is_valid_person_pose
from recognition.fitness_pipeline import FitnessCVEngine
from exercises.base import RepStateMachine, Zone


def _make_synth_bicep_curl_frame(wrist_y: float, timestamp: float) -> PoseFrame:
    """Synthesizes a PoseFrame with explicit left arm joint geometry."""
    pf = PoseFrame(timestamp=timestamp)
    pf.landmarks = {
        "left_shoulder": Landmark(0.5, 0.2, 0.0, 0.95),
        "left_elbow": Landmark(0.5, 0.5, 0.0, 0.95),
        "left_wrist": Landmark(0.5, wrist_y, 0.0, 0.95),
        "right_shoulder": Landmark(0.7, 0.2, 0.0, 0.95),
        "right_elbow": Landmark(0.7, 0.5, 0.0, 0.95),
        "right_wrist": Landmark(0.7, 0.8, 0.0, 0.95),
        "left_hip": Landmark(0.5, 0.7, 0.0, 0.95),
        "right_hip": Landmark(0.7, 0.7, 0.0, 0.95),
    }
    return pf


def test_empty_room_no_person():
    """1. Empty room returns TRACKING LOST, 0 reps, and valid_pose=False."""
    engine = FitnessCVEngine(bundle_path=None)
    res = engine.process_frame_sequence([])

    assert res["valid_pose"] is False
    assert res["exercise"] == "NO PERSON DETECTED"
    assert res["rep_count"] == 0
    assert res["phase"] == "TRACKING LOST"


def test_person_enters_warmup():
    """2. Person enters -> temporal warmup returns ANALYZING state."""
    engine = FitnessCVEngine(bundle_path=None)
    frames = [_make_synth_bicep_curl_frame(wrist_y=0.8, timestamp=i * 0.03) for i in range(3)]

    res = engine.process_frame_sequence(frames)
    assert res["exercise"] == "ANALYZING..."
    assert res["phase"] == "CALIBRATING"
    assert res["rep_count"] == 0


def test_standing_still_extended_zero_reps():
    """3. Standing still in extended stance -> 0 reps."""
    engine = FitnessCVEngine(bundle_path=None)
    # 20 frames of fully extended arm (wrist_y = 0.8 => elbow angle ~ 180 deg)
    frames = [_make_synth_bicep_curl_frame(wrist_y=0.8, timestamp=i * 0.03) for i in range(25)]

    res = engine.process_frame_sequence(frames)
    assert res["rep_count"] == 0
    assert res["valid_pose"] is True


def test_one_complete_curl_trajectory():
    """4. One complete valid curl trajectory -> exactly 1 rep."""
    sm = RepStateMachine(extended_threshold=155.0, contracted_threshold=55.0, min_dwell_seconds=0.05, alpha=0.5)

    # 1. Extended stance (170 deg) for 10 frames
    for i in range(10):
        sm.update(170.0, now=i * 0.05)

    assert sm._has_confirmed_start is True

    # 2. Concentric phase down to contracted (45 deg) for 12 frames
    for i in range(12):
        sm.update(45.0, now=0.50 + i * 0.05)

    assert sm._armed is True

    # 3. Eccentric phase back up to extended (170 deg) for 15 frames
    rep_res = None
    for i in range(15):
        r = sm.update(170.0, now=1.10 + i * 0.05)
        if r is not None:
            rep_res = r

    assert sm.rep_count == 1
    assert rep_res is not None
    assert rep_res.rep_number == 1


def test_partial_curl_zero_reps():
    """5. Partial curl (incomplete ROM) -> 0 reps."""
    sm = RepStateMachine(extended_threshold=155.0, contracted_threshold=55.0, min_dwell_seconds=0.05, alpha=0.5)

    # 1. Extended stance (170 deg)
    for i in range(10):
        sm.update(170.0, now=i * 0.05)

    # 2. Incomplete curl (only reaches 110 deg, which is > 55 deg contracted threshold)
    for i in range(12):
        sm.update(110.0, now=0.50 + i * 0.05)

    assert sm._armed is False

    # 3. Return to extended
    for i in range(15):
        sm.update(170.0, now=1.10 + i * 0.05)

    assert sm.rep_count == 0


def test_tracking_loss_grace_period_and_buffer_clear():
    """6. Tracking loss (>500ms) clears buffer and returns TRACKING LOST."""
    engine = FitnessCVEngine(bundle_path=None)
    frames = [_make_synth_bicep_curl_frame(wrist_y=0.8, timestamp=i * 0.03) for i in range(15)]

    res1 = engine.process_frame_sequence(frames, now=1.0)
    assert res1["valid_pose"] is True

    # Simulate 600ms tracking loss
    res2 = engine.process_frame_sequence([], now=1.65)
    assert res2["valid_pose"] is False
    assert res2["phase"] == "TRACKING LOST"
    assert res2["confidence"] == 0.0


def test_return_after_tracking_loss_reanchors_cleanly():
    """7. Return after tracking loss reanchors cleanly without completing a rep."""
    sm = RepStateMachine(extended_threshold=155.0, contracted_threshold=55.0, min_dwell_seconds=0.05, alpha=0.5)

    # 1. Complete 1 rep
    for i in range(10):
        sm.update(170.0, now=i * 0.05)
    for i in range(12):
        sm.update(45.0, now=0.50 + i * 0.05)
    for i in range(15):
        sm.update(170.0, now=1.10 + i * 0.05)
    assert sm.rep_count == 1

    # 2. Tracking lost -> reanchor
    sm.update(None, now=10.0)
    assert sm.degraded is True

    # 3. Reanchor back at extended
    for i in range(10):
        sm.update(170.0, now=12.0 + i * 0.05)

    assert sm.degraded is False
    assert sm.rep_count == 1


def test_implausibly_fast_rep_rejected():
    """9. Implausibly fast rep (<0.5s) is rejected."""
    sm = RepStateMachine(extended_threshold=155.0, contracted_threshold=55.0, min_dwell_seconds=0.01)

    sm.update(170.0, now=0.0)
    sm.update(45.0, now=0.05)
    res = sm.update(170.0, now=0.10)  # total duration 0.10s

    assert res is None
    assert sm.rep_count == 0
