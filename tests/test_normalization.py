import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from pose.normalization import normalize_frame  # noqa: E402


def make_body(scale=1.0, offset=(0.0, 0.0), rotation_deg=0.0, elbow_bend_deg=90.0):
    """A minimal synthetic 'body' — shoulders, hips, one arm — built so we
    can independently control body size, position, camera-relative tilt,
    and arm pose, and check that normalization removes the first three
    while preserving the fourth."""
    # base positions (un-rotated, un-scaled, un-translated), roughly life-sized
    base = {
        "left_shoulder": np.array([-0.15, 0.0]),
        "right_shoulder": np.array([0.15, 0.0]),
        "left_hip": np.array([-0.12, 0.5]),
        "right_hip": np.array([0.12, 0.5]),
        "left_elbow": np.array([-0.15, 0.3]),
    }
    # wrist position controlled by elbow_bend_deg relative to upper arm direction
    shoulder = base["left_shoulder"]
    elbow = base["left_elbow"]
    upper_arm = shoulder - elbow
    upper_arm_unit = upper_arm / np.linalg.norm(upper_arm)
    theta = np.radians(elbow_bend_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    forearm_dir = rot @ upper_arm_unit
    base["left_wrist"] = elbow + forearm_dir * 0.25

    theta_r = np.radians(rotation_deg)
    R = np.array([[np.cos(theta_r), -np.sin(theta_r)], [np.sin(theta_r), np.cos(theta_r)]])
    off = np.array(offset)

    frame = PoseFrame()
    for name, p in base.items():
        p2 = R @ (p * scale) + off
        frame.landmarks[name] = Landmark(x=float(p2[0]), y=float(p2[1]), z=0.0, visibility=0.9)
    return frame


def test_missing_required_landmarks_returns_none():
    frame = PoseFrame()
    frame.landmarks["left_shoulder"] = Landmark(0, 0, visibility=0.9)
    # right_shoulder / hips missing
    assert normalize_frame(frame) is None


def test_scale_invariance():
    """Same pose, different body size -> same normalized coordinates."""
    small = make_body(scale=0.7)
    large = make_body(scale=1.4)
    n_small = normalize_frame(small)
    n_large = normalize_frame(large)
    assert n_small is not None and n_large is not None

    for name in small.landmarks:
        a = n_small.get(name)
        b = n_large.get(name)
        assert abs(a.x - b.x) < 1e-6, f"{name} x differs: {a.x} vs {b.x}"
        assert abs(a.y - b.y) < 1e-6, f"{name} y differs: {a.y} vs {b.y}"


def test_translation_invariance():
    """Same pose, standing in a different part of the frame -> same
    normalized coordinates."""
    here = make_body(offset=(0.0, 0.0))
    there = make_body(offset=(0.3, -0.2))
    n_here = normalize_frame(here)
    n_there = normalize_frame(there)
    for name in here.landmarks:
        a = n_here.get(name)
        b = n_there.get(name)
        assert abs(a.x - b.x) < 1e-6, f"{name} x differs after translation"
        assert abs(a.y - b.y) < 1e-6, f"{name} y differs after translation"


def test_rotation_invariance():
    """Same pose, person standing slightly tilted relative to the camera
    -> same normalized coordinates (in-plane rotation removed)."""
    straight = make_body(rotation_deg=0.0)
    tilted = make_body(rotation_deg=12.0)
    n_straight = normalize_frame(straight)
    n_tilted = normalize_frame(tilted)
    for name in straight.landmarks:
        a = n_straight.get(name)
        b = n_tilted.get(name)
        assert abs(a.x - b.x) < 1e-4, f"{name} x differs after rotation: {a.x} vs {b.x}"
        assert abs(a.y - b.y) < 1e-4, f"{name} y differs after rotation: {a.y} vs {b.y}"


def test_actual_arm_movement_is_preserved():
    """Normalization must remove nuisance variation WITHOUT flattening out
    the actual signal we care about — different elbow bends should still
    look different after normalization."""
    extended = normalize_frame(make_body(elbow_bend_deg=170.0))
    curled = normalize_frame(make_body(elbow_bend_deg=60.0))
    wrist_extended = np.array(extended.get("left_wrist").xy())
    wrist_curled = np.array(curled.get("left_wrist").xy())
    assert np.linalg.norm(wrist_extended - wrist_curled) > 0.1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
