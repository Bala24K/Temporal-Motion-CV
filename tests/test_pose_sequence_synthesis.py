import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from recognition.pose_sequence_synthesis import (  # noqa: E402
    make_body_profile, generate_window, EXERCISES, dir_from_vertical, child_orientation_for_angle,
)
from body_model.skeleton import SkeletonFrame  # noqa: E402


def test_dir_from_vertical_matches_known_directions():
    assert np.allclose(dir_from_vertical(0), [0, -1])     # up
    assert np.allclose(dir_from_vertical(90), [1, 0])      # right
    assert np.allclose(dir_from_vertical(180), [0, 1])     # down
    assert np.allclose(dir_from_vertical(-90), [-1, 0])    # left


def test_child_orientation_relation_round_trips():
    # if child_orientation = angle + parent - 180, then recovering angle
    # from (child, parent) should give back the original angle.
    angle = 123.4
    parent = 37.0
    child = child_orientation_for_angle(angle, parent)
    recovered_angle = child - parent + 180
    assert abs(recovered_angle - angle) < 1e-9


def test_standing_pose_has_correct_vertical_ordering():
    """Sanity check that the generator produces topologically correct
    bodies: hip above knee above ankle on screen (y increases downward) —
    this is exactly the bug that was caught and fixed during development
    (legs were initially generated pointing 'up' from the hip instead of
    'down', an inverted-orientation-convention bug)."""
    rng = np.random.default_rng(0)
    profile = make_body_profile(rng)
    for exercise in EXERCISES:
        frames = generate_window(rng, exercise, profile, n_frames=5)
        for f in frames:
            hip_y = f.get("left_hip").y
            knee_y = f.get("left_knee").y
            ankle_y = f.get("left_ankle").y
            assert hip_y < knee_y < ankle_y, f"{exercise}: bad vertical ordering at some frame"


def test_squat_shows_large_knee_and_hip_angle_excursion():
    rng = np.random.default_rng(1)
    profile = make_body_profile(rng)
    frames = generate_window(rng, "squat", profile, n_frames=20, dt=0.06, period=1.2)
    knee_angles = [SkeletonFrame.from_pose_frame(f).joint_angles["left_knee"] for f in frames]
    hip_angles = [SkeletonFrame.from_pose_frame(f).joint_angles["left_hip"] for f in frames]
    assert max(knee_angles) - min(knee_angles) > 30
    assert max(hip_angles) - min(hip_angles) > 20


def test_bicep_curl_keeps_legs_near_standing_while_elbow_sweeps():
    rng = np.random.default_rng(2)
    profile = make_body_profile(rng)
    frames = generate_window(rng, "bicep_curl", profile, n_frames=20, dt=0.06, period=1.2)
    knee_angles = [SkeletonFrame.from_pose_frame(f).joint_angles["left_knee"] for f in frames]
    elbow_angles = [SkeletonFrame.from_pose_frame(f).joint_angles["left_elbow"] for f in frames]
    assert max(knee_angles) - min(knee_angles) < 10   # legs barely move
    assert max(elbow_angles) - min(elbow_angles) > 30  # elbow does


def test_all_exercises_produce_complete_skeletons():
    rng = np.random.default_rng(3)
    profile = make_body_profile(rng)
    for exercise in EXERCISES:
        frames = generate_window(rng, exercise, profile, n_frames=5)
        for f in frames:
            sf = SkeletonFrame.from_pose_frame(f)
            assert len(sf.limbs) == 9   # all limbs present, fully visible synthetic body
            assert len(sf.joint_angles) == 8


def test_unknown_exercise_raises():
    rng = np.random.default_rng(0)
    profile = make_body_profile(rng)
    try:
        generate_window(rng, "not_real", profile, n_frames=5)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
