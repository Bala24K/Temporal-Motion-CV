import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from body_model.skeleton import SkeletonFrame, LIMB_NAMES, JOINT_NAMES, SYMMETRY_PAIRS  # noqa: E402
from body_model.sequence import SkeletonSequence  # noqa: E402
from recognition.skeleton_features import (  # noqa: E402
    extract_features, fill_missing, FEATURE_NAMES, FEATURE_KINDS, VALUE_FILL, DERIVATIVE_FILL,
)
from recognition.pose_sequence_synthesis import make_body_profile, generate_window  # noqa: E402


def expected_feature_count():
    return len(LIMB_NAMES) * 3 + len(JOINT_NAMES) * 3 + len(SYMMETRY_PAIRS) * 2


def test_feature_names_and_kinds_have_matching_expected_length():
    assert len(FEATURE_NAMES) == len(FEATURE_KINDS) == expected_feature_count()


def test_empty_sequence_gives_all_nan():
    seq = SkeletonSequence()
    features = extract_features(seq)
    assert len(features) == len(FEATURE_NAMES)
    assert np.all(np.isnan(features))


def test_single_frame_gives_values_but_no_derivatives():
    rng = np.random.default_rng(0)
    profile = make_body_profile(rng)
    frames = generate_window(rng, "bicep_curl", profile, n_frames=1)
    seq = SkeletonSequence()
    seq.push(SkeletonFrame.from_pose_frame(frames[0]))
    features = extract_features(seq)

    idx = FEATURE_NAMES.index("left_elbow_angle")
    assert not np.isnan(features[idx])  # value present
    vel_idx = FEATURE_NAMES.index("left_elbow_ang_velocity")
    assert np.isnan(features[vel_idx])  # no derivative yet, single frame


def test_multi_frame_window_gives_full_kinematics():
    rng = np.random.default_rng(1)
    profile = make_body_profile(rng)
    frames = generate_window(rng, "squat", profile, n_frames=10)
    seq = SkeletonSequence()
    for f in frames:
        seq.push(SkeletonFrame.from_pose_frame(f))
    features = extract_features(seq)
    idx = FEATURE_NAMES.index("left_knee_ang_velocity")
    assert not np.isnan(features[idx])


def test_fill_missing_uses_value_fill_for_angles_and_zero_for_derivatives():
    features = np.full(len(FEATURE_NAMES), np.nan)
    filled = fill_missing(features)
    assert not np.any(np.isnan(filled))
    value_idx = FEATURE_NAMES.index("left_elbow_angle")
    deriv_idx = FEATURE_NAMES.index("left_elbow_ang_velocity")
    assert filled[value_idx] == VALUE_FILL
    assert filled[deriv_idx] == DERIVATIVE_FILL


def test_fill_missing_does_not_touch_present_values():
    features = np.zeros(len(FEATURE_NAMES))
    features[0] = 42.0
    filled = fill_missing(features)
    assert filled[0] == 42.0


def test_end_to_end_all_exercises_produce_valid_filled_features():
    rng = np.random.default_rng(2)
    profile = make_body_profile(rng)
    for exercise in ["bicep_curl", "squat", "shoulder_press", "lateral_raise"]:
        frames = generate_window(rng, exercise, profile, n_frames=16)
        seq = SkeletonSequence()
        for f in frames:
            seq.push(SkeletonFrame.from_pose_frame(f))
        features = fill_missing(extract_features(seq))
        assert features.shape == (len(FEATURE_NAMES),)
        assert not np.any(np.isnan(features))


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
