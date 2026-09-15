import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from recognition.features import extract_features, fill_missing, FEATURE_NAMES, NEUTRAL_FILL_VALUE  # noqa: E402


def test_feature_vector_has_expected_length_and_order():
    frame = PoseFrame()
    features = extract_features(frame)
    assert len(features) == len(FEATURE_NAMES) == 8


def test_known_right_angle_elbow_extracted_correctly():
    frame = PoseFrame()
    frame.landmarks["left_shoulder"] = Landmark(0, 0, visibility=0.9)
    frame.landmarks["left_elbow"] = Landmark(0, -1, visibility=0.9)
    frame.landmarks["left_wrist"] = Landmark(1, -1, visibility=0.9)
    features = extract_features(frame)
    idx = FEATURE_NAMES.index("left_elbow_angle")
    assert abs(features[idx] - 90.0) < 1e-6


def test_missing_joint_gives_nan():
    frame = PoseFrame()  # nothing populated
    features = extract_features(frame)
    assert np.all(np.isnan(features))


def test_low_visibility_joint_gives_nan():
    frame = PoseFrame()
    frame.landmarks["left_shoulder"] = Landmark(0, 0, visibility=0.1)  # below MIN_VISIBILITY
    frame.landmarks["left_elbow"] = Landmark(0, -1, visibility=0.9)
    frame.landmarks["left_wrist"] = Landmark(1, -1, visibility=0.9)
    features = extract_features(frame)
    idx = FEATURE_NAMES.index("left_elbow_angle")
    assert np.isnan(features[idx])


def test_fill_missing_replaces_nan_with_neutral_value():
    features = np.array([90.0, np.nan, 170.0, np.nan, 0, 0, 0, 0])
    filled = fill_missing(features)
    assert not np.any(np.isnan(filled))
    assert filled[1] == NEUTRAL_FILL_VALUE
    assert filled[3] == NEUTRAL_FILL_VALUE
    assert filled[0] == 90.0  # non-NaN values untouched


def test_fill_missing_does_not_mutate_input():
    features = np.array([np.nan, 1.0])
    fill_missing(features)
    assert np.isnan(features[0])  # original untouched


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
