"""
Unit tests for modular temporal feature engineering.
"""

import numpy as np
import pytest

from pose.landmarks import Landmark, PoseFrame
from recognition.feature_engineering import (
    FeatureConfig,
    extract_raw_landmarks_matrix,
    extract_joint_angles_matrix,
    compute_temporal_derivatives,
    compute_symmetry_matrix,
    compute_range_of_motion_vector,
    extract_modular_sequence_features,
    extract_modular_tabular_features,
    LANDMARK_ORDER,
)


def _make_dummy_pose_frame(t: int) -> PoseFrame:
    """Creates a synthetic PoseFrame with moving joints over time t."""
    pf = PoseFrame()
    # Add dummy landmarks
    for i, name in enumerate(LANDMARK_ORDER):
        pf.landmarks[name] = Landmark(
            x=0.5 + 0.01 * t * (i + 1),
            y=0.4 + 0.02 * t,
            z=0.1,
            visibility=0.9
        )
    return pf


def test_feature_config_group_name():
    cfg1 = FeatureConfig(use_raw_landmarks=True, use_joint_angles=False, use_velocity=False,
                         use_acceleration=False, use_symmetry=False, use_range_of_motion=False)
    assert cfg1.get_group_name() == "landmarks"

    cfg2 = FeatureConfig(use_raw_landmarks=True, use_joint_angles=True, use_velocity=True,
                         use_acceleration=True, use_symmetry=True, use_range_of_motion=True)
    assert cfg2.get_group_name() == "landmarks+angles+vel+acc+sym+rom"


def test_extract_raw_landmarks_matrix():
    frames = [_make_dummy_pose_frame(t) for t in range(5)]
    mat = extract_raw_landmarks_matrix(frames)
    assert mat.shape == (5, len(LANDMARK_ORDER) * 3)
    assert not np.isnan(mat).any()


def test_extract_joint_angles_matrix():
    frames = [_make_dummy_pose_frame(t) for t in range(5)]
    mat = extract_joint_angles_matrix(frames)
    assert mat.shape == (5, 8)
    assert not np.isnan(mat).any()


test_extract_joint_angles_matrix()


def test_compute_temporal_derivatives():
    data = np.array([[1.0, 2.0], [2.0, 5.0], [4.0, 9.0]], dtype=np.float32)
    vel, acc = compute_temporal_derivatives(data)
    assert vel.shape == data.shape
    assert acc.shape == data.shape
    # vel[1] = [1.0, 3.0], vel[2] = [2.0, 4.0]
    assert np.isclose(vel[1, 0], 1.0)
    assert np.isclose(vel[1, 1], 3.0)


def test_extract_modular_sequence_features_ablation():
    frames = [_make_dummy_pose_frame(t) for t in range(10)]

    # Level 1: Raw landmarks only
    cfg1 = FeatureConfig(use_raw_landmarks=True, use_joint_angles=False, use_velocity=False,
                         use_acceleration=False, use_symmetry=False, use_range_of_motion=False)
    seq1 = extract_modular_sequence_features(frames, cfg1)
    assert seq1.shape[0] == 10
    assert seq1.shape[1] == 39

    # Level 2: + angles
    cfg2 = FeatureConfig(use_raw_landmarks=True, use_joint_angles=True, use_velocity=False,
                         use_acceleration=False, use_symmetry=False, use_range_of_motion=False)
    seq2 = extract_modular_sequence_features(frames, cfg2)
    assert seq2.shape[1] == 39 + 8

    # Level 3: + velocity
    cfg3 = FeatureConfig(use_raw_landmarks=True, use_joint_angles=True, use_velocity=True,
                         use_acceleration=False, use_symmetry=False, use_range_of_motion=False)
    seq3 = extract_modular_sequence_features(frames, cfg3)
    assert seq3.shape[1] == (39 + 8) * 2  # base + velocity

    # Tabular features check
    tab = extract_modular_tabular_features(frames, cfg3)
    assert len(tab) > 0
    assert not np.isnan(tab).any()
