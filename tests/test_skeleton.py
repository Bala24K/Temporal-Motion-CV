import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from body_model.skeleton import SkeletonFrame, Limb, LIMB_NAMES, JOINT_NAMES  # noqa: E402


def full_body_frame(vis=0.9):
    """A simple, fully-populated standing pose for structural tests."""
    frame = PoseFrame()
    coords = {
        "left_shoulder": (0.4, 0.3), "right_shoulder": (0.6, 0.3),
        "left_elbow": (0.38, 0.5), "right_elbow": (0.62, 0.5),
        "left_wrist": (0.36, 0.7), "right_wrist": (0.64, 0.7),
        "left_hip": (0.42, 0.6), "right_hip": (0.58, 0.6),
        "left_knee": (0.42, 0.8), "right_knee": (0.58, 0.8),
        "left_ankle": (0.42, 1.0), "right_ankle": (0.58, 1.0),
    }
    for name, (x, y) in coords.items():
        frame.landmarks[name] = Landmark(x, y, visibility=vis)
    return frame


def test_limb_length_matches_euclidean_distance():
    limb = Limb("test", (0.0, 0.0), (3.0, 4.0))
    assert abs(limb.length - 5.0) < 1e-9


def test_limb_midpoint():
    limb = Limb("test", (0.0, 0.0), (2.0, 4.0))
    assert limb.midpoint == (1.0, 2.0)


def test_limb_orientation_straight_up_is_zero():
    limb = Limb("test", (0.5, 0.5), (0.5, 0.3))  # end is ABOVE start (smaller y)
    assert abs(limb.orientation_deg - 0.0) < 1e-6


def test_limb_orientation_straight_down_is_180():
    limb = Limb("test", (0.5, 0.3), (0.5, 0.5))
    assert abs(abs(limb.orientation_deg) - 180.0) < 1e-6


def test_limb_orientation_horizontal_right_is_90():
    limb = Limb("test", (0.3, 0.5), (0.5, 0.5))
    assert abs(limb.orientation_deg - 90.0) < 1e-6


def test_all_limbs_present_for_a_fully_visible_body():
    sf = SkeletonFrame.from_pose_frame(full_body_frame())
    assert set(sf.limbs.keys()) == set(LIMB_NAMES)


def test_all_joint_angles_present_for_a_fully_visible_body():
    sf = SkeletonFrame.from_pose_frame(full_body_frame())
    assert set(sf.joint_angles.keys()) == set(JOINT_NAMES)


def test_joint_angle_matches_direct_angle_3pt_computation():
    """Cross-check against the existing, independently-tested angle_3pt —
    the structured representation shouldn't compute a different number
    for the same three points."""
    from biomechanics.angles import angle_3pt
    frame = full_body_frame()
    sf = SkeletonFrame.from_pose_frame(frame)
    expected = angle_3pt(
        frame.get("left_shoulder").xy(), frame.get("left_elbow").xy(), frame.get("left_wrist").xy()
    )
    assert abs(sf.joint_angles["left_elbow"] - expected) < 1e-9


def test_torso_limb_uses_synthetic_midpoints():
    frame = full_body_frame()
    sf = SkeletonFrame.from_pose_frame(frame)
    torso = sf.limbs["torso"]
    expected_hip_center = ((0.42 + 0.58) / 2, (0.6 + 0.6) / 2)
    expected_shoulder_center = ((0.4 + 0.6) / 2, (0.3 + 0.3) / 2)
    assert np.allclose(torso.start, expected_hip_center)
    assert np.allclose(torso.end, expected_shoulder_center)


def test_missing_landmarks_omit_limbs_and_joints_rather_than_fabricating():
    frame = PoseFrame()
    frame.landmarks["left_shoulder"] = Landmark(0.4, 0.3, visibility=0.9)
    frame.landmarks["left_elbow"] = Landmark(0.4, 0.5, visibility=0.9)
    # left_wrist missing entirely
    sf = SkeletonFrame.from_pose_frame(frame)
    assert "upper_arm_left" in sf.limbs      # shoulder+elbow present
    assert "forearm_left" not in sf.limbs    # wrist missing
    assert "left_elbow" not in sf.joint_angles  # needs all three points


def test_low_visibility_landmark_excluded_like_missing():
    frame = full_body_frame()
    frame.landmarks["left_wrist"] = Landmark(0.36, 0.7, visibility=0.05)  # below threshold
    sf = SkeletonFrame.from_pose_frame(frame)
    assert "forearm_left" not in sf.limbs


def test_limb_orientation_and_length_accessors():
    sf = SkeletonFrame.from_pose_frame(full_body_frame())
    assert sf.limb_length("thigh_left") is not None
    assert sf.limb_orientation("thigh_left") is not None
    assert sf.limb_length("nonexistent_limb") is None
    assert sf.limb_orientation("nonexistent_limb") is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
