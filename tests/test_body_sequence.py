import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from body_model.skeleton import SkeletonFrame  # noqa: E402
from body_model.sequence import SkeletonSequence  # noqa: E402


def frame_with_elbow_angle(angle_deg, t, vis=0.9):
    """Builds a full standing body but with the left forearm rotated to
    hit a specific left_elbow joint angle, using the SAME forward-
    kinematics convention validated numerically before building this
    module (elbow_angle = forearm_orientation - upper_arm_orientation + 180)."""
    frame = PoseFrame(timestamp=t)
    shoulder = (0.4, 0.3)
    upper_arm_orient = 175.0

    def dir_from_vertical(deg):
        r = np.radians(deg)
        return np.array([np.sin(r), -np.cos(r)])

    elbow = np.array(shoulder) + 0.2 * dir_from_vertical(upper_arm_orient)
    forearm_orient = angle_deg - (180 - upper_arm_orient)
    wrist = elbow + 0.2 * dir_from_vertical(forearm_orient)

    coords = {
        "left_shoulder": shoulder, "right_shoulder": (0.6, 0.3),
        "left_elbow": tuple(elbow), "right_elbow": (0.62, 0.5),
        "left_wrist": tuple(wrist), "right_wrist": (0.64, 0.7),
        "left_hip": (0.42, 0.6), "right_hip": (0.58, 0.6),
        "left_knee": (0.42, 0.8), "right_knee": (0.58, 0.8),
        "left_ankle": (0.42, 1.0), "right_ankle": (0.58, 1.0),
    }
    for name, (x, y) in coords.items():
        frame.landmarks[name] = Landmark(x, y, visibility=vis)
    return frame


def test_forward_kinematics_helper_hits_target_angle():
    frame = frame_with_elbow_angle(120.0, t=0.0)
    sf = SkeletonFrame.from_pose_frame(frame)
    assert abs(sf.joint_angles["left_elbow"] - 120.0) < 1e-6


def test_empty_sequence_kinematics_are_none():
    seq = SkeletonSequence()
    val, vel, acc = seq.joint_kinematics("left_elbow")
    assert val is None and vel is None and acc is None


def test_single_frame_gives_value_but_no_derivatives():
    seq = SkeletonSequence()
    seq.push(SkeletonFrame.from_pose_frame(frame_with_elbow_angle(160.0, t=0.0)))
    val, vel, acc = seq.joint_kinematics("left_elbow")
    assert abs(val - 160.0) < 1e-6
    assert vel is None
    assert acc is None


def test_velocity_matches_known_linear_change():
    seq = SkeletonSequence()
    dt = 0.1
    angles = [160.0, 150.0, 140.0, 130.0]  # constant -100 deg/s
    for i, a in enumerate(angles):
        seq.push(SkeletonFrame.from_pose_frame(frame_with_elbow_angle(a, t=i * dt)))
    val, vel, acc = seq.joint_kinematics("left_elbow")
    assert abs(val - 130.0) < 1e-6
    assert abs(vel - (-100.0)) < 1e-3
    assert abs(acc - 0.0) < 1e-3  # constant velocity -> ~zero acceleration


def test_gap_in_window_returns_none_rather_than_interpolating():
    """If a landmark drops out mid-window (angle unavailable for one
    frame), kinematics should report None rather than silently skipping
    the gap and computing a derivative across it."""
    seq = SkeletonSequence()
    dt = 0.1
    seq.push(SkeletonFrame.from_pose_frame(frame_with_elbow_angle(160.0, t=0.0)))
    seq.push(SkeletonFrame.from_pose_frame(frame_with_elbow_angle(150.0, t=dt)))
    # third frame: wrist missing -> no left_elbow joint angle this frame
    gap_frame = PoseFrame(timestamp=2 * dt)
    gap_frame.landmarks["left_shoulder"] = Landmark(0.4, 0.3, visibility=0.9)
    gap_frame.landmarks["left_elbow"] = Landmark(0.38, 0.5, visibility=0.9)
    seq.push(SkeletonFrame.from_pose_frame(gap_frame))

    val, vel, acc = seq.joint_kinematics("left_elbow")
    assert val is None and vel is None and acc is None


def test_symmetry_features_zero_for_a_mirrored_standing_pose():
    seq = SkeletonSequence()
    frame = frame_with_elbow_angle(160.0, t=0.0)  # right side is NOT mirrored to match left in this helper
    seq.push(SkeletonFrame.from_pose_frame(frame))
    features = seq.symmetry_features()
    assert "upper_arm_left_vs_upper_arm_right_orientation_diff" in features
    # thigh/shin ARE symmetric in this helper's fixed standing legs
    assert features["thigh_left_vs_thigh_right_orientation_diff"] is not None
    assert features["thigh_left_vs_thigh_right_orientation_diff"] < 1.0


def test_symmetry_features_detect_asymmetry():
    seq = SkeletonSequence()
    frame = frame_with_elbow_angle(40.0, t=0.0)  # left arm curled, right arm untouched (still hanging)
    seq.push(SkeletonFrame.from_pose_frame(frame))
    features = seq.symmetry_features()
    # left forearm is bent way up, right forearm still hangs down -> large orientation diff
    assert features["forearm_left_vs_forearm_right_orientation_diff"] > 30


def test_symmetry_features_on_empty_sequence_are_none():
    seq = SkeletonSequence()
    features = seq.symmetry_features()
    assert all(v is None for v in features.values())


def test_max_len_evicts_oldest_frames():
    seq = SkeletonSequence(max_len=3)
    for i in range(5):
        seq.push(SkeletonFrame.from_pose_frame(frame_with_elbow_angle(160.0 - i, t=i * 0.1)))
    assert len(seq) == 3
    assert seq.frames[0].timestamp == 0.2  # frames at t=0.0, 0.1 evicted


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
