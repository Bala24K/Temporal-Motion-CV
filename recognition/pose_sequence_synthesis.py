"""
Synthetic FULL-BODY pose sequences for the structured/temporal pipeline
(body_model/ -> recognition/skeleton_features.py). Extends the same
geometric-definition approach as dataset_synthesis.py (see that module's
docstring for the synthetic-vs-real caveats, which apply equally here) —
but generates actual per-frame LANDMARK sequences via forward kinematics,
so the whole pipeline (pose landmarks -> SkeletonFrame -> SkeletonSequence
-> features -> model) gets exercised end to end, not shortcut through
pre-computed angles.

Orientation convention: every limb's absolute orientation is in degrees
from "straight up" in image coordinates (0 = up, 90 = right, 180 = down)
— see body_model/skeleton.py's Limb.orientation_deg. Joint angles are
DERIVED from these via forward kinematics, following the relationship
verified numerically before writing this module:
    child_orientation = joint_angle + parent_orientation - 180
(e.g. forearm_orientation = elbow_angle + upper_arm_orientation - 180,
 shin_orientation = knee_angle + thigh_orientation - 180)
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from pose.landmarks import PoseFrame, Landmark

EXERCISES = ["bicep_curl", "squat", "shoulder_press", "lateral_raise"]


def dir_from_vertical(deg: float) -> np.ndarray:
    r = np.radians(deg)
    return np.array([np.sin(r), -np.cos(r)])


def child_orientation_for_angle(joint_angle_deg: float, parent_orientation_deg: float) -> float:
    """See module docstring — the forward-kinematics relation connecting
    a joint angle to the child limb's absolute orientation, given the
    parent limb's orientation."""
    return joint_angle_deg + parent_orientation_deg - 180


@dataclass
class BodyProfile:
    hip_center: Tuple[float, float]
    shoulder_width: float
    hip_width: float
    upper_arm_len: float
    forearm_len: float
    thigh_len: float
    shin_len: float
    torso_len: float
    noise_std: float


def make_body_profile(rng: np.random.Generator) -> BodyProfile:
    return BodyProfile(
        hip_center=(0.5, 0.55),
        shoulder_width=rng.uniform(0.16, 0.22),
        hip_width=rng.uniform(0.12, 0.17),
        upper_arm_len=rng.uniform(0.16, 0.22),
        forearm_len=rng.uniform(0.16, 0.22),
        thigh_len=rng.uniform(0.20, 0.28),
        shin_len=rng.uniform(0.20, 0.28),
        torso_len=rng.uniform(0.22, 0.28),
        noise_std=rng.uniform(0.001, 0.006),
    )


def build_frame(profile: BodyProfile, t: float, rng: np.random.Generator, *,
                 torso_orient: float = 0.0,
                 upper_arm_orient_l: float, forearm_orient_l: float,
                 upper_arm_orient_r: float, forearm_orient_r: float,
                 thigh_orient_l: float, shin_orient_l: float,
                 thigh_orient_r: float, shin_orient_r: float) -> PoseFrame:
    hip_center = np.array(profile.hip_center)
    shoulder_center = hip_center + profile.torso_len * dir_from_vertical(torso_orient)

    left_shoulder = shoulder_center + np.array([-profile.shoulder_width / 2, 0.0])
    right_shoulder = shoulder_center + np.array([profile.shoulder_width / 2, 0.0])
    left_hip = hip_center + np.array([-profile.hip_width / 2, 0.0])
    right_hip = hip_center + np.array([profile.hip_width / 2, 0.0])

    left_elbow = left_shoulder + profile.upper_arm_len * dir_from_vertical(upper_arm_orient_l)
    left_wrist = left_elbow + profile.forearm_len * dir_from_vertical(forearm_orient_l)
    right_elbow = right_shoulder + profile.upper_arm_len * dir_from_vertical(upper_arm_orient_r)
    right_wrist = right_elbow + profile.forearm_len * dir_from_vertical(forearm_orient_r)

    left_knee = left_hip + profile.thigh_len * dir_from_vertical(thigh_orient_l)
    left_ankle = left_knee + profile.shin_len * dir_from_vertical(shin_orient_l)
    right_knee = right_hip + profile.thigh_len * dir_from_vertical(thigh_orient_r)
    right_ankle = right_knee + profile.shin_len * dir_from_vertical(shin_orient_r)

    coords = {
        "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
        "left_elbow": left_elbow, "right_elbow": right_elbow,
        "left_wrist": left_wrist, "right_wrist": right_wrist,
        "left_hip": left_hip, "right_hip": right_hip,
        "left_knee": left_knee, "right_knee": right_knee,
        "left_ankle": left_ankle, "right_ankle": right_ankle,
    }
    frame = PoseFrame(timestamp=t)
    for name, pos in coords.items():
        noisy = pos + rng.normal(0, profile.noise_std, size=2)
        frame.landmarks[name] = Landmark(float(noisy[0]), float(noisy[1]), visibility=0.9)
    return frame


def _smooth_phase(t: float, period: float, t0: float) -> float:
    """0->1->0 smooth sweep (half a cosine cycle per rep-half), avoiding
    the sharp velocity discontinuities a linear ramp would create at the
    turnaround points — those would show up as spurious acceleration
    spikes in the very feature (angular acceleration) we're trying to
    give the model real signal in."""
    return 0.5 * (1 - np.cos(2 * np.pi * (t - t0) / period))


REST_UPPER_ARM_ORIENT = 172.0   # ~8deg shoulder elevation at rest, arm relaxed at the side
REST_ELBOW_ANGLE = 160.0
REST_THIGH_SHIN_ORIENT = 180.0   # standing: thigh/shin point DOWN from hip/knee (180 = down, per dir_from_vertical)


def generate_window(rng: np.random.Generator, exercise: str, profile: BodyProfile,
                     n_frames: int = 16, dt: float = 0.06, period: float = 2.4) -> list:
    """
    Returns a list of n_frames PoseFrames spanning a short window sampled
    from a random point in a longer conceptual rep cycle — long enough to
    give body_model/sequence.py's velocity/acceleration a real signal to
    work with, short enough to resemble a live rolling buffer.
    """
    t0 = rng.uniform(0, period)
    frames = []
    for i in range(n_frames):
        t = i * dt
        phase = _smooth_phase(t, period, -t0)

        rest_forearm = child_orientation_for_angle(REST_ELBOW_ANGLE, REST_UPPER_ARM_ORIENT)

        if exercise == "bicep_curl":
            elbow_angle = REST_ELBOW_ANGLE + (45 - REST_ELBOW_ANGLE) * phase
            upper_l = REST_UPPER_ARM_ORIENT
            forearm_l = child_orientation_for_angle(elbow_angle, upper_l)
            upper_r, forearm_r = REST_UPPER_ARM_ORIENT, rest_forearm
            thigh_l = thigh_r = REST_THIGH_SHIN_ORIENT
            shin_l = shin_r = REST_THIGH_SHIN_ORIENT

        elif exercise == "squat":
            knee_angle = 170 + (90 - 170) * phase
            thigh = REST_THIGH_SHIN_ORIENT + (120 - REST_THIGH_SHIN_ORIENT) * phase  # thigh rotates forward as the person descends
            shin = child_orientation_for_angle(knee_angle, thigh)
            upper_l = upper_r = REST_UPPER_ARM_ORIENT
            forearm_l = forearm_r = rest_forearm
            thigh_l = thigh_r = thigh
            shin_l = shin_r = shin

        elif exercise == "shoulder_press":
            elbow_angle = 85 + (172 - 85) * phase
            # shoulder elevation ~75deg(rack) -> ~172deg(lockout); via the
            # 180-shoulder_elevation relation verified numerically:
            shoulder_elev = 75 + (172 - 75) * phase
            upper_l = upper_r = 180 - shoulder_elev
            forearm_l = child_orientation_for_angle(elbow_angle, upper_l)
            forearm_r = child_orientation_for_angle(elbow_angle, upper_r)
            thigh_l = thigh_r = REST_THIGH_SHIN_ORIENT
            shin_l = shin_r = REST_THIGH_SHIN_ORIENT

        elif exercise == "lateral_raise":
            shoulder_elev = 8 + (80 - 8) * phase
            upper_l = upper_r = 180 - shoulder_elev
            forearm_l = child_orientation_for_angle(REST_ELBOW_ANGLE, upper_l)
            forearm_r = child_orientation_for_angle(REST_ELBOW_ANGLE, upper_r)
            thigh_l = thigh_r = REST_THIGH_SHIN_ORIENT
            shin_l = shin_r = REST_THIGH_SHIN_ORIENT

        else:
            raise ValueError(f"unknown exercise: {exercise}")

        frame = build_frame(
            profile, t, rng,
            upper_arm_orient_l=upper_l, forearm_orient_l=forearm_l,
            upper_arm_orient_r=upper_r, forearm_orient_r=forearm_r,
            thigh_orient_l=thigh_l, shin_orient_l=shin_l,
            thigh_orient_r=thigh_r, shin_orient_r=shin_r,
        )
        frames.append(frame)
    return frames
