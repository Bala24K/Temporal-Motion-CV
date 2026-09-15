"""
Kinematics (V3 section 3): turn a joint-angle time series into velocity
and acceleration — q(t), q'(t), q''(t) — the input inverse dynamics needs.

Uses numpy.gradient (second-order-accurate central differences, first-order
at the boundaries), which tolerates unevenly spaced timestamps — real pose
timestamps won't be perfectly uniform.
"""
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


@dataclass
class JointKinematics:
    times: List[float]
    angle_deg: List[float]
    angular_velocity_deg_s: List[float]
    angular_acceleration_deg_s2: List[float]


def compute_velocity(times: Sequence[float], angles_deg: Sequence[float]) -> List[float]:
    """First derivative of angle w.r.t. time, degrees/second."""
    t = np.asarray(times, dtype=float)
    a = np.asarray(angles_deg, dtype=float)
    if len(t) < 2:
        return [0.0] * len(t)
    return list(np.gradient(a, t))


def compute_acceleration(times: Sequence[float], angular_velocity_deg_s: Sequence[float]) -> List[float]:
    """Second derivative of angle w.r.t. time (first derivative of
    velocity), degrees/second^2."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(angular_velocity_deg_s, dtype=float)
    if len(t) < 2:
        return [0.0] * len(t)
    return list(np.gradient(v, t))


def compute_kinematics(times: Sequence[float], angles_deg: Sequence[float]) -> JointKinematics:
    velocity = compute_velocity(times, angles_deg)
    acceleration = compute_acceleration(times, velocity)
    return JointKinematics(
        times=list(times),
        angle_deg=list(angles_deg),
        angular_velocity_deg_s=velocity,
        angular_acceleration_deg_s2=acceleration,
    )
