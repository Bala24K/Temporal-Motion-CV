"""
Temporal movement representation (a sequence of SkeletonFrames, not just
one). A single SkeletonFrame tells you the body's configuration; this
tells you how it's changing — which is the actual "movement," not the
static pose.

Reuses biomechanics.kinematics for the derivative math (already built and
validated against exact analytic derivatives in V3) rather than
reimplementing velocity/acceleration a third time.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from body_model.skeleton import SkeletonFrame, LIMB_NAMES, JOINT_NAMES, SYMMETRY_PAIRS
from biomechanics.kinematics import compute_velocity, compute_acceleration


class SkeletonSequence:
    """Rolling buffer of SkeletonFrames. push() a new frame each tick;
    the derived-feature methods below use whatever's currently buffered."""

    def __init__(self, max_len: int = 60):
        self.max_len = max_len
        self.frames: List[SkeletonFrame] = []

    def push(self, frame: SkeletonFrame):
        self.frames.append(frame)
        if len(self.frames) > self.max_len:
            self.frames.pop(0)

    def __len__(self):
        return len(self.frames)

    def _times(self) -> List[float]:
        return [f.timestamp for f in self.frames]

    def limb_orientation_series(self, limb_name: str) -> List[Optional[float]]:
        return [f.limb_orientation(limb_name) for f in self.frames]

    def joint_angle_series(self, joint_name: str) -> List[Optional[float]]:
        return [f.joint_angles.get(joint_name) for f in self.frames]

    def _kinematics_of(self, series: List[Optional[float]]):
        """Returns (current_value, velocity, acceleration) for the LAST
        frame in the buffer, or (None, None, None) if the series has any
        gaps (missing landmark at some point in the window) — rather than
        silently interpolating through a tracking dropout."""
        if not series or series[-1] is None or any(v is None for v in series):
            return None, None, None
        times = self._times()
        values = series
        if len(values) < 2:
            return values[-1], None, None
        vel = compute_velocity(times, values)
        acc = compute_acceleration(times, vel)
        return values[-1], vel[-1], acc[-1]

    def limb_kinematics(self, limb_name: str):
        """(orientation, angular_velocity, angular_acceleration) for the
        most recent frame, using the buffered window's history."""
        return self._kinematics_of(self.limb_orientation_series(limb_name))

    def joint_kinematics(self, joint_name: str):
        return self._kinematics_of(self.joint_angle_series(joint_name))

    def symmetry_features(self) -> Dict[str, Optional[float]]:
        """
        For each left/right limb pair, the difference in current
        orientation (degrees) and in current angular velocity — a cheap,
        per-frame symmetry signal, distinct from V2's DTW-based
        align_and_compare (which compares two whole trajectories after
        the fact; this is instantaneous, suitable as a live ML feature).
        None if either side's data is currently unavailable.
        """
        out: Dict[str, Optional[float]] = {}
        if not self.frames:
            for left, right in SYMMETRY_PAIRS:
                out[f"{left}_vs_{right}_orientation_diff"] = None
                out[f"{left}_vs_{right}_velocity_diff"] = None
            return out

        for left, right in SYMMETRY_PAIRS:
            l_val, l_vel, _ = self.limb_kinematics(left)
            r_val, r_vel, _ = self.limb_kinematics(right)
            out[f"{left}_vs_{right}_orientation_diff"] = (
                abs(l_val - r_val) if l_val is not None and r_val is not None else None
            )
            out[f"{left}_vs_{right}_velocity_diff"] = (
                abs(l_vel - r_vel) if l_vel is not None and r_vel is not None else None
            )
        return out
