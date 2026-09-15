"""
Structured geometric body representation (replaces "a handful of
hand-picked angles" with a systematic representation of the whole
skeleton — joints, limbs, and the relationships between them).

Where recognition/features.py picked 8 specific angles by hand per
exercise, this module builds the representation FROM the skeleton's own
connectivity graph: every limb gets a position, length, and orientation;
every pair of limbs that share a joint gets an angle between them —
computed once, generically, not re-derived per exercise. Downstream code
(recognition/skeleton_features.py, body_model/sequence.py) reads off of
this structure rather than calling angle_3pt() on hand-picked triplets.

This does NOT use 3D pose, OpenSim, or any of the V3 biomechanics
machinery — it's a purely 2D geometric skeleton graph, one level up from
raw landmarks and one level below "which exercise is this."
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from pose.landmarks import PoseFrame

# Which two landmarks each limb spans. This IS the skeleton topology —
# everything else (joint angles, symmetry pairs) is derived from it below,
# not hand-maintained separately.
LIMB_DEFINITIONS: Dict[str, Tuple[str, str]] = {
    "torso": ("hip_center", "shoulder_center"),  # synthetic midpoint landmarks, see SkeletonFrame.from_pose_frame
    "upper_arm_left": ("left_shoulder", "left_elbow"),
    "forearm_left": ("left_elbow", "left_wrist"),
    "upper_arm_right": ("right_shoulder", "right_elbow"),
    "forearm_right": ("right_elbow", "right_wrist"),
    "thigh_left": ("left_hip", "left_knee"),
    "shin_left": ("left_knee", "left_ankle"),
    "thigh_right": ("right_hip", "right_knee"),
    "shin_right": ("right_knee", "right_ankle"),
}

# Joint angle = angle_3pt(A, shared_joint, C) between two limbs that share
# an endpoint. Derived by construction: any two limbs in LIMB_DEFINITIONS
# sharing a landmark form a joint. Listed explicitly here (rather than
# auto-detected) only for readability and stable ordering/naming.
JOINT_ANGLE_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder": ("hip_center", "left_shoulder", "left_elbow"),  # torso-to-upper-arm
    "right_shoulder": ("hip_center", "right_shoulder", "right_elbow"),
    "left_hip": ("shoulder_center", "left_hip", "left_knee"),  # torso-to-thigh
    "right_hip": ("shoulder_center", "right_hip", "right_knee"),
}

# Left/right counterpart limbs, for symmetry features (body_model/sequence.py).
SYMMETRY_PAIRS: List[Tuple[str, str]] = [
    ("upper_arm_left", "upper_arm_right"),
    ("forearm_left", "forearm_right"),
    ("thigh_left", "thigh_right"),
    ("shin_left", "shin_right"),
]

LIMB_NAMES: List[str] = list(LIMB_DEFINITIONS.keys())
JOINT_NAMES: List[str] = list(JOINT_ANGLE_DEFINITIONS.keys())


@dataclass
class Limb:
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.array(self.end) - np.array(self.start)))

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)

    @property
    def orientation_deg(self) -> float:
        """
        Absolute orientation, in degrees measured from "straight up" in
        image coordinates (0 = pointing up the screen, 90 = pointing
        right, 180 = pointing down), independent of any parent limb. This
        is the SAME convention used elsewhere in the codebase for
        vertical-reference angles (e.g. squat.py's torso-lean check).
        """
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        # atan2 with axes arranged so 0deg = straight up (dx=0, dy<0)
        return float(np.degrees(np.arctan2(dx, -dy)))


def _midpoint_landmark(frame: PoseFrame, a: str, b: str) -> Optional[Tuple[float, float]]:
    la, lb = frame.get(a), frame.get(b)
    if la is None or lb is None:
        return None
    return ((la.x + lb.x) / 2, (la.y + lb.y) / 2)


class SkeletonFrame:
    """
    The structured body representation for a single frame: every limb in
    LIMB_DEFINITIONS (with position/length/orientation) and every joint
    angle in JOINT_ANGLE_DEFINITIONS, built from a PoseFrame's landmarks.

    Missing landmarks simply mean the corresponding limb/joint is absent
    from .limbs / .joint_angles (not filled with a fabricated value) —
    callers (skeleton_features.py) decide how to handle that.
    """

    def __init__(self, timestamp: float = 0.0):
        self.timestamp = timestamp
        self.limbs: Dict[str, Limb] = {}
        self.joint_angles: Dict[str, float] = {}

    @classmethod
    def from_pose_frame(cls, frame: PoseFrame, min_visibility: float = 0.5) -> "SkeletonFrame":
        sf = cls(timestamp=frame.timestamp)

        # Synthetic midline points, needed for the torso limb and for the
        # shoulder/hip joint-angle definitions above.
        points: Dict[str, Tuple[float, float]] = {}
        for name, lm in frame.landmarks.items():
            if lm.visibility >= min_visibility:
                points[name] = lm.xy()
        hip_center = _midpoint_landmark(frame, "left_hip", "right_hip")
        shoulder_center = _midpoint_landmark(frame, "left_shoulder", "right_shoulder")
        if hip_center is not None:
            points["hip_center"] = hip_center
        if shoulder_center is not None:
            points["shoulder_center"] = shoulder_center

        for limb_name, (a, b) in LIMB_DEFINITIONS.items():
            if a in points and b in points:
                sf.limbs[limb_name] = Limb(limb_name, points[a], points[b])

        for joint_name, (a, b, c) in JOINT_ANGLE_DEFINITIONS.items():
            if a in points and b in points and c in points:
                angle = _angle_3pt_raw(points[a], points[b], points[c])
                if angle is not None:
                    sf.joint_angles[joint_name] = angle

        return sf

    def limb_orientation(self, name: str) -> Optional[float]:
        limb = self.limbs.get(name)
        return limb.orientation_deg if limb else None

    def limb_length(self, name: str) -> Optional[float]:
        limb = self.limbs.get(name)
        return limb.length if limb else None


def _angle_3pt_raw(a, b, c) -> Optional[float]:
    """Local copy of the angle_3pt calculation to avoid a circular import
    with biomechanics.angles (which doesn't depend on pose/ or body_model/
    itself, but keeping this module dependency-free of the biomechanics
    package makes the layering direction explicit: body_model is a peer
    of biomechanics, not a consumer of it)."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return None
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))
