"""
Our own internal pose representation. The rest of the app depends on THIS,
never directly on MediaPipe's output types — so MediaPipe can be swapped
for another pose model later without touching biomechanics/exercises/ui.

IMPORTANT — anatomical vs screen-space:
MediaPipe landmark names are ANATOMICAL, i.e. "left_shoulder" means the
SUBJECT's left shoulder. On a webcam feed that's typically mirrored for
display (so it feels like a mirror to the user), the subject's left shoulder
appears on the RIGHT side of the screen. Every module in this codebase
should reason in anatomical terms. Only ui/overlay.py, at the point it
draws pixels, needs to think about mirroring — see the note there.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import time

# The subset of MediaPipe Pose Landmarker's 33 landmarks we actually use.
# (Indices are fixed by the MediaPipe Pose model spec.)
MEDIAPIPE_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class Landmark:
    """Normalized coordinates (0-1 range as given by MediaPipe), plus depth
    and a confidence score. x/y are relative to the image frame; z is a
    rough relative depth (less reliable than x/y — don't lean on it hard
    for anything angle-critical without validating first)."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class PoseFrame:
    """One frame's worth of landmarks, keyed by ANATOMICAL name."""
    landmarks: Dict[str, Landmark] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get(self, name: str) -> Optional[Landmark]:
        return self.landmarks.get(name)

    def visible(self, name: str, min_visibility: float = 0.5) -> bool:
        lm = self.landmarks.get(name)
        return lm is not None and lm.visibility >= min_visibility

    def all_visible(self, names, min_visibility: float = 0.5) -> bool:
        return all(self.visible(n, min_visibility) for n in names)

    def torso_scale(self) -> Optional[float]:
        """Rough body-scale normalizer: shoulder-to-hip distance, averaged
        across sides where available. Used to turn pixel-ish displacements
        into scale-invariant thresholds (see biomechanics/distances.py)."""
        pairs = [("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")]
        vals = []
        for s, h in pairs:
            ls, lh = self.get(s), self.get(h)
            if ls and lh:
                vals.append(((ls.x - lh.x) ** 2 + (ls.y - lh.y) ** 2) ** 0.5)
        if not vals:
            return None
        return sum(vals) / len(vals)


class PoseSequence:
    """Rolling buffer of PoseFrames."""

    def __init__(self, max_len: int = 90):
        self.max_len = max_len
        self.frames = []

    def push(self, frame: PoseFrame):
        self.frames.append(frame)
        if len(self.frames) > self.max_len:
            self.frames.pop(0)

    def clear(self):
        """Clears all stored historical pose frames when tracking is lost."""
        self.frames.clear()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


def is_valid_person_pose(frame: Optional[PoseFrame], exercise_id: Optional[str] = None, min_visibility: float = 0.05) -> bool:
    """
    Evaluates whether a PoseFrame contains a valid, high-confidence person pose.
    Requires minimum landmark presence, minimum visibility, and exercise-relevant key joints.
    """
    if frame is None or not frame.landmarks:
        return False

    # 1. Total landmarks present check
    valid_lms = [lm for lm in frame.landmarks.values() if lm.visibility >= min_visibility]
    if len(valid_lms) < 3:
        return False

    # 2. Key body joint requirements (at least shoulders or hips visible)
    has_shoulders = frame.visible("left_shoulder", min_visibility) or frame.visible("right_shoulder", min_visibility)
    has_hips = frame.visible("left_hip", min_visibility) or frame.visible("right_hip", min_visibility)
    if not (has_shoulders or has_hips):
        return False

    # 3. Exercise-specific joint requirements if specified
    if exercise_id == "lateral_raise":
        left_arm = frame.all_visible(["left_hip", "left_shoulder", "left_wrist"], min_visibility)
        right_arm = frame.all_visible(["right_hip", "right_shoulder", "right_wrist"], min_visibility)
        if not (left_arm or right_arm):
            return False
    elif exercise_id in ("bicep_curl", "shoulder_press"):
        left_arm = frame.all_visible(["left_shoulder", "left_elbow", "left_wrist"], min_visibility)
        right_arm = frame.all_visible(["right_shoulder", "right_elbow", "right_wrist"], min_visibility)
        if not (left_arm or right_arm):
            return False
    elif exercise_id == "squat":
        left_leg = frame.all_visible(["left_hip", "left_knee", "left_ankle"], min_visibility)
        right_leg = frame.all_visible(["right_hip", "right_knee", "right_ankle"], min_visibility)
        if not (left_leg or right_leg):
            return False

    return True
