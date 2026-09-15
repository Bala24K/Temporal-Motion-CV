"""
Geometric feature extraction for exercise classification (V1 §15).

Extracts a fixed-length feature vector from a single PoseFrame — the same
joint angles the rule-based analyzers in exercises/ already compute,
repackaged for a classifier rather than a state machine. Missing/low-
visibility joints produce NaN, not a fabricated angle; see fill_missing()
for how the classifier pipeline handles that before scikit-learn (which
can't take NaN directly).
"""
from typing import List
import numpy as np

from biomechanics.angles import angle_3pt
from pose.landmarks import PoseFrame

FEATURE_NAMES: List[str] = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_shoulder_elevation_angle", "right_shoulder_elevation_angle",
]

MIN_VISIBILITY = 0.5
NEUTRAL_FILL_VALUE = 135.0  # a mid-range angle, used to impute NaN features before classification


def _angle_or_nan(frame: PoseFrame, a: str, b: str, c: str) -> float:
    pa, pb, pc = frame.get(a), frame.get(b), frame.get(c)
    if pa is None or pb is None or pc is None:
        return float("nan")
    if min(pa.visibility, pb.visibility, pc.visibility) < MIN_VISIBILITY:
        return float("nan")
    angle = angle_3pt(pa.xy(), pb.xy(), pc.xy())
    return angle if angle is not None else float("nan")


def extract_features(frame: PoseFrame) -> np.ndarray:
    """Returns an 8-element array matching FEATURE_NAMES, in order."""
    return np.array([
        _angle_or_nan(frame, "left_shoulder", "left_elbow", "left_wrist"),
        _angle_or_nan(frame, "right_shoulder", "right_elbow", "right_wrist"),
        _angle_or_nan(frame, "left_hip", "left_knee", "left_ankle"),
        _angle_or_nan(frame, "right_hip", "right_knee", "right_ankle"),
        _angle_or_nan(frame, "left_shoulder", "left_hip", "left_knee"),
        _angle_or_nan(frame, "right_shoulder", "right_hip", "right_knee"),
        _angle_or_nan(frame, "left_hip", "left_shoulder", "left_wrist"),
        _angle_or_nan(frame, "right_hip", "right_shoulder", "right_wrist"),
    ], dtype=float)


def fill_missing(features: np.ndarray, fill_value: float = NEUTRAL_FILL_VALUE) -> np.ndarray:
    """
    Replaces NaN entries (missing/low-confidence joints) with a neutral
    mid-range angle, since RandomForestClassifier can't handle NaN
    directly. A neutral fill — rather than 0, which would look like a
    physically extreme, fully-folded joint — keeps a missing reading from
    masquerading as a real (and misleading) data point.
    """
    out = features.copy()
    out[np.isnan(out)] = fill_value
    return out
