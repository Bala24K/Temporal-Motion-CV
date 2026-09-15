"""
Feature extraction from the structured, temporal body representation
(body_model/), replacing the earlier hand-picked 8-angle feature vector
(recognition/features.py) with a systematic one derived from the whole
skeleton graph: every limb's orientation/velocity/acceleration, every
joint's angle/velocity/acceleration, and left/right symmetry — read off
the same structure for every exercise, not picked per-exercise by hand.

This is the "vision" version's actual point: the representation doesn't
know what a bicep curl is. It just describes the body and how it's
moving; whatever's predictive is left for the model to find.
"""
from typing import List, Tuple
import numpy as np

from body_model.skeleton import LIMB_NAMES, JOINT_NAMES, SYMMETRY_PAIRS
from body_model.sequence import SkeletonSequence

VALUE_FILL = 135.0    # neutral mid-range angle, for missing orientation/angle readings
DERIVATIVE_FILL = 0.0  # "no motion" is a reasonable neutral guess for missing velocity/acceleration


def feature_names_and_kinds() -> Tuple[List[str], List[str]]:
    names: List[str] = []
    kinds: List[str] = []
    for limb in LIMB_NAMES:
        names += [f"{limb}_orientation", f"{limb}_ang_velocity", f"{limb}_ang_accel"]
        kinds += ["value", "derivative", "derivative"]
    for joint in JOINT_NAMES:
        names += [f"{joint}_angle", f"{joint}_ang_velocity", f"{joint}_ang_accel"]
        kinds += ["value", "derivative", "derivative"]
    for left, right in SYMMETRY_PAIRS:
        names += [f"{left}_vs_{right}_orientation_diff", f"{left}_vs_{right}_velocity_diff"]
        kinds += ["derivative", "derivative"]  # a diff of 0 is a reasonable neutral default too
    return names, kinds


FEATURE_NAMES, FEATURE_KINDS = feature_names_and_kinds()


def extract_features(seq: SkeletonSequence) -> np.ndarray:
    """Returns a len(FEATURE_NAMES)-element array. Missing values (gaps in
    the window, occluded landmarks) come through as NaN — see
    fill_missing() before feeding this to a model that can't handle NaN."""
    values: List[float] = []
    for limb in LIMB_NAMES:
        v, vel, acc = seq.limb_kinematics(limb)
        values += [v, vel, acc]
    for joint in JOINT_NAMES:
        v, vel, acc = seq.joint_kinematics(joint)
        values += [v, vel, acc]
    sym = seq.symmetry_features()
    for left, right in SYMMETRY_PAIRS:
        values.append(sym[f"{left}_vs_{right}_orientation_diff"])
        values.append(sym[f"{left}_vs_{right}_velocity_diff"])
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def fill_missing(features: np.ndarray) -> np.ndarray:
    """Fills NaN using a DIFFERENT default depending on feature kind — a
    missing orientation/angle gets a neutral mid-range value (0 would
    look like a real, extreme reading), while a missing velocity or
    acceleration reasonably defaults to 0 (no known motion)."""
    out = features.copy()
    for i, kind in enumerate(FEATURE_KINDS):
        if np.isnan(out[i]):
            out[i] = VALUE_FILL if kind == "value" else DERIVATIVE_FILL
    return out
