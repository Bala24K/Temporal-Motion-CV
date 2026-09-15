"""
Dataset builder tying together the synthetic full-body pose generator
(pose_sequence_synthesis.py), the structured body representation
(body_model/), and feature extraction — in two paired forms generated
from the EXACT SAME underlying synthetic windows (same seed, same
iteration order), so model comparisons are apples-to-apples:

  - "engineered" (skeleton_features.py): final-frame limb/joint
    kinematics + symmetry, a fixed-length vector for tree-based/MLP
    models.
  - "sequence": the raw per-frame limb-orientation/joint-angle values
    across the whole window (T x F), for a model that learns temporal
    patterns itself (recognition/temporal_model.py) rather than being
    handed pre-computed velocity/acceleration.

Same synthetic-data caveats as dataset_synthesis.py and
pose_sequence_synthesis.py apply — see those modules' docstrings.
"""
from typing import Tuple
import numpy as np

from recognition.pose_sequence_synthesis import make_body_profile, generate_window, EXERCISES
from body_model.skeleton import SkeletonFrame, LIMB_NAMES, JOINT_NAMES
from body_model.sequence import SkeletonSequence
from recognition.skeleton_features import extract_features, fill_missing

RAW_PER_FRAME_NAMES = [f"{l}_orientation" for l in LIMB_NAMES] + [f"{j}_angle" for j in JOINT_NAMES]
RAW_NEUTRAL_FILL = 135.0


def _raw_frame_vector(sf: SkeletonFrame) -> np.ndarray:
    values = [sf.limb_orientation(l) for l in LIMB_NAMES] + [sf.joint_angles.get(j) for j in JOINT_NAMES]
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def generate_engineered_dataset(n_subjects: int = 30, windows_per_subject_per_exercise: int = 15,
                                 n_frames: int = 16, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y, subject_ids = [], [], []
    for subj_id in range(n_subjects):
        profile = make_body_profile(rng)
        for exercise in EXERCISES:
            for _ in range(windows_per_subject_per_exercise):
                frames = generate_window(rng, exercise, profile, n_frames=n_frames)
                seq = SkeletonSequence()
                for f in frames:
                    seq.push(SkeletonFrame.from_pose_frame(f))
                X.append(fill_missing(extract_features(seq)))
                y.append(exercise)
                subject_ids.append(subj_id)
    return np.array(X), np.array(y), np.array(subject_ids)


def generate_sequence_dataset(n_subjects: int = 30, windows_per_subject_per_exercise: int = 15,
                               n_frames: int = 16, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same loop structure/order and same seed as generate_engineered_dataset
    by design — the two must consume the RNG identically so example i in
    one dataset is the same underlying synthetic window as example i in
    the other. Verified by a test that checks this pairing directly."""
    rng = np.random.default_rng(seed)
    X, y, subject_ids = [], [], []
    for subj_id in range(n_subjects):
        profile = make_body_profile(rng)
        for exercise in EXERCISES:
            for _ in range(windows_per_subject_per_exercise):
                frames = generate_window(rng, exercise, profile, n_frames=n_frames)
                seq_matrix = []
                for f in frames:
                    sf = SkeletonFrame.from_pose_frame(f)
                    vec = _raw_frame_vector(sf)
                    vec = np.nan_to_num(vec, nan=RAW_NEUTRAL_FILL)
                    seq_matrix.append(vec)
                X.append(np.array(seq_matrix))  # (n_frames, F_raw)
                y.append(exercise)
                subject_ids.append(subj_id)
    return np.array(X), np.array(y), np.array(subject_ids)
