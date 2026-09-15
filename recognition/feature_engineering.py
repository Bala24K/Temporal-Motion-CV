"""
Modular temporal feature engineering for exercise pose sequence classification.

Extracts independently switchable feature groups from pose landmark sequences:
  - raw_landmarks: Normalized (x, y, z) coordinates relative to mid-hip center.
  - joint_angles: 8 biomechanical joint angles across frames.
  - velocity: 1st-order temporal derivative (frame-to-frame difference).
  - acceleration: 2nd-order temporal derivative (velocity difference).
  - range_of_motion: Excursion (max - min) across the temporal sequence.
  - symmetry: Left vs right joint angle absolute differences.

Supports generating both:
  - Frame-level sequence matrices (T x F) for temporal 1D-CNN models.
  - Aggregated 1D feature vectors (F_total,) for tree/linear/MLP models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from pose.landmarks import PoseFrame
from body_model.skeleton import SkeletonFrame, LIMB_NAMES, JOINT_NAMES
from body_model.sequence import SkeletonSequence
from recognition.features import extract_features as extract_8angles, fill_missing as fill_8missing

LANDMARK_ORDER = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

DEFAULT_FILL_ANGLE = 135.0


@dataclass
class FeatureConfig:
    """Controls which feature groups are included in temporal feature extraction."""
    use_raw_landmarks: bool = True
    use_joint_angles: bool = True
    use_velocity: bool = True
    use_acceleration: bool = True
    use_range_of_motion: bool = True
    use_symmetry: bool = True

    def get_group_name(self) -> str:
        """Returns descriptive name for ablation study levels."""
        groups = []
        if self.use_raw_landmarks:
            groups.append("landmarks")
        if self.use_joint_angles:
            groups.append("angles")
        if self.use_velocity:
            groups.append("vel")
        if self.use_acceleration:
            groups.append("acc")
        if self.use_symmetry:
            groups.append("sym")
        if self.use_range_of_motion:
            groups.append("rom")
        return "+".join(groups) if groups else "none"


def extract_raw_landmarks_matrix(pose_frames: List[PoseFrame]) -> np.ndarray:
    """
    Extracts normalized (x, y, z) coordinates relative to mid-hip center
    scaled by torso length.
    Output shape: (T, 13 * 3) = (T, 39).
    """
    T = len(pose_frames)
    mat = np.zeros((T, len(LANDMARK_ORDER) * 3), dtype=np.float32)

    for t, pf in enumerate(pose_frames):
        lh, rh = pf.get("left_hip"), pf.get("right_hip")
        if lh is not None and rh is not None:
            mid_hip_x = (lh.x + rh.x) / 2.0
            mid_hip_y = (lh.y + rh.y) / 2.0
            mid_hip_z = (lh.z + rh.z) / 2.0
        else:
            mid_hip_x, mid_hip_y, mid_hip_z = 0.5, 0.5, 0.0

        torso_scale = pf.torso_scale() or 1.0
        if torso_scale <= 1e-4:
            torso_scale = 1.0

        row = []
        for name in LANDMARK_ORDER:
            lm = pf.get(name)
            if lm is not None:
                norm_x = (lm.x - mid_hip_x) / torso_scale
                norm_y = (lm.y - mid_hip_y) / torso_scale
                norm_z = (lm.z - mid_hip_z) / torso_scale
                row.extend([norm_x, norm_y, norm_z])
            else:
                row.extend([0.0, 0.0, 0.0])

        mat[t] = np.array(row, dtype=np.float32)

    return mat


def extract_joint_angles_matrix(pose_frames: List[PoseFrame]) -> np.ndarray:
    """
    Extracts 8 joint angles across frames.
    Output shape: (T, 8).
    """
    T = len(pose_frames)
    mat = np.zeros((T, 8), dtype=np.float32)

    for t, pf in enumerate(pose_frames):
        angles = extract_8angles(pf)
        angles_filled = fill_8missing(angles, fill_value=DEFAULT_FILL_ANGLE)
        mat[t] = angles_filled

    return mat


def compute_temporal_derivatives(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1st order velocity (frame-to-frame diff) and 2nd order acceleration.
    Given matrix of shape (T, F), returns:
      velocity: (T, F)
      acceleration: (T, F)
    """
    if len(matrix) <= 1:
        vel = np.zeros_like(matrix)
        acc = np.zeros_like(matrix)
        return vel, acc

    # 1st derivative (velocity)
    vel = np.zeros_like(matrix)
    vel[1:] = np.diff(matrix, axis=0)
    vel[0] = vel[1] if len(vel) > 1 else 0.0

    # 2nd derivative (acceleration)
    acc = np.zeros_like(matrix)
    acc[1:] = np.diff(vel, axis=0)
    acc[0] = acc[1] if len(acc) > 1 else 0.0

    return vel, acc


def compute_symmetry_matrix(angles_mat: np.ndarray) -> np.ndarray:
    """
    Computes absolute difference between left and right side joint angles.
    angles_mat order: [L_elbow, R_elbow, L_knee, R_knee, L_hip, R_hip, L_sh_elev, R_sh_elev]
    Output shape: (T, 4).
    """
    T = len(angles_mat)
    sym = np.zeros((T, 4), dtype=np.float32)

    if angles_mat.shape[1] >= 8:
        sym[:, 0] = np.abs(angles_mat[:, 0] - angles_mat[:, 1])  # elbow
        sym[:, 1] = np.abs(angles_mat[:, 2] - angles_mat[:, 3])  # knee
        sym[:, 2] = np.abs(angles_mat[:, 4] - angles_mat[:, 5])  # hip
        sym[:, 3] = np.abs(angles_mat[:, 6] - angles_mat[:, 7])  # shoulder elevation

    return sym


def compute_range_of_motion_vector(matrix: np.ndarray) -> np.ndarray:
    """
    Computes excursion (max - min) for each column feature across time T.
    Output shape: (F,).
    """
    if len(matrix) == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return np.ptp(matrix, axis=0).astype(np.float32)


def extract_modular_sequence_features(pose_frames: List[PoseFrame],
                                       config: FeatureConfig) -> np.ndarray:
    """
    Extracts frame-level matrix of features (T, F_total) according to FeatureConfig.
    Used for 1D-CNN or temporal models.
    """
    feature_blocks = []

    raw_lm = extract_raw_landmarks_matrix(pose_frames)
    angles = extract_joint_angles_matrix(pose_frames)

    if config.use_raw_landmarks:
        feature_blocks.append(raw_lm)

    if config.use_joint_angles:
        feature_blocks.append(angles)

    base_mat = np.hstack(feature_blocks) if feature_blocks else raw_lm

    vel, acc = compute_temporal_derivatives(base_mat)

    if config.use_velocity:
        feature_blocks.append(vel)

    if config.use_acceleration:
        feature_blocks.append(acc)

    if config.use_symmetry:
        sym = compute_symmetry_matrix(angles)
        feature_blocks.append(sym)

    if not feature_blocks:
        # Fallback if all turned off
        return raw_lm

    return np.hstack(feature_blocks).astype(np.float32)


def extract_modular_tabular_features(pose_frames: List[PoseFrame],
                                     config: FeatureConfig) -> np.ndarray:
    """
    Aggregates temporal sequence into a single 1D feature vector for non-sequential
    classifiers (Random Forest, XGBoost, Logistic Regression, MLP).

    Combines temporal summary statistics: mean, std, min, max, excursion (ROM),
    last frame value, and symmetry.
    """
    seq_mat = extract_modular_sequence_features(pose_frames, config)
    if len(seq_mat) == 0:
        return np.zeros(1, dtype=np.float32)

    vec_parts = [
        np.mean(seq_mat, axis=0),
        np.std(seq_mat, axis=0),
        np.min(seq_mat, axis=0),
        np.max(seq_mat, axis=0),
        seq_mat[-1],  # final frame snapshot
    ]

    if config.use_range_of_motion:
        rom = compute_range_of_motion_vector(seq_mat)
        vec_parts.append(rom)

    return np.concatenate(vec_parts).astype(np.float32)
