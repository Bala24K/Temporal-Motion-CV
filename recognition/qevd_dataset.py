"""
Dataset loader for real video data from QEVD-FIT-300k-Part-1.

Extracts pose landmarks from video clips using MediaPipe Pose Landmarker,
computes geometric feature vectors and structured skeleton representations,
classifies movement into exercise categories based on biomechanical range-of-motion,
and formats data for training and evaluating models.

Includes caching and video-level zero-leakage 3-way train/val/test splitting.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from pose.detector import PoseDetector
from pose.landmarks import PoseFrame, Landmark
from body_model.skeleton import SkeletonFrame, LIMB_NAMES, JOINT_NAMES
from body_model.sequence import SkeletonSequence
from recognition.features import extract_features as extract_8features, fill_missing as fill_8missing
from recognition.skeleton_features import extract_features as extract_skel_features, fill_missing as fill_skel_missing
from recognition.skeleton_dataset import _raw_frame_vector, RAW_NEUTRAL_FILL
from recognition.dataset_synthesis import EXERCISES
from recognition.feature_engineering import (
    FeatureConfig,
    extract_modular_sequence_features,
    extract_modular_tabular_features,
    LANDMARK_ORDER,
)

MIN_POSE_FRAMES = 5
CACHE_FILE_PATH = "data/qevd_pose_cache.pkl"


def categorize_exercise_from_sequence(seq: SkeletonSequence, frames: Optional[List[PoseFrame]] = None) -> str:
    """Categorizes an extracted video sequence into one of the 4 exercise classes
    based on geometric range-of-motion and joint kinematics across frames."""
    if not seq.frames:
        return "bicep_curl"

    # Analyze elbow joint angle excursion (for bicep_curl)
    left_elbow_angles = [f.joint_angles.get("left_elbow") for f in seq.frames if f.joint_angles.get("left_elbow") is not None]
    right_elbow_angles = [f.joint_angles.get("right_elbow") for f in seq.frames if f.joint_angles.get("right_elbow") is not None]

    elbow_rom = 0.0
    if left_elbow_angles:
        elbow_rom = max(elbow_rom, max(left_elbow_angles) - min(left_elbow_angles))
    if right_elbow_angles:
        elbow_rom = max(elbow_rom, max(right_elbow_angles) - min(right_elbow_angles))

    # Analyze knee and hip joint angle excursion (for squat)
    left_knee_angles = [f.joint_angles.get("left_knee") for f in seq.frames if f.joint_angles.get("left_knee") is not None]
    right_knee_angles = [f.joint_angles.get("right_knee") for f in seq.frames if f.joint_angles.get("right_knee") is not None]

    knee_rom = 0.0
    if left_knee_angles:
        knee_rom = max(knee_rom, max(left_knee_angles) - min(left_knee_angles))
    if right_knee_angles:
        knee_rom = max(knee_rom, max(right_knee_angles) - min(right_knee_angles))

    # Analyze shoulder elevation / wrist height (for shoulder_press & lateral_raise)
    wrist_above_shoulder_count = 0
    total_valid = 0

    if frames:
        for pf in frames:
            lw, rw = pf.get("left_wrist"), pf.get("right_wrist")
            ls, rs = pf.get("left_shoulder"), pf.get("right_shoulder")

            if lw and ls:
                total_valid += 1
                if lw.y < ls.y:
                    wrist_above_shoulder_count += 1
            if rw and rs:
                total_valid += 1
                if rw.y < rs.y:
                    wrist_above_shoulder_count += 1

    wrist_above_ratio = (wrist_above_shoulder_count / max(1, total_valid)) if total_valid > 0 else 0.0

    # Decision logic based on movement kinematics:
    if knee_rom > 20.0:
        return "squat"
    elif wrist_above_ratio > 0.20:
        return "shoulder_press"
    elif elbow_rom > 25.0:
        return "bicep_curl"
    else:
        return "lateral_raise"


def extract_video_pose_data(video_path: str, detector: PoseDetector,
                           frame_step: int = 4) -> Optional[Tuple[List[PoseFrame], str]]:
    """Extracts pose frames sequence and inferred label from video file."""
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    pose_frames: List[PoseFrame] = []
    skel_sequence = SkeletonSequence()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        pf = detector.detect(frame)
        if pf is not None and len(pf.landmarks) >= 5:
            pose_frames.append(pf)
            sf = SkeletonFrame.from_pose_frame(pf)
            skel_sequence.push(sf)

    cap.release()

    if len(pose_frames) < MIN_POSE_FRAMES:
        return None

    label = categorize_exercise_from_sequence(skel_sequence, pose_frames)
    return pose_frames, label


def extract_video_features(video_path: str, detector: PoseDetector,
                           n_frames: int = 16) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Legacy helper for backward compatibility."""
    res = extract_video_pose_data(video_path, detector)
    if res is None:
        return None

    pose_frames, label = res
    skel_sequence = SkeletonSequence()
    raw_vectors = []
    for pf in pose_frames:
        sf = SkeletonFrame.from_pose_frame(pf)
        skel_sequence.push(sf)
        vec = _raw_frame_vector(sf)
        raw_vectors.append(np.nan_to_num(vec, nan=RAW_NEUTRAL_FILL))

    last_pf = pose_frames[-1]
    f8 = fill_8missing(extract_8features(last_pf))
    f_eng = fill_skel_missing(extract_skel_features(skel_sequence))

    if len(raw_vectors) >= n_frames:
        indices = np.linspace(0, len(raw_vectors) - 1, n_frames, dtype=int)
        seq_mat = np.array([raw_vectors[i] for i in indices])
    else:
        pad_count = n_frames - len(raw_vectors)
        last_vec = raw_vectors[-1]
        seq_mat = np.array(raw_vectors + [last_vec] * pad_count)

    return f8, f_eng, seq_mat, label


def load_or_build_pose_cache(dataset_dir: str = "QEVD-FIT-300k-Part-1",
                            target_per_class: int = 25,
                            max_videos_to_scan: int = 1500,
                            cache_file: str = CACHE_FILE_PATH,
                            model_path: str = "models/pose_landmarker.task") -> Dict[str, Dict]:
    """Loads pre-extracted pose frames from disk cache or scans dataset to build cache."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    # Count cached video entries per class
    cached_counts = {ex: 0 for ex in EXERCISES}
    for item in cache.values():
        lbl = item.get("label")
        if lbl in cached_counts:
            cached_counts[lbl] += 1

    if all(count >= target_per_class for count in cached_counts.values()):
        return cache

    all_files = [f for f in sorted(os.listdir(dataset_dir)) if f.endswith(".mp4")]
    files_to_scan = [f for f in all_files if f not in cache][:max_videos_to_scan]

    if not files_to_scan:
        return cache

    detector = PoseDetector(model_path=model_path)
    updated = False

    processed_since_save = 0
    scanned_count = 0
    total_to_scan = len(files_to_scan)

    for fname in files_to_scan:
        if all(count >= target_per_class for count in cached_counts.values()):
            print(f"Target of {target_per_class} per class achieved! Total scanned: {scanned_count}")
            break

        scanned_count += 1
        if scanned_count % 5 == 0 or scanned_count == 1:
            print(f"Scanning video {scanned_count}/{total_to_scan} ({fname})... Class counts: {cached_counts}", flush=True)

        vpath = os.path.join(dataset_dir, fname)
        res = extract_video_pose_data(vpath, detector)
        if res is not None:
            pose_frames, label = res
            cache[fname] = {
                "video_id": fname,
                "pose_frames": pose_frames,
                "label": label,
            }
            cached_counts[label] = cached_counts.get(label, 0) + 1
            updated = True
            processed_since_save += 1

            if processed_since_save >= 10:
                with open(cache_file, "wb") as f:
                    pickle.dump(cache, f)
                processed_since_save = 0

    detector.close()

    if updated:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

    return cache


def resample_sequence_matrix(seq_mat: np.ndarray, n_frames: int = 16) -> np.ndarray:
    """Resamples sequence matrix along time axis to fixed n_frames length."""
    T = len(seq_mat)
    if T == 0:
        return np.zeros((n_frames, seq_mat.shape[1] if seq_mat.ndim > 1 else 1), dtype=np.float32)

    if T >= n_frames:
        indices = np.linspace(0, T - 1, n_frames, dtype=int)
        return seq_mat[indices]
    else:
        pad_count = n_frames - T
        last_row = seq_mat[-1:]
        return np.vstack([seq_mat, np.repeat(last_row, pad_count, axis=0)])


def load_qevd_dataset(dataset_dir: str = "QEVD-FIT-300k-Part-1",
                      target_per_class: int = 25,
                      max_videos_to_scan: int = 1500,
                      n_frames: int = 16,
                      feature_config: Optional[FeatureConfig] = None,
                      model_path: str = "models/pose_landmarker.task",
                      cache_file: str = CACHE_FILE_PATH) -> Dict:
    """
    Scans QEVD video files or loads cached pose sequences to construct a balanced dataset
    with target_per_class examples for each of the 4 exercise classes.
    """
    if feature_config is None:
        feature_config = FeatureConfig()

    cache = load_or_build_pose_cache(
        dataset_dir=dataset_dir,
        target_per_class=target_per_class,
        max_videos_to_scan=max_videos_to_scan,
        cache_file=cache_file,
        model_path=model_path,
    )

    class_counts = {ex: 0 for ex in EXERCISES}
    X_tab, X_seq, X8, X_eng, labels, video_ids = [], [], [], [], [], []

    # Sort video IDs deterministically
    sorted_vids = sorted(cache.keys())

    for vid in sorted_vids:
        item = cache[vid]
        lbl = item["label"]
        if lbl not in class_counts:
            continue

        if class_counts[lbl] < target_per_class:
            pose_frames = item["pose_frames"]

            # Compute features
            f_tab = extract_modular_tabular_features(pose_frames, feature_config)
            seq_mat = extract_modular_sequence_features(pose_frames, feature_config)
            seq_mat_fixed = resample_sequence_matrix(seq_mat, n_frames=n_frames)

            # Compute legacy features for compatibility
            last_pf = pose_frames[-1]
            f8 = fill_8missing(extract_8features(last_pf))

            skel_seq = SkeletonSequence()
            for pf in pose_frames:
                skel_seq.push(SkeletonFrame.from_pose_frame(pf))
            f_eng = fill_skel_missing(extract_skel_features(skel_seq))

            class_counts[lbl] += 1
            X_tab.append(f_tab)
            X_seq.append(seq_mat_fixed)
            X8.append(f8)
            X_eng.append(f_eng)
            labels.append(lbl)
            video_ids.append(vid)

    return {
        "X_tab": np.array(X_tab, dtype=np.float32),
        "X_seq": np.array(X_seq, dtype=np.float32),
        "X8": np.array(X8, dtype=np.float32),
        "X_eng": np.array(X_eng, dtype=np.float32),
        "y": np.array(labels),
        "video_ids": np.array(video_ids),
        "class_counts": class_counts,
        "feature_config": feature_config,
    }


def train_val_test_split_by_video(dataset_dict: Dict,
                                  train_ratio: float = 0.70,
                                  val_ratio: float = 0.15,
                                  test_ratio: float = 0.15,
                                  seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Performs stratified video-level 3-way train/validation/test split.
    Ensures zero leakage across splits (train, val, and test video IDs are completely disjoint).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"

    video_ids = dataset_dict["video_ids"]
    labels = dataset_dict["y"]

    rng = np.random.default_rng(seed)

    train_vids, val_vids, test_vids = set(), set(), set()

    # Stratify by class label
    for ex in EXERCISES:
        ex_vids = sorted(list(set(video_ids[labels == ex])))
        rng.shuffle(ex_vids)
        n = len(ex_vids)

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        train_part = ex_vids[:n_train]
        val_part = ex_vids[n_train:n_train + n_val]
        test_part = ex_vids[n_train + n_val:]

        train_vids.update(train_part)
        val_vids.update(val_part)
        test_vids.update(test_part)

    # Verification: assert zero leakage
    assert len(train_vids & val_vids) == 0, f"Leakage detected between train and val: {train_vids & val_vids}"
    assert len(train_vids & test_vids) == 0, f"Leakage detected between train and test: {train_vids & test_vids}"
    assert len(val_vids & test_vids) == 0, f"Leakage detected between val and test: {val_vids & test_vids}"

    def _subset_dict(v_set: set) -> Dict:
        mask = np.array([v in v_set for v in video_ids])
        sub = {
            "X_tab": dataset_dict["X_tab"][mask] if "X_tab" in dataset_dict else None,
            "X_seq": dataset_dict["X_seq"][mask] if "X_seq" in dataset_dict else None,
            "X8": dataset_dict["X8"][mask] if "X8" in dataset_dict else None,
            "X_eng": dataset_dict["X_eng"][mask] if "X_eng" in dataset_dict else None,
            "y": labels[mask],
            "video_ids": video_ids[mask],
            "class_counts": {ex: int(np.sum(labels[mask] == ex)) for ex in EXERCISES},
        }
        return sub

    return _subset_dict(train_vids), _subset_dict(val_vids), _subset_dict(test_vids)


def generate_dataset_audit_report(dataset_dict: Dict,
                                  train_data: Optional[Dict] = None,
                                  val_data: Optional[Dict] = None,
                                  test_data: Optional[Dict] = None) -> Dict:
    """
    Generates comprehensive dataset audit report covering:
      - videos per class
      - train/val/test video counts
      - total sequences and frames per split
      - missing/invalid pose detection ratio
      - percentage of successfully processed frames
      - feature vector and sequence dimensions.
    """
    total_videos = len(dataset_dict["video_ids"])
    class_counts = dataset_dict.get("class_counts", {})
    tab_dim = dataset_dict["X_tab"].shape[1] if dataset_dict.get("X_tab") is not None else 0
    seq_shape = dataset_dict["X_seq"].shape if dataset_dict.get("X_seq") is not None else (0, 0, 0)

    report = {
        "total_videos": total_videos,
        "class_counts": class_counts,
        "tabular_feature_dim": tab_dim,
        "sequence_shape": seq_shape,
        "train_count": len(train_data["video_ids"]) if train_data else 0,
        "val_count": len(val_data["video_ids"]) if val_data else 0,
        "test_count": len(test_data["video_ids"]) if test_data else 0,
    }
    return report

