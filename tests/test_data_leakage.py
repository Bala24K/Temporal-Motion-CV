"""
Automated data leakage tests verifying video-level splitting integrity.
Ensures zero overlap between train, validation, and test video IDs.
"""

import numpy as np
import pytest

from recognition.qevd_dataset import train_val_test_split_by_video
from recognition.dataset_synthesis import EXERCISES


def test_train_val_test_split_by_video_leakage():
    # Create synthetic mock dataset dictionary with 100 video IDs across 4 classes
    n_per_class = 25
    mock_video_ids = []
    mock_labels = []

    for ex in EXERCISES:
        for i in range(n_per_class):
            mock_video_ids.append(f"{ex}_vid_{i:03d}.mp4")
            mock_labels.append(ex)

    mock_video_ids = np.array(mock_video_ids)
    mock_labels = np.array(mock_labels)
    mock_X_tab = np.random.randn(len(mock_video_ids), 20)
    mock_X_seq = np.random.randn(len(mock_video_ids), 16, 10)

    dataset_dict = {
        "X_tab": mock_X_tab,
        "X_seq": mock_X_seq,
        "y": mock_labels,
        "video_ids": mock_video_ids,
        "class_counts": {ex: n_per_class for ex in EXERCISES},
    }

    train_data, val_data, test_data = train_val_test_split_by_video(
        dataset_dict, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    train_ids = set(train_data["video_ids"])
    val_ids = set(val_data["video_ids"])
    test_ids = set(test_data["video_ids"])

    # 1. Assert zero intersection between sets
    assert len(train_ids & val_ids) == 0, f"Train-Val Leakage: {train_ids & val_ids}"
    assert len(train_ids & test_ids) == 0, f"Train-Test Leakage: {train_ids & test_ids}"
    assert len(val_ids & test_ids) == 0, f"Val-Test Leakage: {val_ids & test_ids}"

    # 2. Assert union equals all unique video IDs
    all_split_ids = train_ids | val_ids | test_ids
    assert all_split_ids == set(mock_video_ids)

    # 3. Assert shapes and non-empty splits
    assert len(train_data["y"]) > 0
    assert len(val_data["y"]) > 0
    assert len(test_data["y"]) > 0

    # 4. Assert stratified representation across all 4 classes in train, val, test
    for ex in EXERCISES:
        assert np.sum(train_data["y"] == ex) > 0, f"Class {ex} missing in train split"
        assert np.sum(val_data["y"] == ex) > 0, f"Class {ex} missing in val split"
        assert np.sum(test_data["y"] == ex) > 0, f"Class {ex} missing in test split"


def test_reproducible_split_seed():
    # Verify split is deterministic with same seed
    mock_video_ids = np.array([f"vid_{i:02d}.mp4" for i in range(40)])
    mock_labels = np.array(EXERCISES * 10)
    dataset_dict = {
        "X_tab": np.zeros((40, 5)),
        "X_seq": np.zeros((40, 16, 5)),
        "y": mock_labels,
        "video_ids": mock_video_ids,
        "class_counts": {ex: 10 for ex in EXERCISES},
    }

    train1, val1, test1 = train_val_test_split_by_video(dataset_dict, seed=42)
    train2, val2, test2 = train_val_test_split_by_video(dataset_dict, seed=42)

    assert list(train1["video_ids"]) == list(train2["video_ids"])
    assert list(val1["video_ids"]) == list(val2["video_ids"])
    assert list(test1["video_ids"]) == list(test2["video_ids"])
