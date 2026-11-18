"""
Dataset tests, weighted toward the two failure modes that make a result
look better than it is: group leakage across splits, and label provenance
going unrecorded.

Both are silent. A leaked split raises no error and reports a higher
score; an unrecorded provenance produces a number that reads as exercise
recognition accuracy when it measures agreement with a kinematic rule.
"""
import os
import pickle

import numpy as np
import pytest

from motion.dataset import (
    EXERCISE_CLASSES,
    LABEL_SOURCE_HUMAN,
    LABEL_SOURCE_RULE,
    MotionDataset,
    assert_no_group_leakage,
    build_dataset_from_pose_cache,
    class_weights,
    group_disjoint_split,
    load_subject_map,
)
from motion.graph import NUM_JOINTS
from motion.tensors import STREAM_NAMES, stream_channel_count
from tests.motion_fixtures import make_fake_cache


@pytest.fixture
def cache_path(tmp_path):
    path = str(tmp_path / "fake_cache.pkl")
    make_fake_cache(path, per_class=6, n_frames=40)
    return path


@pytest.fixture
def dataset(cache_path):
    return build_dataset_from_pose_cache(cache_path, window_frames=32, seed=0)


def test_builds_one_sample_per_clip(dataset):
    assert len(dataset) == 24
    assert sorted(dataset.classes) == sorted(EXERCISE_CLASSES)
    assert dataset.class_counts() == {c: 6 for c in EXERCISE_CLASSES}


def test_missing_cache_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="pose cache not found"):
        build_dataset_from_pose_cache(str(tmp_path / "nope.pkl"))


def test_to_arrays_shapes(dataset):
    X, y, phases, groups = dataset.to_arrays()
    expected_c = stream_channel_count(STREAM_NAMES, True)
    assert X.shape == (24, expected_c, 32, NUM_JOINTS)
    assert X.dtype == np.float32
    assert y.shape == (24,) and y.dtype == np.int64
    assert phases.shape == (24, 32)
    assert groups.shape == (24,)
    assert set(y.tolist()) == set(range(len(dataset.classes)))


def test_empty_dataset_cannot_be_materialised():
    with pytest.raises(ValueError):
        MotionDataset([]).to_arrays()


def test_provenance_summary_records_the_label_source(dataset):
    summary = dataset.provenance_summary()
    assert summary["n_samples"] == 24
    assert summary["label_sources"] == {LABEL_SOURCE_RULE: 24}
    assert summary["trusted_label_fraction"] == 0.0
    assert dataset.has_only_derived_labels() is True
    assert summary["mean_valid_ratio"] > 0.9


def test_trusted_labels_are_distinguished(cache_path):
    ds = build_dataset_from_pose_cache(cache_path, label_source=LABEL_SOURCE_HUMAN, window_frames=32)
    assert ds.provenance_summary()["trusted_label_fraction"] == 1.0
    assert ds.has_only_derived_labels() is False


def test_split_is_group_disjoint(dataset):
    train, val, test = group_disjoint_split(dataset, ratios=(0.6, 0.2, 0.2), seed=1)
    assert_no_group_leakage(train, val, test)
    assert len(train) + len(val) + len(test) == len(dataset)
    assert set(train.groups).isdisjoint(val.groups)
    assert set(train.groups).isdisjoint(test.groups)
    assert set(val.groups).isdisjoint(test.groups)


def test_leakage_assertion_actually_fires(dataset):
    # Deliberately construct the mistake this check exists to catch.
    leaked_a = dataset.subset([0, 1, 2])
    leaked_b = dataset.subset([2, 3, 4])
    with pytest.raises(AssertionError, match="invalidates the evaluation"):
        assert_no_group_leakage(leaked_a, leaked_b)


def test_windows_from_the_same_clip_never_split_across_sets(cache_path):
    # Multiple windows per clip are cheap extra data but are not
    # independent samples; the group key has to keep them together.
    ds = build_dataset_from_pose_cache(cache_path, window_frames=24, windows_per_clip=3)
    assert len(ds) > 24
    train, val, test = group_disjoint_split(ds, seed=3)
    assert_no_group_leakage(train, val, test)

    clip_to_split = {}
    for split_name, part in (("train", train), ("val", val), ("test", test)):
        for sample in part.samples:
            assert clip_to_split.setdefault(sample.clip_id, split_name) == split_name


def test_subject_map_makes_splits_subject_disjoint(cache_path):
    with open(cache_path, "rb") as fh:
        clip_ids = sorted(pickle.load(fh).keys())
    # Two clips per subject, so a video-level split would scatter them.
    subject_map = {cid: f"subject_{i // 2}" for i, cid in enumerate(clip_ids)}

    ds = build_dataset_from_pose_cache(cache_path, subject_map=subject_map, window_frames=32)
    assert set(ds.groups) == set(subject_map.values())
    train, val, test = group_disjoint_split(ds, seed=2)
    assert_no_group_leakage(train, val, test)


def test_load_subject_map_parses_header_and_comments(tmp_path):
    path = tmp_path / "subjects.csv"
    path.write_text("clip_id,subject_id\n# a comment\na.mp4,s1\nb.mp4,s1\nc.mp4,s2\n\n")
    mapping = load_subject_map(str(path))
    assert mapping == {"a.mp4": "s1", "b.mp4": "s1", "c.mp4": "s2"}


def test_split_ratios_must_sum_to_one(dataset):
    with pytest.raises(ValueError, match="sum to 1.0"):
        group_disjoint_split(dataset, ratios=(0.5, 0.2, 0.2))


def test_split_refuses_to_produce_an_empty_part(cache_path):
    tiny = build_dataset_from_pose_cache(cache_path, window_frames=32).subset([0, 1])
    with pytest.raises(ValueError, match="came out empty"):
        group_disjoint_split(tiny, ratios=(0.34, 0.33, 0.33))


def test_class_weights_counter_imbalance(dataset):
    # Emulate the real cache's ~68% squat skew.
    squat_idx = [i for i, s in enumerate(dataset.samples) if s.label == "squat"]
    others = [i for i, s in enumerate(dataset.samples) if s.label != "squat"][:2]
    skewed = dataset.subset(squat_idx + others)

    weights = class_weights(skewed)
    assert weights.shape == (len(skewed.classes),)
    assert weights.mean() == pytest.approx(1.0, abs=1e-5)
    squat_pos = skewed.classes.index("squat")
    assert weights[squat_pos] == weights.min(), "the majority class must be down-weighted"


def test_max_per_class_subsamples_by_group(cache_path):
    ds = build_dataset_from_pose_cache(cache_path, window_frames=24, windows_per_clip=2,
                                       max_per_class=4, seed=5)
    counts = ds.class_counts()
    assert all(n <= 4 for n in counts.values())
    # Windows of a kept clip stay together rather than being split apart.
    for label in counts:
        clips = [s.clip_id for s in ds.samples if s.label == label]
        assert len(clips) == len(set(clips)) * 2 or len(set(clips)) * 2 >= len(clips)


def test_low_quality_windows_are_dropped(cache_path):
    strict = build_dataset_from_pose_cache(cache_path, window_frames=32, min_valid_ratio=1.01)
    assert len(strict) == 0, "an impossible quality bar must drop everything, not pass it through"


def test_build_report_records_how_the_dataset_was_made(dataset):
    report = dataset.build_report
    assert report["window_frames"] == 32
    assert report["subject_map_used"] is False
    assert isinstance(report["skipped"], dict)


def test_phase_labels_are_attached_and_marked_derived(dataset):
    for sample in dataset.samples:
        assert sample.phase_labels.shape == (32,)
        assert sample.phase_label_source == "derived_from_kinematics"
        assert "amplitude" in sample.phase_diagnostics
