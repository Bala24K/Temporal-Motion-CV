import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from recognition.dataset_synthesis import generate_dataset, generate_example, _make_subject, EXERCISES  # noqa: E402
from recognition.features import FEATURE_NAMES  # noqa: E402


def test_dataset_shapes_and_labels():
    X, y, subject_ids = generate_dataset(n_subjects=5, examples_per_subject_per_exercise=10, seed=1)
    n_expected = 5 * len(EXERCISES) * 10
    assert X.shape == (n_expected, 8)
    assert len(y) == n_expected
    assert len(subject_ids) == n_expected
    assert set(y) == set(EXERCISES)
    assert set(subject_ids) == set(range(5))


def test_dataset_is_reproducible_with_same_seed():
    X1, y1, s1 = generate_dataset(n_subjects=3, examples_per_subject_per_exercise=5, seed=42)
    X2, y2, s2 = generate_dataset(n_subjects=3, examples_per_subject_per_exercise=5, seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_different_seeds_give_different_data():
    X1, _, _ = generate_dataset(n_subjects=3, examples_per_subject_per_exercise=5, seed=1)
    X2, _, _ = generate_dataset(n_subjects=3, examples_per_subject_per_exercise=5, seed=2)
    assert not np.array_equal(X1, X2)


def test_squat_examples_show_lower_mean_knee_angle_than_curl():
    """Sanity check that the encoded geometric definitions actually
    produce separable data: a squat sweeps the knee angle down toward 90,
    a curl doesn't touch the knee at all — squat's mean knee angle across
    many random phases should be meaningfully lower than curl's."""
    rng = np.random.default_rng(0)
    subject = _make_subject(rng)
    knee_idx = FEATURE_NAMES.index("left_knee_angle")

    squat_knees = [generate_example(rng, "squat", subject)[knee_idx] for _ in range(500)]
    curl_knees = [generate_example(rng, "bicep_curl", subject)[knee_idx] for _ in range(500)]

    assert np.mean(squat_knees) < np.mean(curl_knees) - 20


def test_press_and_raise_differ_in_elbow_variance():
    """The feature meant to separate press from raise: press's elbow
    angle sweeps a large range (rack to lockout), raise's stays roughly
    constant (arm stays mostly straight) — press's elbow variance should
    be much higher."""
    rng = np.random.default_rng(0)
    subject = _make_subject(rng)
    elbow_idx = FEATURE_NAMES.index("left_elbow_angle")

    press_elbows = [generate_example(rng, "shoulder_press", subject)[elbow_idx] for _ in range(500)]
    raise_elbows = [generate_example(rng, "lateral_raise", subject)[elbow_idx] for _ in range(500)]

    assert np.var(press_elbows) > np.var(raise_elbows) * 3


def test_unknown_exercise_raises():
    rng = np.random.default_rng(0)
    subject = _make_subject(rng)
    try:
        generate_example(rng, "not_a_real_exercise", subject)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
