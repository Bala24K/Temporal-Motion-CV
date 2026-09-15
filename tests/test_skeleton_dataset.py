import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from recognition.skeleton_dataset import generate_engineered_dataset, generate_sequence_dataset, RAW_PER_FRAME_NAMES  # noqa: E402
from recognition.skeleton_features import FEATURE_NAMES  # noqa: E402
from recognition.dataset_synthesis import EXERCISES  # noqa: E402


def test_engineered_dataset_shapes():
    X, y, subj = generate_engineered_dataset(n_subjects=3, windows_per_subject_per_exercise=4, n_frames=10, seed=0)
    n_expected = 3 * len(EXERCISES) * 4
    assert X.shape == (n_expected, len(FEATURE_NAMES))
    assert len(y) == n_expected
    assert set(y) == set(EXERCISES)
    assert not np.any(np.isnan(X))  # fill_missing already applied


def test_sequence_dataset_shapes():
    X, y, subj = generate_sequence_dataset(n_subjects=3, windows_per_subject_per_exercise=4, n_frames=10, seed=0)
    n_expected = 3 * len(EXERCISES) * 4
    assert X.shape == (n_expected, 10, len(RAW_PER_FRAME_NAMES))
    assert len(y) == n_expected
    assert not np.any(np.isnan(X))


def test_engineered_and_sequence_datasets_are_paired():
    """Same seed and iteration order -> example i in each dataset must
    correspond to the same underlying synthetic window: same label, same
    subject. This is what makes a model comparison between them fair."""
    X_eng, y_eng, subj_eng = generate_engineered_dataset(n_subjects=4, windows_per_subject_per_exercise=3,
                                                          n_frames=8, seed=7)
    X_seq, y_seq, subj_seq = generate_sequence_dataset(n_subjects=4, windows_per_subject_per_exercise=3,
                                                        n_frames=8, seed=7)
    assert np.array_equal(y_eng, y_seq)
    assert np.array_equal(subj_eng, subj_seq)


def test_reproducible_with_same_seed():
    X1, y1, s1 = generate_engineered_dataset(n_subjects=2, windows_per_subject_per_exercise=3, seed=5)
    X2, y2, s2 = generate_engineered_dataset(n_subjects=2, windows_per_subject_per_exercise=3, seed=5)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
