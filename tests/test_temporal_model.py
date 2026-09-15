import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from recognition.temporal_model import TemporalCNNClassifier  # noqa: E402
from recognition.skeleton_dataset import generate_sequence_dataset  # noqa: E402
from recognition.dataset_synthesis import EXERCISES  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402


def test_fit_predict_returns_valid_labels():
    X, y, subj = generate_sequence_dataset(n_subjects=6, windows_per_subject_per_exercise=8,
                                            n_frames=12, seed=0)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups=subj))

    clf = TemporalCNNClassifier(epochs=5, seed=0)  # few epochs, just checking the wiring
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])

    assert len(preds) == len(test_idx)
    assert all(p in EXERCISES for p in preds)


def test_predict_proba_sums_to_one_and_matches_predict():
    X, y, subj = generate_sequence_dataset(n_subjects=6, windows_per_subject_per_exercise=8,
                                            n_frames=12, seed=1)
    clf = TemporalCNNClassifier(epochs=5, seed=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, len(set(y)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    preds = clf.predict(X[:5])
    proba_preds = np.array(clf.classes_)[np.argmax(proba, axis=1)]
    assert np.array_equal(preds, proba_preds)


def test_achieves_reasonable_accuracy_with_enough_training():
    """Not a tight bound — just confirms the model can actually learn
    something from this synthetic, geometrically-separable-by-
    construction data, given enough epochs. A real accuracy claim needs
    real data; see STATUS.md."""
    X, y, subj = generate_sequence_dataset(n_subjects=20, windows_per_subject_per_exercise=15,
                                            n_frames=16, seed=0)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups=subj))

    clf = TemporalCNNClassifier(epochs=25, seed=0)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    accuracy = np.mean(preds == y[test_idx])
    assert accuracy > 0.5, f"expected the model to learn something, got accuracy={accuracy}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
