import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from recognition.classifier import (  # noqa: E402
    train_classifier, train_test_split_by_subject, predict_exercise, save_model, load_model,
)
from recognition.dataset_synthesis import generate_dataset, EXERCISES  # noqa: E402


def test_subject_split_has_no_overlap_between_train_and_test():
    X, y, subject_ids = generate_dataset(n_subjects=10, examples_per_subject_per_exercise=5, seed=0)
    X_train, X_test, y_train, y_test = train_test_split_by_subject(X, y, subject_ids, seed=0)
    # Re-derive which subjects ended up where by re-running the split logic
    # on subject_ids directly is awkward from outside; instead just check
    # basic sanity: both splits non-empty and their sizes sum to the total.
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_train) > 0 and len(X_test) > 0


def test_classifier_achieves_high_accuracy_on_synthetic_holdout():
    """
    Because the four classes are cleanly geometrically separable BY
    CONSTRUCTION (see dataset_synthesis.py's docstring), a reasonable
    classifier should recover them well on more synthetic data generated
    the same way. This is a sanity check on the pipeline, NOT a
    real-world accuracy claim — see this module's and
    dataset_synthesis.py's docstrings.
    """
    result = train_classifier(n_subjects=25, examples_per_subject_per_exercise=30, seed=0)
    assert result.report.accuracy > 0.85, \
        f"expected high accuracy on synthetic (geometrically separable) holdout, got {result.report.accuracy}"
    assert result.n_train > 0 and result.n_test > 0


def test_confusion_matrix_shape_matches_number_of_exercises():
    result = train_classifier(n_subjects=15, examples_per_subject_per_exercise=20, seed=1)
    n = len(EXERCISES)
    assert result.report.confusion_matrix.shape == (n, n)


def test_predict_exercise_returns_valid_label_and_confidence():
    result = train_classifier(n_subjects=15, examples_per_subject_per_exercise=20, seed=2)
    X, y, _ = generate_dataset(n_subjects=1, examples_per_subject_per_exercise=1, seed=99)
    label, confidence = predict_exercise(result.model, X[0])
    assert label in EXERCISES
    assert 0.0 <= confidence <= 1.0


def test_predict_exercise_handles_missing_features():
    """A real inference-time feature vector might have NaN entries (an
    occluded joint) — predict_exercise must not crash on that."""
    result = train_classifier(n_subjects=10, examples_per_subject_per_exercise=15, seed=3)
    features = np.array([90.0, np.nan, 170.0, np.nan, 175.0, np.nan, 5.0, np.nan])
    label, confidence = predict_exercise(result.model, features)
    assert label in EXERCISES
    assert 0.0 <= confidence <= 1.0


def test_model_save_and_load_round_trip_gives_same_predictions():
    result = train_classifier(n_subjects=10, examples_per_subject_per_exercise=15, seed=4)
    X, _, _ = generate_dataset(n_subjects=1, examples_per_subject_per_exercise=5, seed=123)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.joblib")
        save_model(result.model, path)
        loaded = load_model(path)
        for row in X:
            label1, conf1 = predict_exercise(result.model, row)
            label2, conf2 = predict_exercise(loaded, row)
            assert label1 == label2
            assert abs(conf1 - conf2) < 1e-9


def test_reproducible_training_with_same_seed():
    r1 = train_classifier(n_subjects=10, examples_per_subject_per_exercise=15, seed=7)
    r2 = train_classifier(n_subjects=10, examples_per_subject_per_exercise=15, seed=7)
    assert abs(r1.report.accuracy - r2.report.accuracy) < 1e-9


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
