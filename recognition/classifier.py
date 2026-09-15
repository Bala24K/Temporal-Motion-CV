"""
Exercise classifier — the project's first trained ML model (V1 §15).

Distinguishes which of the four exercises (bicep_curl, squat,
shoulder_press, lateral_raise) is being performed, from a single frame's
8-feature geometric vector (see recognition/features.py).

READ recognition/dataset_synthesis.py's module docstring before trusting
the accuracy number this produces. Short version: training AND evaluation
data here are synthetic-by-construction from the same geometric
definitions the rule-based analyzers use. The reported accuracy measures
recovery of those definitions on held-out synthetic examples — not
real-world accuracy on an actual camera feed. See STATUS.md.

Train/test split is done BY SUBJECT (sklearn's GroupShuffleSplit), not by
individual example, so the classifier is evaluated on synthetic
"subjects" it never saw during training — the synthetic-data analogue of
V1's own "split by person, not by rep" warning.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from recognition.dataset_synthesis import generate_dataset, EXERCISES
from recognition.features import fill_missing
from evaluation.classification import classification_report, ClassificationReport


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    report: ClassificationReport
    n_train: int
    n_test: int


def train_test_split_by_subject(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                                 test_size: float = 0.25, seed: int = 0):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=subject_ids))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_classifier(n_subjects: int = 30, examples_per_subject_per_exercise: int = 40,
                      seed: int = 0, n_estimators: int = 200, max_depth: Optional[int] = 8) -> TrainingResult:
    X, y, subject_ids = generate_dataset(n_subjects, examples_per_subject_per_exercise, seed=seed)
    X = np.array([fill_missing(row) for row in X])  # no NaNs expected here (synthetic data is complete), but keep the pipeline consistent with real inference input

    X_train, X_test, y_train, y_test = train_test_split_by_subject(X, y, subject_ids, seed=seed)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, labels=EXERCISES)

    return TrainingResult(model=model, report=report, n_train=len(X_train), n_test=len(X_test))


def predict_exercise(model: RandomForestClassifier, features: np.ndarray) -> Tuple[str, float]:
    """Returns (predicted_label, confidence) where confidence is the
    fraction of the forest's trees that voted for the predicted class —
    a reasonable, standard proxy for classifier confidence, but (like
    everything else in this project) not a calibrated probability without
    real-world validation."""
    filled = fill_missing(features).reshape(1, -1)
    proba = model.predict_proba(filled)[0]
    idx = int(np.argmax(proba))
    return model.classes_[idx], float(proba[idx])


def save_model(model: RandomForestClassifier, path: str):
    import joblib
    joblib.dump(model, path)


def load_model(path: str) -> RandomForestClassifier:
    import joblib
    return joblib.load(path)
