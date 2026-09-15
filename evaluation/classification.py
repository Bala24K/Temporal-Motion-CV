"""
Classification metrics (V1 §18 / V2 §19) — accuracy, precision, recall,
F1, confusion matrix. Implemented directly rather than only calling
sklearn's equivalents, so these are independently testable against
hand-computed small examples: the ruler shouldn't share a bug with the
thing it's measuring.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np


@dataclass
class ClassificationReport:
    labels: List[str]
    confusion_matrix: np.ndarray   # rows = true label, columns = predicted label
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    macro_f1: float


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> np.ndarray:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def classification_report(y_true: Sequence[str], y_pred: Sequence[str],
                           labels: List[str]) -> ClassificationReport:
    cm = confusion_matrix(y_true, y_pred, labels)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    precision, recall, f1 = {}, {}, {}
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision[label] = p
        recall[label] = r
        f1[label] = f

    macro_f1 = float(np.mean(list(f1.values()))) if f1 else 0.0

    return ClassificationReport(
        labels=labels, confusion_matrix=cm, accuracy=accuracy,
        precision=precision, recall=recall, f1=f1, macro_f1=macro_f1,
    )
