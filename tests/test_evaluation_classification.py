import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from evaluation.classification import confusion_matrix, classification_report  # noqa: E402


# Hand-computed example, worked out on paper before writing the assertions:
# y_true = [a, a, b, b], y_pred = [a, b, b, b]
# confusion matrix (rows=true, cols=pred), labels=[a,b]:
#   [[1, 1],
#    [0, 2]]
# accuracy = (1+2)/4 = 0.75
# precision(a) = TP/(TP+FP) = 1/(1+0) = 1.0      (column 'a' has only 1 total)
# recall(a)    = TP/(TP+FN) = 1/(1+1) = 0.5
# f1(a)        = 2*1.0*0.5/(1.0+0.5) = 0.6667
# precision(b) = TP/(TP+FP) = 2/(2+1) = 0.6667
# recall(b)    = TP/(TP+FN) = 2/(2+0) = 1.0
# f1(b)        = 2*0.6667*1.0/(0.6667+1.0) = 0.8
Y_TRUE = ["a", "a", "b", "b"]
Y_PRED = ["a", "b", "b", "b"]


def test_confusion_matrix_matches_hand_computation():
    cm = confusion_matrix(Y_TRUE, Y_PRED, labels=["a", "b"])
    assert np.array_equal(cm, np.array([[1, 1], [0, 2]]))


def test_accuracy_matches_hand_computation():
    report = classification_report(Y_TRUE, Y_PRED, labels=["a", "b"])
    assert abs(report.accuracy - 0.75) < 1e-9


def test_precision_recall_f1_match_hand_computation():
    report = classification_report(Y_TRUE, Y_PRED, labels=["a", "b"])
    assert abs(report.precision["a"] - 1.0) < 1e-9
    assert abs(report.recall["a"] - 0.5) < 1e-9
    assert abs(report.f1["a"] - (2 / 3)) < 1e-6

    assert abs(report.precision["b"] - (2 / 3)) < 1e-6
    assert abs(report.recall["b"] - 1.0) < 1e-9
    assert abs(report.f1["b"] - 0.8) < 1e-9


def test_perfect_predictions_give_accuracy_one():
    y = ["a", "b", "c", "a", "b"]
    report = classification_report(y, y, labels=["a", "b", "c"])
    assert report.accuracy == 1.0
    assert all(v == 1.0 for v in report.precision.values())
    assert all(v == 1.0 for v in report.recall.values())
    assert report.macro_f1 == 1.0


def test_completely_wrong_predictions_give_low_scores():
    y_true = ["a", "a", "a"]
    y_pred = ["b", "b", "b"]
    report = classification_report(y_true, y_pred, labels=["a", "b"])
    assert report.accuracy == 0.0
    assert report.precision["a"] == 0.0
    assert report.recall["a"] == 0.0


def test_mismatched_lengths_raise():
    try:
        confusion_matrix(["a", "b"], ["a"], labels=["a", "b"])
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
