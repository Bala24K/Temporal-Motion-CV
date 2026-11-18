"""
Confidence calibration.

The current pipeline displays `min(0.96, model_score)` and calls it
confidence. That is a display cap: it stops the UI showing 100%, but it
does not make the number mean anything. A calibrated 0.7 should mean the
model is right about 70% of the time when it says 0.7, and nothing in the
present system establishes that.

This module provides the real version — temperature scaling fitted on the
validation split, plus the metrics that show whether it worked. Pure
numpy, so it runs without torch and can be tested directly.

WHY TEMPERATURE SCALING
    One parameter, fitted on held-out data, cannot change the argmax and
    therefore cannot change accuracy — it only rescales confidence. With
    a validation split of a few dozen windows, anything with more
    parameters (isotonic regression, vector scaling) will overfit the
    calibration set and produce worse calibration on test than doing
    nothing. Isotonic becomes the better choice at roughly a thousand
    validation samples; `fit_isotonic` is here for that point, marked as
    such.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Mean NLL of the true class under temperature-scaled logits."""
    probs = softmax(logits / max(temperature, 1e-6))
    picked = probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.maximum(picked, 1e-12))))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: Tuple[float, float] = (0.05, 10.0),
    tolerance: float = 1e-4,
) -> float:
    """
    Fit the scalar temperature T minimising NLL on held-out logits.

    Golden-section search rather than gradient descent: the objective is
    one-dimensional and unimodal in T, the search needs no learning rate
    or step count, and it cannot diverge — which matters because this is
    run automatically at the end of every training job.

    T > 1 means the model was overconfident and probabilities are being
    softened; T < 1 means underconfident.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected logits of shape (N, C), got {logits.shape}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have the same first dimension")
    if logits.shape[0] < 2:
        return 1.0

    lo, hi = bounds
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = negative_log_likelihood(logits, labels, c), negative_log_likelihood(logits, labels, d)

    while (b - a) > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = negative_log_likelihood(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = negative_log_likelihood(logits, labels, d)

    return float((a + b) / 2.0)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    """
    ECE and the per-bin breakdown behind it.

    ECE is the average gap between confidence and accuracy, weighted by
    bin population. The bins are returned too, because a single ECE
    number hides the shape of the miscalibration — a model overconfident
    only in its top bin and one uniformly off by the same average are
    very different problems.

    Empty bins are omitted rather than counted as perfectly calibrated.
    """
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same first dimension")
    if probs.shape[0] == 0:
        return 0.0, []

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = probs.shape[0]
    ece = 0.0
    bins: List[Dict[str, float]] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the left edge on the first bin so confidence exactly 0
        # is not silently dropped.
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (n / total) * abs(bin_acc - bin_conf)
        bins.append({
            "bin_lower": float(lo), "bin_upper": float(hi), "count": n,
            "confidence": bin_conf, "accuracy": bin_acc, "gap": bin_acc - bin_conf,
        })

    return float(ece), bins


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Worst per-bin gap. The number to quote when a confident wrong
    answer is more costly than an average-sized error."""
    _, bins = expected_calibration_error(probs, labels, n_bins)
    return max((abs(b["gap"]) for b in bins), default=0.0)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multiclass Brier score: mean squared error between the predicted
    distribution and the one-hot truth. Lower is better; 0 is perfect.

    Unlike ECE it is a proper scoring rule — it cannot be gamed by a
    model that reports a constant confidence equal to its accuracy.
    """
    if probs.shape[0] == 0:
        return 0.0
    onehot = np.zeros_like(probs)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


@dataclass
class CalibrationReport:
    """Everything needed to judge whether the confidence number is honest."""
    temperature: float
    ece_before: float
    ece_after: float
    mce_after: float
    brier_before: float
    brier_after: float
    nll_before: float
    nll_after: float
    bins_after: List[Dict[str, float]]

    @property
    def improved(self) -> bool:
        return self.ece_after <= self.ece_before

    def to_dict(self) -> Dict:
        return {
            "temperature": self.temperature,
            "ece_before": self.ece_before, "ece_after": self.ece_after,
            "mce_after": self.mce_after,
            "brier_before": self.brier_before, "brier_after": self.brier_after,
            "nll_before": self.nll_before, "nll_after": self.nll_after,
            "improved": self.improved,
            "bins_after": self.bins_after,
        }

    def summary(self) -> str:
        direction = "overconfident" if self.temperature > 1.0 else "underconfident"
        return (
            f"temperature={self.temperature:.3f} (model was {direction})  "
            f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f}  "
            f"Brier {self.brier_before:.4f} -> {self.brier_after:.4f}"
        )


def calibrate(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> CalibrationReport:
    """
    Fit temperature on validation, report the effect on test.

    Fitting and reporting on the same split would report how well one
    free parameter can fit that split, which is not the question. When no
    test split is given the validation split is reported with that caveat
    implicit — useful during development, not for a final number.
    """
    temperature = fit_temperature(val_logits, val_labels)

    eval_logits = test_logits if test_logits is not None else val_logits
    eval_labels = test_labels if test_labels is not None else val_labels

    probs_before = softmax(eval_logits)
    probs_after = softmax(eval_logits / temperature)

    ece_before, _ = expected_calibration_error(probs_before, eval_labels, n_bins)
    ece_after, bins_after = expected_calibration_error(probs_after, eval_labels, n_bins)

    return CalibrationReport(
        temperature=temperature,
        ece_before=ece_before,
        ece_after=ece_after,
        mce_after=maximum_calibration_error(probs_after, eval_labels, n_bins),
        brier_before=brier_score(probs_before, eval_labels),
        brier_after=brier_score(probs_after, eval_labels),
        nll_before=negative_log_likelihood(eval_logits, eval_labels, 1.0),
        nll_after=negative_log_likelihood(eval_logits, eval_labels, temperature),
        bins_after=bins_after,
    )


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled probabilities. Inference-time counterpart of `calibrate`."""
    return softmax(logits / max(float(temperature), 1e-6))


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (confidence, accuracy, count) per non-empty bin, ready to plot against
    the y = x diagonal.
    """
    _, bins = expected_calibration_error(probs, labels, n_bins)
    if not bins:
        return np.array([]), np.array([]), np.array([])
    return (
        np.array([b["confidence"] for b in bins]),
        np.array([b["accuracy"] for b in bins]),
        np.array([b["count"] for b in bins]),
    )


def fit_isotonic(val_probs: np.ndarray, val_labels: np.ndarray):
    """
    Isotonic calibration of the top-class confidence.

    Non-parametric and strictly more flexible than temperature scaling,
    and correspondingly hungrier for data: below roughly a thousand
    validation samples it reliably overfits and makes calibration worse.
    Provided for when the corpus reaches that size; not wired into the
    training script.

    Returns a callable mapping raw confidence to calibrated confidence.
    """
    from sklearn.isotonic import IsotonicRegression

    confidence = val_probs.max(axis=1)
    correct = (val_probs.argmax(axis=1) == val_labels).astype(np.float64)
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(confidence, correct)
    return lambda p: model.predict(np.asarray(p).reshape(-1))
