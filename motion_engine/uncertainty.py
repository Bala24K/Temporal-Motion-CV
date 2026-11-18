"""Uncertainty toolkit (numpy only): class-conditional Mahalanobis OOD, split-conformal prediction
sets, and a prototype (few-shot) classifier. Works on any feature/embedding vectors, so the same
code serves the engineered features today and a learned encoder's embeddings later.

Why not just softmax: a softmax/max-logit score cannot detect extrapolation (docs/README: a Y-raise was
scored *higher* than real raises). Distance-to-training-distribution and conformal coverage are
calibrated on held-out data and give an explicit, auditable abstention rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties split. pos = OOD, neg = in-distribution."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


@dataclass
class MahalanobisOOD:
    """Min over classes of (x-mu_c)' S^-1 (x-mu_c), with a shared shrunk covariance."""
    means: np.ndarray
    prec: np.ndarray
    threshold: float = np.inf

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, shrinkage: float = 0.1) -> "MahalanobisOOD":
        classes = np.unique(y)
        means = np.stack([X[y == c].mean(0) for c in classes])
        R = X - means[np.searchsorted(classes, y)]
        cov = R.T @ R / max(1, len(X) - len(classes))
        cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        return cls(means, np.linalg.pinv(cov))

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        d = X[:, None, :] - self.means[None]
        return np.einsum("nki,ij,nkj->nk", d, self.prec, d).min(1)

    def calibrate(self, X_val: np.ndarray, id_retention: float = 0.99, margin: float = 1.25) -> "MahalanobisOOD":
        """Threshold = margin x the `id_retention` quantile of held-out in-distribution scores."""
        self.threshold = float(np.quantile(self.score(X_val), id_retention) * margin)
        return self

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold


@dataclass
class PrototypeClassifier:
    """Nearest-prototype classifier: a new class needs only a handful of examples (few-shot)."""
    labels: List[str]
    protos: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, labels: Sequence[str]) -> "PrototypeClassifier":
        labels = np.asarray(labels)
        names = sorted(set(labels))
        protos = np.stack([X[labels == n].mean(0) for n in names])
        return cls(names, protos, X.std(0) + 1e-6)

    def add_class(self, name: str, X: np.ndarray) -> "PrototypeClassifier":
        """Few-shot extension: no retraining, just one new prototype."""
        names = self.labels + [name]
        return PrototypeClassifier(names, np.vstack([self.protos, X.mean(0)]), self.scale)

    def logits(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        X = np.atleast_2d(X)
        d = (((X[:, None, :] - self.protos[None]) / self.scale) ** 2).sum(-1)
        return -d / temperature

    def predict(self, X: np.ndarray) -> List[str]:
        return [self.labels[i] for i in self.logits(X).argmax(1)]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


@dataclass
class ConformalClassifier:
    """Split-conformal prediction sets with finite-sample coverage >= 1-alpha (exchangeable data).
    Score = 1 - p(true class). A set of size 1 is a confident answer; size>1 = ambiguous; empty = reject."""
    qhat: float
    alpha: float

    @classmethod
    def fit(cls, probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> "ConformalClassifier":
        n = len(y_cal)
        s = 1.0 - probs_cal[np.arange(n), y_cal]
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return cls(float(np.quantile(s, level, method="higher")), alpha)

    def predict_set(self, probs: np.ndarray) -> List[List[int]]:
        probs = np.atleast_2d(probs)
        return [list(np.flatnonzero(1.0 - p <= self.qhat)) for p in probs]


@dataclass
class Decision:
    action: str            # analyze | confirm | fallback
    candidates: List[int]
    reason: str


def decide(prob: np.ndarray, conf: ConformalClassifier, ood: Optional[MahalanobisOOD], x: np.ndarray) -> Decision:
    """The abstention rule used in production: OOD first, then the conformal set size."""
    if ood is not None and bool(ood.is_ood(x)[0]):
        return Decision("fallback", [], "input is far from everything seen in training")
    s = conf.predict_set(prob)[0]
    if len(s) == 1:
        return Decision("analyze", s, "single-class conformal set")
    if len(s) == 0:
        return Decision("fallback", [], "no class is plausible at the target coverage")
    return Decision("confirm", s, f"{len(s)} classes remain plausible")
