"""ONNX Runtime recognizer: a learned model behind the same contract as `recognize()`.

Model I/O: input `x` float32 [N, F] (see `window_features`), output `logits` [N, K].
Sidecar `<model>.json`: groups, temperature, a Mahalanobis OOD detector and a split-conformal
threshold, all calibrated on held-out data at export time. Decision rule (motion_engine/uncertainty.py):
far from training data -> "unsupported"; conformal set size 1 -> analyse; >1 -> ask; 0 -> unsupported.
Softmax confidence alone is never used: it cannot detect extrapolation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .embedding import CHANNELS, FULL_SCALE, NAMES, _channel_series
from .recognition import Candidate, Recognition, MIN_MOVEMENT
from .uncertainty import ConformalClassifier, MahalanobisOOD, softmax as _sm
from .skeleton import fill_gaps, frames_to_array, normalize

FEATURE_VERSION = 1
STATS = ("range", "std", "median", "min", "max")


def feature_names():
    return [f"{c}.{s}" for c in NAMES for s in STATS]


def window_features(frames: Sequence) -> np.ndarray:
    """Phase-invariant per-channel statistics (so a window may start mid-rep). Shape (7*5,)."""
    arr = normalize(fill_gaps(frames_to_array(frames)))
    ch = _channel_series(arr)
    f = []
    for n in NAMES:
        x = ch[n][np.isfinite(ch[n])] / FULL_SCALE[n]
        x = x if len(x) else np.zeros(1)
        f += [np.ptp(x), np.std(x), np.median(x), np.min(x), np.max(x)]
    return np.asarray(f, np.float32)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class LearnedRecognizer:
    def __init__(self, onnx_path: str | Path, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        p = Path(onnx_path)
        self.meta = json.loads(p.with_suffix(".json").read_text())
        if self.meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError("model/feature version mismatch - re-export the model")
        self.sess = ort.InferenceSession(str(p), providers=list(providers))
        self.groups = [list(g) for g in self.meta["groups"]]
        o = self.meta["ood"]
        self.ood = MahalanobisOOD(np.asarray(o["means"]), np.asarray(o["prec"]), o["threshold"])
        self.conformal = ConformalClassifier(self.meta["conformal"]["qhat"], self.meta["conformal"]["alpha"])

    def logits(self, frames) -> np.ndarray:
        x = window_features(frames)[None]
        return self.sess.run(["logits"], {"x": x})[0][0]

    def recognize(self, frames, min_frames: int = 30, **_) -> Recognition:
        if len(frames) < min_frames:
            return Recognition("no_movement", "wait", 0.0, None, message="Collecting movement...")
        x = window_features(frames)
        names = feature_names()
        if max(x[names.index(f"{c}.range")] for c in NAMES[:6]) < MIN_MOVEMENT:
            return Recognition("no_movement", "wait", 0.0, None, message="Waiting for you to start moving.")
        z = self.sess.run(["logits"], {"x": x[None]})[0][0]
        p = _sm(z / self.meta["temperature"])
        order = np.argsort(-p)
        cands = [Candidate(self.groups[i][0], self.groups[i], float(z[i]), float(p[i]), len(self.groups[i]) == 1)
                 for i in order[:3]]
        if bool(self.ood.is_ood(x[None])[0]):
            return Recognition("unsupported", "fallback", 0.0, None, cands,
                               "This movement isn't fully supported yet. You can still track it manually.")
        s = self.conformal.predict_set(p)[0]
        if not s:
            return Recognition("unsupported", "fallback", 0.0, None, cands,
                               "This movement isn't fully supported yet. You can still track it manually.")
        top = cands[0]
        if len(s) > 1:
            return Recognition("ambiguous", "confirm", top.probability, top, cands, "Not sure which exercise this is - please confirm.")
        if not top.distinguishable:
            return Recognition("ambiguous", "confirm", top.probability, top, cands,
                               "These exercises look identical to the camera - tell us which one you're doing.")
        return Recognition("recognized", "analyze", top.probability, top, cands)
