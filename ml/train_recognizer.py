"""Train + calibrate + export the ONNX recognizer.   python -m ml.train_recognizer out/recognizer.onnx

HONEST SCOPE: trains on *synthetic* clips (scale/noise/speed/depth/view-jitter augmentation, subject-disjoint
by construction: val/test use unseen body scales and seeds). That validates the pipeline and the abstention
logic. It says nothing about real-world accuracy; replace `make_clips` with real, human-labelled,
subject-disjoint data before promoting anything (the registry gate enforces this).
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from exercise_system.registry import ExerciseRegistry
from motion_engine.learned import FEATURE_VERSION, window_features
from motion_engine.synthetic import EX, synth_sequence
from motion_engine.uncertainty import ConformalClassifier, MahalanobisOOD, softmax


def groups_for(registry):
    g = {}
    for d in registry.recognisable():
        if d.id in EX:
            g.setdefault(d.equivalence_key(), []).append(d.id)
    return [sorted(v) for v in g.values()]


def make_clips(groups, seeds, scales, rng, win=90):
    X, y = [], []
    for gi, ids in enumerate(groups):
        for ex in ids:
            for seed in seeds:
                for sc in scales:
                    frames, _ = synth_sequence(ex, reps=4, seed=seed, scale=sc, noise=float(rng.uniform(0.001, 0.006)),
                                               rep_seconds=float(rng.uniform(1.8, 3.6)), depth=float(rng.uniform(0.85, 1.15)))
                    for _ in range(2):
                        s = int(rng.integers(0, max(1, len(frames) - win)))
                        X.append(window_features(frames[s:s + win]))
                        y.append(gi)
    return np.array(X, np.float32), np.array(y)


def train_mlp(X, y, K, hidden=32, epochs=400, lr=0.05, wd=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    W1, b1 = rng.normal(0, 0.3, (X.shape[1], hidden)), np.zeros(hidden)
    W2, b2 = rng.normal(0, 0.3, (hidden, K)), np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(epochs):
        H = np.maximum(Z @ W1 + b1, 0)
        P = np.exp(H @ W2 + b2 - (H @ W2 + b2).max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(Z)
        gW2, gb2 = H.T @ G + wd * W2, G.sum(0)
        GH = (G @ W2.T) * (H > 0)
        gW1, gb1 = Z.T @ GH + wd * W1, GH.sum(0)
        W1 -= lr * 20 * gW1; b1 -= lr * 20 * gb1; W2 -= lr * 20 * gW2; b2 -= lr * 20 * gb2
    return mu, sd, W1, b1, W2, b2


def forward(params, X):
    mu, sd, W1, b1, W2, b2 = params
    return np.maximum((X - mu) / sd @ W1 + b1, 0) @ W2 + b2


def fit_temperature(z, y):
    best, bt = 1e9, 1.0
    for t in np.linspace(0.5, 10, 96):
        p = np.exp(z / t - (z / t).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
        if nll < best:
            best, bt = nll, t
    return float(bt)


def export_onnx(params, path):
    mu, sd, W1, b1, W2, b2 = [p.astype(np.float32) for p in params]
    inits = [numpy_helper.from_array(a, n) for a, n in
             ((mu, "mu"), (sd, "sd"), (W1, "W1"), (b1, "b1"), (W2, "W2"), (b2, "b2"))]
    nodes = [helper.make_node("Sub", ["x", "mu"], ["c"]), helper.make_node("Div", ["c", "sd"], ["z"]),
             helper.make_node("Gemm", ["z", "W1", "b1"], ["h0"]), helper.make_node("Relu", ["h0"], ["h"]),
             helper.make_node("Gemm", ["h", "W2", "b2"], ["logits"])]
    g = helper.make_graph(nodes, "recognizer",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", mu.shape[0]])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", W2.shape[1]])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def main(out="out/recognizer.onnx", fast=False):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    reg = ExerciseRegistry.load()
    groups = groups_for(reg)
    rng = np.random.default_rng(0)
    Xtr, ytr = make_clips(groups, seeds=range(0, 2 if fast else 4), scales=(0.9, 1.1) if fast else (0.8, 0.95, 1.1), rng=rng)
    Xva, yva = make_clips(groups, seeds=range(100, 102), scales=(0.7, 1.0, 1.25), rng=rng)       # unseen bodies
    Xte, yte = make_clips(groups, seeds=range(200, 202), scales=(0.75, 0.9, 1.2), rng=rng)
    params = train_mlp(Xtr, ytr, len(groups), epochs=150 if fast else 400)
    T = fit_temperature(forward(params, Xva), yva)
    zva = forward(params, Xva)
    ood = float(np.percentile(zva.max(1), 2) - 0.5)            # floor just under in-distribution max-logit
    acc = float((forward(params, Xte).argmax(1) == yte).mean())
    export_onnx(params, out)
    Xcal, ycal = make_clips(groups, seeds=range(300, 302), scales=(0.8, 1.0, 1.2), rng=rng)    # conformal set
    ood = MahalanobisOOD.fit(Xtr, ytr).calibrate(Xva)
    P = lambda X: softmax(forward(params, X) / T)
    conf = ConformalClassifier.fit(P(Xcal), ycal, alpha=0.1)
    sets = conf.predict_set(P(Xte))
    acc = float((forward(params, Xte).argmax(1) == yte).mean())
    export_onnx(params, out)
    meta = {"groups": groups, "temperature": T, "feature_version": FEATURE_VERSION,
            "ood": {"means": ood.means.tolist(), "prec": ood.prec.tolist(), "threshold": ood.threshold},
            "conformal": {"qhat": conf.qhat, "alpha": conf.alpha},
            "trained_on": "synthetic", "test_accuracy_synthetic": acc,
            "conformal_coverage_synthetic": float(np.mean([y in s_ for s_, y in zip(sets, yte)])),
            "id_ood_false_flag_rate": float(ood.is_ood(Xte).mean()), "n_train": int(len(Xtr))}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("groups", "ood")}), "groups:", len(groups))
    return meta


if __name__ == "__main__":
    main(*sys.argv[1:])
