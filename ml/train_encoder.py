"""Train the movement encoder on a dataset store.   python -m ml.train_encoder --store data/dataset --out models/encoder.onnx

Pipeline: frozen, subject-disjoint split -> SupCon+CE training with skeleton augmentation -> embeddings ->
prototypes + Mahalanobis OOD + conformal (calibrated on the val split) -> test metrics -> ONNX export with a parity
check -> run log. `--synthetic` builds a throwaway synthetic store to smoke-test the plumbing (NOT a result).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ml.data.report import report
from ml.data.schema import ClipMeta, TRUSTED
from ml.data.split import subject_disjoint_split
from ml.data.store import DatasetStore
from ml.seq import augment, channels, clip_positions, windows
from ml.tracking import Run
from motion_engine.uncertainty import ConformalClassifier, MahalanobisOOD, PrototypeClassifier, softmax


def synthetic_store(path: str, subjects: int = 8) -> DatasetStore:
    from motion_engine.synthetic import EX, synth_sequence
    st, k = DatasetStore(path), 0
    for s in range(subjects):
        for ex in sorted(EX):
            fr, ts = synth_sequence(ex, reps=3, seed=k, scale=0.7 + 0.06 * s)
            st.add_clip(fr, ts, ClipMeta(clip_id=f"s{k}", subject_id=f"sub{s}", exercise_id=ex, view="front", label_source="rule", pose_model="synthetic"))
            k += 1
    return st


def build(store: DatasetStore, ids, classes, window):
    metas = store.meta_index()
    X, y = [], []
    for cid in ids:
        m = metas[cid]
        if m.exercise_id not in classes:
            continue
        frames, _ = store.load_clip(cid)
        X.append(clip_positions(frames))
        y.append(classes.index(m.exercise_id))
    return X, np.array(y)


def embed(model, clips, window, device):
    import torch
    out, lab = [], []
    model.eval()
    with torch.no_grad():
        for i, xy in enumerate(clips):
            ws = windows(xy, window, window // 2)
            if ws:
                z = model(torch.tensor(np.stack([channels(w) for w in ws]), device=device))
                out.append(z.cpu().numpy())
                lab += [i] * len(ws)
    return np.concatenate(out), np.array(lab)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/dataset"); ap.add_argument("--out", default="models/encoder.onnx")
    ap.add_argument("--config", default="train.yaml"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--allow-untrusted", action="store_true")
    a = ap.parse_args(argv)
    import torch
    from ml.torch_models import Classifier, export_onnx, supcon_loss
    cfg = yaml.safe_load(open(a.config))
    if a.epochs:
        cfg["epochs"] = a.epochs
    torch.manual_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"])
    store = synthetic_store(a.store + "_synthetic") if a.synthetic else DatasetStore(a.store)
    metas = store.effective()
    rep = report(metas)
    if not a.allow_untrusted and rep["trusted_fraction"] < 1.0:
        raise SystemExit(f"refusing to train: only {rep['trusted_fraction']:.0%} of labels are human-verified "
                         f"(use --allow-untrusted for experiments; the registry gate will still block promotion)")
    classes = sorted({m.exercise_id for m in metas if m.exercise_id})
    split = subject_disjoint_split([m for m in metas if m.exercise_id], seed=cfg["split_seed"])
    W = cfg["window"]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (build(store, split[s], classes, W) for s in ("train", "val", "test"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier(len(classes), embed_dim=cfg["embed_dim"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    run = Run("encoder", cfg, dataset_version=store.version_id())
    for ep in range(cfg["epochs"]):
        model.train()
        idx = rng.permutation(len(Xtr))
        tot = 0.0
        for b in range(0, len(idx), cfg["batch_size"]):
            sel = idx[b:b + cfg["batch_size"]]
            if len(sel) < 2:
                continue
            xb = torch.tensor(np.stack([channels(augment(Xtr[i], W, rng)) for i in sel]), device=dev)
            yb = torch.tensor(ytr[sel], device=dev)
            z, logits = model(xb)
            loss = supcon_loss(z, yb, cfg["temperature"]) + torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step(); run.log(ep, loss=tot)

    # embeddings -> prototypes / OOD / conformal (calibrated on val, evaluated on test)
    Ztr, ltr = embed(model.enc, Xtr, W, dev); Zva, lva = embed(model.enc, Xva, W, dev); Zte, lte = embed(model.enc, Xte, W, dev)
    ytr_w, yva_w, yte_w = ytr[ltr], yva[lva], yte[lte]
    pc = PrototypeClassifier.fit(Ztr, [str(i) for i in ytr_w])
    order = [int(l) for l in pc.labels]; remap = {l: i for i, l in enumerate(order)}
    P = lambda Z: softmax(pc.logits(Z, temperature=0.05))
    conf = ConformalClassifier.fit(P(Zva), np.array([remap[v] for v in yva_w]), cfg.get("alpha", 0.1))
    ood = MahalanobisOOD.fit(Ztr, ytr_w).calibrate(Zva)
    pred = np.array([order[i] for i in P(Zte).argmax(1)])
    from sklearn.metrics import accuracy_score, f1_score
    sets = conf.predict_set(P(Zte))
    metrics = {"subject_disjoint_accuracy": float(accuracy_score(yte_w, pred)), "macro_f1": float(f1_score(yte_w, pred, average="macro")),
               "conformal_coverage": float(np.mean([remap[y] in s for s, y in zip(sets, yte_w)])),
               "id_ood_false_flag_rate": float(ood.is_ood(Zte).mean()), "n_train_clips": len(Xtr), "n_test_clips": len(Xte),
               "label_source": "synthetic" if a.synthetic else ("mixed_verified" if rep["trusted_fraction"] == 1.0 else "rule")}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    parity = export_onnx(model.enc.cpu(), str(out), W)
    out.with_suffix(".json").write_text(json.dumps({
        "classes": classes, "prototypes": pc.protos.tolist(), "proto_labels": order, "proto_scale": pc.scale.tolist(),
        "ood": {"means": ood.means.tolist(), "prec": ood.prec.tolist(), "threshold": ood.threshold},
        "conformal": {"qhat": conf.qhat, "alpha": conf.alpha}, "window": W, "dataset_version": store.version_id(),
        "feature_version": 2}))
    metrics["onnx_parity_max_err"] = parity
    run.finish(metrics, {"onnx": str(out)})
    Path("reports").mkdir(exist_ok=True); Path("reports/encoder_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
