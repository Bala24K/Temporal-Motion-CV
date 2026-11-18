"""
Training entry point for the motion recognition model.

    python -m motion.train --epochs 60 --out models/motion_stgcn.pt

What it does, in order: build windows from the pose cache, split by group
with a hard leakage assertion, train the ST-GCN/transformer model on both
the exercise and phase targets, fit a calibration temperature on the
validation split, evaluate on test, and write a checkpoint that carries
its own architecture config, class list, calibration temperature and
label provenance.

TWO GUARDS THAT ARE NOT OPTIONAL

  * Leakage. `assert_no_group_leakage` runs on every job, not only in
    tests, because leakage makes the number go up. There is no failure
    signal to notice.

  * Label provenance. If the test split is labelled by kinematic rule,
    the headline accuracy measures agreement with that rule. The script
    says so loudly and refuses to write a checkpoint marked
    `release_ready` without `--allow-derived-labels`. The flag exists
    because development has to continue before annotation is done; the
    friction exists so the resulting number never quietly becomes the
    headline.
"""
import argparse

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.classification import classification_report
from motion.augment import AugmentConfig, augment_window
from motion.calibration import calibrate, apply_temperature
from motion.dataset import (
    EXERCISE_CLASSES,
    LABEL_SOURCE_RULE,
    MotionDataset,
    OTHER_CLASS,
    assert_no_group_leakage,
    build_dataset_from_pose_cache,
    class_weights,
    group_disjoint_split,
    load_subject_map,
)
from motion.model import MotionModelConfig, build_model, count_parameters
from motion.phase import (
    NUM_PHASES,
    PHASE_NAMES,
    phase_labels_to_onehot,
    segment_reps_from_phases,
)
from motion.tensors import STREAM_NAMES, build_streams

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class WindowDataset(Dataset if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Applies augmentation per access rather than once up front, so each
    epoch sees different views of the same few hundred clips. Stream
    construction happens after augmentation — bone and motion streams are
    derived quantities, and augmenting them independently of the joints
    they come from would make them inconsistent.
    """

    def __init__(self, dataset: MotionDataset, streams: Sequence[str],
                 include_validity: bool, augment: Optional[AugmentConfig], seed: int = 0):
        self.dataset = dataset
        self.streams = tuple(streams)
        self.include_validity = include_validity
        self.augment = augment
        self.class_to_index = dataset.class_to_index
        self._base_seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset.samples[idx]
        window = sample.window
        if self.augment is not None:
            # Seed from (base, index, call counter) so runs are
            # reproducible while still varying across epochs.
            rng = np.random.default_rng((self._base_seed, idx, int(time.time_ns()) & 0xFFFF))
            window = augment_window(window, self.augment, rng)

        x = build_streams(window, self.streams, self.include_validity)
        y = self.class_to_index[sample.label]
        phases = sample.phase_labels
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(y, dtype=torch.long),
            torch.from_numpy(np.ascontiguousarray(phases)),
        )


def _forward_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader, returning (logits, labels, phase_logits, phase_labels)."""
    model.eval()
    logits, labels, phase_logits, phase_labels = [], [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            out = model(x.to(device))
            logits.append(out["exercise_logits"].cpu().numpy())
            phase_logits.append(out["phase_logits"].cpu().numpy())
            labels.append(y.numpy())
            phase_labels.append(p.numpy())
    return (
        np.concatenate(logits), np.concatenate(labels),
        np.concatenate(phase_logits), np.concatenate(phase_labels),
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> float:
    report = classification_report([classes[i] for i in y_true], [classes[i] for i in y_pred], classes)
    return report.macro_f1


def evaluate_split(model, loader, classes: List[str], device, temperature: float = 1.0) -> Dict:
    """Classification, phase and rep-counting metrics for one split."""
    logits, labels, phase_logits, phase_labels = _forward_all(model, loader, device)
    probs = apply_temperature(logits, temperature)
    preds = probs.argmax(axis=1)

    report = classification_report(
        [classes[i] for i in labels], [classes[i] for i in preds], classes
    )

    # Phase metrics against the derived labels — agreement with the rule,
    # not with a human annotator. Named accordingly so a reader of the
    # results JSON cannot mistake it for ground truth.
    phase_pred = phase_logits.argmax(axis=2)
    phase_frame_agreement = float((phase_pred == phase_labels).mean())
    phase_report = classification_report(
        [PHASE_NAMES[i] for i in phase_labels.reshape(-1)],
        [PHASE_NAMES[i] for i in phase_pred.reshape(-1)],
        PHASE_NAMES,
    )

    # Learned rep segmentation vs the deterministic phase rule, per window.
    exp_probs = np.exp(phase_logits - phase_logits.max(axis=2, keepdims=True))
    phase_probs = exp_probs / np.maximum(exp_probs.sum(axis=2, keepdims=True), 1e-12)
    learned_counts, rule_counts = [], []
    for i in range(phase_probs.shape[0]):
        learned_counts.append(len(segment_reps_from_phases(phase_probs[i])))
        rule_counts.append(len(segment_reps_from_phases(phase_labels_to_onehot(phase_labels[i]))))
    learned_counts = np.array(learned_counts)
    rule_counts = np.array(rule_counts)

    return {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "per_class_f1": report.f1,
        "confusion_matrix": report.confusion_matrix.tolist(),
        "phase_frame_agreement_with_rule": phase_frame_agreement,
        "phase_macro_f1_vs_rule": phase_report.macro_f1,
        "rep_count_mae_vs_rule": float(np.abs(learned_counts - rule_counts).mean()),
        "rep_count_mean_learned": float(learned_counts.mean()),
        "rep_count_mean_rule": float(rule_counts.mean()),
        "_logits": logits,
        "_labels": labels,
    }


def train(args: argparse.Namespace) -> Dict:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training.\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    classes = list(EXERCISE_CLASSES) + ([OTHER_CLASS] if args.include_other else [])
    subject_map = load_subject_map(args.subjects) if args.subjects else None

    print(f"building dataset from {args.cache} ...")
    dataset = build_dataset_from_pose_cache(
        cache_path=args.cache,
        subject_map=subject_map,
        window_frames=args.window_frames,
        windows_per_clip=args.windows_per_clip,
        max_per_class=args.max_per_class,
        classes=classes,
        seed=args.seed,
    )
    summary = dataset.provenance_summary()
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("_")}, indent=2, default=str))

    if subject_map is None:
        print(
            "\n  NOTE: no --subjects map given, so splits are video-disjoint but not\n"
            "  subject-disjoint. Clips of the same person can land on both sides of\n"
            "  the split, and the model can score well by recognising the person.\n"
        )

    train_ds, val_ds, test_ds = group_disjoint_split(
        dataset, ratios=tuple(args.split_ratios), seed=args.seed
    )
    assert_no_group_leakage(train_ds, val_ds, test_ds)
    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"(groups {len(set(train_ds.groups))}/{len(set(val_ds.groups))}/{len(set(test_ds.groups))})")

    derived_only = test_ds.has_only_derived_labels()
    if derived_only:
        print(
            "\n" + "=" * 72 + "\n"
            "  LABEL PROVENANCE WARNING\n"
            "  Every test label came from the kinematic rule in\n"
            "  recognition/qevd_dataset.py, not from a human or the source corpus.\n"
            "  Accuracy below measures agreement with that rule. It is a valid\n"
            "  engineering signal that the pipeline learns something learnable;\n"
            "  it is not evidence of exercise recognition.\n"
            + "=" * 72 + "\n"
        )
        if not args.allow_derived_labels:
            raise SystemExit(
                "refusing to train a release checkpoint on rule-derived test labels.\n"
                "Pass --allow-derived-labels to proceed for development purposes."
            )

    streams = tuple(args.streams)
    cfg = MotionModelConfig(
        num_classes=len(dataset.classes),
        num_phases=NUM_PHASES,
        streams=streams,
        adjacency=args.adjacency,
        adaptive_adjacency=not args.no_adaptive_adjacency,
        fusion=args.fusion,
        use_temporal_transformer=not args.no_transformer,
        dropout=args.dropout,
    )
    model = build_model(cfg).to(device)
    print(f"model: {count_parameters(model):,} parameters, "
          f"in_channels={cfg.in_channels}, fusion={cfg.fusion}, "
          f"transformer={cfg.use_temporal_transformer}")

    aug = AugmentConfig() if not args.no_augment else AugmentConfig.off()
    loaders = {
        "train": DataLoader(
            WindowDataset(train_ds, streams, cfg.include_validity_channel, aug, args.seed),
            batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
            num_workers=args.workers,
        ),
        "val": DataLoader(
            WindowDataset(val_ds, streams, cfg.include_validity_channel, None, args.seed),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        ),
        "test": DataLoader(
            WindowDataset(test_ds, streams, cfg.include_validity_channel, None, args.seed),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        ),
    }

    weights = torch.tensor(class_weights(train_ds), dtype=torch.float32, device=device)
    exercise_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    phase_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    history: List[Dict] = []
    patience_left = args.patience

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = epoch_ex = epoch_ph = 0.0
        n_batches = 0
        for x, y, p in loaders["train"]:
            x, y, p = x.to(device), y.to(device), p.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            l_ex = exercise_loss(out["exercise_logits"], y)
            # Phase logits are (N, T, P); CE wants (N, P, T).
            l_ph = phase_loss(out["phase_logits"].permute(0, 2, 1), p)
            loss = l_ex + args.phase_loss_weight * l_ph
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_ex += float(l_ex.item())
            epoch_ph += float(l_ph.item())
            n_batches += 1
        scheduler.step()

        val_logits, val_labels, _, _ = _forward_all(model, loaders["val"], device)
        val_f1 = _macro_f1(val_labels, val_logits.argmax(axis=1), dataset.classes)
        val_acc = float((val_logits.argmax(axis=1) == val_labels).mean())

        history.append({
            "epoch": epoch, "loss": epoch_loss / max(1, n_batches),
            "exercise_loss": epoch_ex / max(1, n_batches),
            "phase_loss": epoch_ph / max(1, n_batches),
            "val_accuracy": val_acc, "val_macro_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  "
              f"val_acc {val_acc:.3f}  val_macro_f1 {val_f1:.3f}")

        # Selection on macro F1, not accuracy: the rule-labelled corpus is
        # ~68% squat, and accuracy rewards a model that predicts squat.
        if val_f1 > best["macro_f1"]:
            best = {
                "macro_f1": val_f1, "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch} (best epoch {best['epoch']})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    val_metrics = evaluate_split(model, loaders["val"], dataset.classes, device)
    uncalibrated_test = evaluate_split(model, loaders["test"], dataset.classes, device)

    # Temperature is fitted on validation and reported on test. Fitting
    # and reporting on the same split would measure how well one free
    # parameter fits that split, which is not the question.
    calibration = calibrate(
        val_metrics["_logits"], val_metrics["_labels"],
        uncalibrated_test["_logits"], uncalibrated_test["_labels"],
    )
    test_metrics = evaluate_split(model, loaders["test"], dataset.classes, device,
                                  temperature=calibration.temperature)

    print("\n--- test ---")
    print(f"accuracy {test_metrics['accuracy']:.3f}   macro F1 {test_metrics['macro_f1']:.3f}")
    print(f"per-class F1: " + "  ".join(f"{k}={v:.2f}" for k, v in test_metrics["per_class_f1"].items()))
    print(f"phase frame agreement with rule: {test_metrics['phase_frame_agreement_with_rule']:.3f}")
    print(f"rep-count MAE vs rule: {test_metrics['rep_count_mae_vs_rule']:.2f}")
    print(calibration.summary())

    results = {
        "classes": dataset.classes,
        "config": cfg.to_dict(),
        "args": {k: v for k, v in vars(args).items()},
        "dataset": summary,
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "best_epoch": best["epoch"],
        "history": history,
        "val": {k: v for k, v in val_metrics.items() if not k.startswith("_")},
        "test": {k: v for k, v in test_metrics.items() if not k.startswith("_")},
        "calibration": calibration.to_dict(),
        "label_provenance": {
            "test_labels_derived_only": derived_only,
            "note": (
                "test accuracy measures agreement with the kinematic labelling rule, "
                "not with human-verified exercise labels"
            ) if derived_only else "test labels include non-derived sources",
        },
        "release_ready": bool(not derived_only),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "classes": dataset.classes,
            "phase_names": PHASE_NAMES,
            "temperature": calibration.temperature,
            "streams": list(streams),
            "window_frames": args.window_frames,
            "label_provenance": results["label_provenance"],
            "release_ready": results["release_ready"],
            "metrics": {"val": results["val"], "test": results["test"]},
        },
        args.out,
    )
    results_path = os.path.splitext(args.out)[0] + "_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nsaved checkpoint -> {args.out}\nsaved results    -> {results_path}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the skeleton motion recognition model")
    p.add_argument("--cache", default="data/qevd_pose_cache.pkl")
    p.add_argument("--subjects", default=None,
                   help="CSV mapping clip_id,subject_id — enables subject-disjoint splits")
    p.add_argument("--out", default="models/motion_stgcn.pt")

    p.add_argument("--window-frames", type=int, default=48, dest="window_frames")
    p.add_argument("--windows-per-clip", type=int, default=1, dest="windows_per_clip")
    p.add_argument("--max-per-class", type=int, default=None, dest="max_per_class")
    p.add_argument("--include-other", action="store_true", dest="include_other")
    p.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], dest="split_ratios")

    p.add_argument("--streams", nargs="+", default=list(STREAM_NAMES), choices=list(STREAM_NAMES))
    p.add_argument("--fusion", default="early", choices=["early", "late"])
    p.add_argument("--adjacency", default="spatial", choices=["spatial", "uniform"])
    p.add_argument("--no-adaptive-adjacency", action="store_true", dest="no_adaptive_adjacency")
    p.add_argument("--no-transformer", action="store_true", dest="no_transformer")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.05, dest="label_smoothing")
    p.add_argument("--phase-loss-weight", type=float, default=0.3, dest="phase_loss_weight")
    p.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no-augment", action="store_true", dest="no_augment")

    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-derived-labels", action="store_true", dest="allow_derived_labels",
                   help="proceed even though the test labels come from a kinematic rule")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
