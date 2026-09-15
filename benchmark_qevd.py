"""
Reproducible Benchmark Suite, Multi-Seed Evaluation, & Feature Ablation for QEVD Exercise Classification.

Evaluates 7 baseline models:
  1. Majority-class Baseline
  2. Uniform Random Baseline
  3. Logistic Regression
  4. Random Forest
  5. XGBoost
  6. Multi-Layer Perceptron (MLP)
  7. Temporal 1D-CNN

Statistical Evaluation:
  Runs multi-seed evaluations across deterministic seeds [42, 101, 2024, 7, 999],
  computing Mean ± Standard Deviation and 95% Confidence Intervals for Accuracy and Macro F1.

Feature Ablation Study:
  - Level 1: Raw Landmarks
  - Level 2: + Joint Angles
  - Level 3: + Velocity
  - Level 4: + Acceleration
  - Level 5: + Symmetry & Range of Motion
"""

import argparse
import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from recognition.qevd_dataset import (
    load_qevd_dataset,
    train_val_test_split_by_video,
    generate_dataset_audit_report,
    EXERCISES,
)
from recognition.feature_engineering import FeatureConfig
from recognition.temporal_model import TemporalCNNClassifier
from recognition.exercise_recognizer import train_and_save_recognizer


def evaluate_model_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                                classes: List[str]) -> Dict:
    """Computes comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weight_p, weight_r, weight_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class_metrics = {}
    for i, c in enumerate(classes):
        per_class_metrics[c] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_supp[i]),
        }

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weight_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
    }


def plot_and_save_confusion_matrix(cm: np.ndarray, classes: List[str],
                                   title: str, output_path: str):
    """Generates and saves a confusion matrix visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def train_and_evaluate_all_models(train_data: Dict, test_data: Dict,
                                  classes: List[str], seed: int = 42) -> Dict[str, Dict]:
    """Trains 7 baseline models on train_data and evaluates on test_data."""
    X_tr_tab, y_tr = train_data["X_tab"], train_data["y"]
    X_te_tab, y_te = test_data["X_tab"], test_data["y"]

    X_tr_seq = train_data["X_seq"]
    X_te_seq = test_data["X_seq"]

    # Preprocessing: Fit StandardScaler STRICTLY on training split
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_tab)
    X_te_scaled = scaler.transform(X_te_tab)

    results = {}

    # 1. Majority-Class Baseline
    maj_model = DummyClassifier(strategy="most_frequent")
    maj_model.fit(X_tr_scaled, y_tr)
    results["Majority Baseline"] = evaluate_model_predictions(y_te, maj_model.predict(X_te_scaled), classes)

    # 2. Random Baseline
    rand_model = DummyClassifier(strategy="uniform", random_state=seed)
    rand_model.fit(X_tr_scaled, y_tr)
    results["Random Baseline"] = evaluate_model_predictions(y_te, rand_model.predict(X_te_scaled), classes)

    # 3. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=seed)
    lr_model.fit(X_tr_scaled, y_tr)
    results["Logistic Regression"] = evaluate_model_predictions(y_te, lr_model.predict(X_te_scaled), classes)

    # 4. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    rf_model.fit(X_tr_tab, y_tr)
    results["Random Forest"] = evaluate_model_predictions(y_te, rf_model.predict(X_te_tab), classes)

    # 5. XGBoost
    le = LabelEncoder()
    le.fit(classes)
    y_tr_enc = le.transform(y_tr)

    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, random_state=seed, eval_metric="mlogloss")
    xgb_model.fit(X_tr_tab, y_tr_enc)
    xgb_preds_enc = xgb_model.predict(X_te_tab)
    xgb_preds = le.inverse_transform(xgb_preds_enc)
    results["XGBoost"] = evaluate_model_predictions(y_te, xgb_preds, classes)

    # 6. MLP
    mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed)
    mlp_model.fit(X_tr_scaled, y_tr)
    results["MLP Classifier"] = evaluate_model_predictions(y_te, mlp_model.predict(X_te_scaled), classes)

    # 7. Temporal 1D-CNN
    cnn_model = TemporalCNNClassifier(epochs=40, seed=seed)
    cnn_model.fit(X_tr_seq, y_tr)
    results["Temporal 1D-CNN"] = evaluate_model_predictions(y_te, cnn_model.predict(X_te_seq), classes)

    return results


def run_multi_seed_evaluation(full_ds: Dict, seeds: List[int] = [42, 101, 2024, 7, 999],
                              verbose: bool = True) -> Dict[str, Dict]:
    """
    Runs multi-seed evaluation across seeds and computes Mean ± Std Dev and 95% Confidence Intervals.
    Ensures:
      - Each seed creates an independent video-level split
      - Preprocessing is fitted ONLY on training data
      - Exact sample count (N_train, N_val, N_test) is reported per seed
      - Split-to-split variance is explicitly distinguished from population confidence
    """
    classes = EXERCISES
    model_seed_results = {}

    if verbose:
        print("\n" + "=" * 90)
        print("PER-SEED INDEPENDENT VIDEO-LEVEL SPLIT EVALUATION")
        print("=" * 90)

    for seed in seeds:
        train_data, val_data, test_data = train_val_test_split_by_video(full_ds, seed=seed)
        n_tr, n_val, n_te = len(train_data["y"]), len(val_data["y"]), len(test_data["y"])

        if verbose:
            print(f"\n---> Seed {seed:<4} | Train: {n_tr} vids | Val: {n_val} vids | Test: {n_te} vids")

        res = train_and_evaluate_all_models(train_data, test_data, classes, seed=seed)

        if verbose:
            rf_res = res["Random Forest"]
            print(f"     Random Forest Test Acc: {rf_res['accuracy']*100:5.1f}% | Macro F1: {rf_res['macro_f1']:5.3f}")

        for model_name, m in res.items():
            if model_name not in model_seed_results:
                model_seed_results[model_name] = {"acc": [], "macro_f1": [], "wtd_f1": []}
            model_seed_results[model_name]["acc"].append(m["accuracy"])
            model_seed_results[model_name]["macro_f1"].append(m["macro_f1"])
            model_seed_results[model_name]["wtd_f1"].append(m["weighted_f1"])

    summary_stats = {}
    K = len(seeds)
    for model_name, vals in model_seed_results.items():
        acc_arr = np.array(vals["acc"])
        f1_arr = np.array(vals["macro_f1"])

        acc_mean, acc_std = np.mean(acc_arr), np.std(acc_arr)
        f1_mean, f1_std = np.mean(f1_arr), np.std(f1_arr)

        acc_ci = 1.96 * acc_std / np.sqrt(K)
        f1_ci = 1.96 * f1_std / np.sqrt(K)

        summary_stats[model_name] = {
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "acc_ci95": acc_ci,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "f1_ci95": f1_ci,
        }

    return summary_stats


def run_ablation_experiment(dataset_dir: str, target_per_class: int,
                           seed: int = 42) -> List[Dict]:
    """Runs 5-stage feature group ablation study using Random Forest classifier."""
    ablation_levels = [
        ("Level 1: Landmarks Only", FeatureConfig(
            use_raw_landmarks=True, use_joint_angles=False, use_velocity=False,
            use_acceleration=False, use_symmetry=False, use_range_of_motion=False
        )),
        ("Level 2: + Joint Angles", FeatureConfig(
            use_raw_landmarks=True, use_joint_angles=True, use_velocity=False,
            use_acceleration=False, use_symmetry=False, use_range_of_motion=False
        )),
        ("Level 3: + Velocity", FeatureConfig(
            use_raw_landmarks=True, use_joint_angles=True, use_velocity=True,
            use_acceleration=False, use_symmetry=False, use_range_of_motion=False
        )),
        ("Level 4: + Acceleration", FeatureConfig(
            use_raw_landmarks=True, use_joint_angles=True, use_velocity=True,
            use_acceleration=True, use_symmetry=False, use_range_of_motion=False
        )),
        ("Level 5: + Symmetry & ROM", FeatureConfig(
            use_raw_landmarks=True, use_joint_angles=True, use_velocity=True,
            use_acceleration=True, use_symmetry=True, use_range_of_motion=True
        )),
    ]

    ablation_results = []
    classes = EXERCISES

    print("\n" + "=" * 80)
    print(f"RUNNING FEATURE ABLATION EXPERIMENT (Target per class: {target_per_class})")
    print("=" * 80)

    for level_name, cfg in ablation_levels:
        ds = load_qevd_dataset(dataset_dir=dataset_dir, target_per_class=target_per_class, feature_config=cfg)
        train_data, val_data, test_data = train_val_test_split_by_video(ds, seed=seed)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
        rf.fit(train_data["X_tab"], train_data["y"])
        preds = rf.predict(test_data["X_tab"])

        eval_res = evaluate_model_predictions(test_data["y"], preds, classes)
        eval_res["level"] = level_name
        eval_res["n_features"] = train_data["X_tab"].shape[1]
        ablation_results.append(eval_res)

        print(f"{level_name:<30} | Features: {train_data['X_tab'].shape[1]:<4} | Accuracy: {eval_res['accuracy']*100:5.1f}% | Macro F1: {eval_res['macro_f1']:5.3f}")

    # Plot ablation chart
    os.makedirs("analysis", exist_ok=True)
    levels = [r["level"].split(":")[0] for r in ablation_results]
    accs = [r["accuracy"] * 100 for r in ablation_results]
    f1s = [r["macro_f1"] for r in ablation_results]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color = 'tab:blue'
    ax1.set_xlabel('Feature Ablation Level')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(levels, accs, color=color, marker='o', linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Macro F1', color=color)
    ax2.plot(levels, f1s, color=color, marker='s', linewidth=2, linestyle='--', label='Macro F1')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Feature Ablation Study: Performance vs Feature Group')
    fig.tight_layout()
    plt.savefig('analysis/feature_ablation_plot.png', dpi=200)
    plt.close(fig)

    return ablation_results


def run_temporal_window_experiment(dataset_dir: str, target_per_class: int = 25,
                                   seed: int = 42) -> List[Dict]:
    """
    Evaluates controlled temporal window resolutions T in [8, 16, 32, 64].
    Verifies whether temporal velocity advantage remains consistent across temporal lengths.
    """
    window_lengths = [8, 16, 32, 64]
    window_results = []
    classes = EXERCISES

    print("\n" + "=" * 80)
    print(f"RUNNING TEMPORAL WINDOW RESOLUTION EXPERIMENT (T in [8, 16, 32, 64])")
    print("=" * 80)

    for T_len in window_lengths:
        cfg = FeatureConfig(use_raw_landmarks=True, use_joint_angles=True, use_velocity=True,
                            use_acceleration=True, use_symmetry=True, use_range_of_motion=True)
        ds = load_qevd_dataset(dataset_dir=dataset_dir, target_per_class=target_per_class, n_frames=T_len, feature_config=cfg)
        train_data, val_data, test_data = train_val_test_split_by_video(ds, seed=seed)

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
        rf.fit(train_data["X_tab"], train_data["y"])
        rf_eval = evaluate_model_predictions(test_data["y"], rf.predict(test_data["X_tab"]), classes)

        # 2. Temporal 1D-CNN
        cnn = TemporalCNNClassifier(epochs=40, seed=seed)
        cnn.fit(train_data["X_seq"], train_data["y"])
        cnn_eval = evaluate_model_predictions(test_data["y"], cnn.predict(test_data["X_seq"]), classes)

        item = {
            "T": T_len,
            "rf_acc": rf_eval["accuracy"],
            "rf_f1": rf_eval["macro_f1"],
            "cnn_acc": cnn_eval["accuracy"],
            "cnn_f1": cnn_eval["macro_f1"],
        }
        window_results.append(item)

        print(f"T = {T_len:<2} frames | RF Acc: {rf_eval['accuracy']*100:5.1f}% (F1: {rf_eval['macro_f1']:5.3f}) | 1D-CNN Acc: {cnn_eval['accuracy']*100:5.1f}% (F1: {cnn_eval['macro_f1']:5.3f})")

    return window_results


def print_multi_seed_table(stats: Dict[str, Dict]):
    """Prints formatted ASCII markdown table of multi-seed benchmark results."""
    print("\n" + "=" * 105)
    print("QEVD EXERCISE CLASSIFICATION STATISTICAL BENCHMARK (5 SEEDS: MEAN ± STD, 95% CI)")
    print("=" * 105)
    header = f"{'Model':<25} | {'Accuracy (Mean ± Std)':<25} | {'95% CI (Acc)':<15} | {'Macro F1 (Mean ± Std)':<25} | {'95% CI (F1)':<15}"
    print(header)
    print("-" * 105)
    for model_name, s in stats.items():
        acc_str = f"{s['acc_mean']*100:5.1f}% ± {s['acc_std']*100:4.1f}%"
        acc_ci_str = f"± {s['acc_ci95']*100:4.1f}%"
        f1_str = f"{s['f1_mean']:5.3f} ± {s['f1_std']:5.3f}"
        f1_ci_str = f"± {s['f1_ci95']:5.3f}"
        print(f"{model_name:<25} | {acc_str:<25} | {acc_ci_str:<15} | {f1_str:<25} | {f1_ci_str:<15}")
    print("-" * 105)


def main():
    parser = argparse.ArgumentParser(description="Benchmark QEVD Exercise Classification")
    parser.add_argument("--dataset-dir", type=str, default="QEVD-FIT-300k-Part-1")
    parser.add_argument("--target-per-class", type=int, default=25, help="Number of clips per class (25, 62, 125, 250)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-plots", action="store_true", default=True)
    args = parser.parse_args()

    classes = EXERCISES

    print(f"Loading QEVD Dataset (target: {args.target_per_class} per class)...")
    full_ds = load_qevd_dataset(dataset_dir=args.dataset_dir, target_per_class=args.target_per_class)
    train_data, val_data, test_data = train_val_test_split_by_video(full_ds, seed=args.seed)

    audit = generate_dataset_audit_report(full_ds, train_data, val_data, test_data)
    print("\n--- DATASET AUDIT REPORT ---")
    print(f"Total Videos: {audit['total_videos']} across {audit['class_counts']}")
    print(f"Split Videos -> Train: {audit['train_count']}, Val: {audit['val_count']}, Test: {audit['test_count']}")
    print(f"Tabular Feature Dim: {audit['tabular_feature_dim']}, Sequence Tensor Shape: {audit['sequence_shape']}")

    # Single seed evaluation
    single_res = train_and_evaluate_all_models(train_data, test_data, classes, seed=args.seed)

    if args.save_plots:
        os.makedirs("analysis", exist_ok=True)
        for model_name, m in single_res.items():
            safe_name = model_name.lower().replace(" ", "_")
            plot_and_save_confusion_matrix(
                m["confusion_matrix"], classes,
                f"Confusion Matrix - {model_name}",
                f"analysis/qevd_cm_{safe_name}.png"
            )

    # Multi-seed statistical evaluation
    stats = run_multi_seed_evaluation(full_ds, seeds=[42, 101, 2024, 7, 999])
    print_multi_seed_table(stats)

    # Run feature ablation
    run_ablation_experiment(args.dataset_dir, target_per_class=args.target_per_class, seed=args.seed)

    # Run temporal window experiment (T=8, 16, 32, 64)
    run_temporal_window_experiment(args.dataset_dir, target_per_class=args.target_per_class, seed=args.seed)

    # Train and serialize production recognizer bundle
    print("\nTraining and serializing production model bundle to 'models/exercise_recognizer_bundle.joblib'...")
    train_and_save_recognizer(dataset_dir=args.dataset_dir, target_per_class=args.target_per_class, seed=args.seed)
    print("Saved model bundle successfully.")


if __name__ == "__main__":
    main()
