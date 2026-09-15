"""
Trains and tests exercise classifiers (Random Forest, XGBoost, MLP, 1D-CNN)
on a subset of the real video dataset QEVD-FIT-300k-Part-1.

Run:
    python train_on_qevd.py --max-videos 50
"""
import argparse
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from recognition.qevd_dataset import load_qevd_dataset
from recognition.temporal_model import TemporalCNNClassifier
from recognition.classifier import save_model
from recognition.dataset_synthesis import EXERCISES
from evaluation.classification import classification_report
from collections import Counter
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as sk_classification_report, confusion_matrix, ConfusionMatrixDisplay

def save_confusion_matrix_plot(y_test, y_pred, model_name, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=EXERCISES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EXERCISES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"QEVD Exercise Classification ({model_name})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_confusion_matrix_table(cm, labels):
    width = max(len(l) for l in labels) + 2
    header = " " * width + "".join(f"{l[:10]:>12}" for l in labels)
    print(header, flush=True)
    for i, label in enumerate(labels):
        row = f"{label:<{width}}" + "".join(f"{cm[i, j]:>12}" for j in range(len(labels)))
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train and test models on QEVD-FIT-300k-Part-1 subset.")
    parser.add_argument("--target-per-class", type=int, default=10, help="Target valid video clips per exercise class (default: 10)")
    parser.add_argument("--max-videos", type=int, default=300, help="Max QEVD videos to scan (default: 300)")
    parser.add_argument("--n-frames", type=int, default=16, help="Frames per sequence (default: 16)")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test set fraction (default: 0.25)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()

    print(f"Scanning up to {args.max_videos} videos from dataset QEVD-FIT-300k-Part-1 for {args.target_per_class} clips per class...", flush=True)
    start_time = time.time()
    ds = load_qevd_dataset(target_per_class=args.target_per_class, max_videos_to_scan=args.max_videos, n_frames=args.n_frames)
    elapsed_load = time.time() - start_time

    n_samples = len(ds["y"])
    print(f"Extracted {n_samples} valid exercise video clips in {elapsed_load:.1f}s.", flush=True)
    if n_samples < 4:
        print("ERROR: Too few valid pose sequences extracted to perform train/test split. Try increasing --max-videos.", flush=True)
        return

    # Print class counts
    class_counts = Counter(ds["y"])
    print("Extracted class distribution:", dict(class_counts), flush=True)

    X_eng = ds["X_eng"]
    X_seq = ds["X_seq"]
    y = ds["y"]

    # Stratified split by video clip
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, stratify=y)

    print("\n" + "=" * 70, flush=True)
    print("DATASET SPLIT (BY VIDEO CLIP):", flush=True)
    print("Train:", Counter(y[train_idx]), flush=True)
    print("Test :", Counter(y[test_idx]), flush=True)
    print("=" * 70, flush=True)

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=args.seed)
    rf.fit(X_eng[train_idx], y[train_idx])
    y_pred_rf = rf.predict(X_eng[test_idx])
    report_rf = classification_report(y[test_idx], y_pred_rf, labels=EXERCISES)
    cm_rf = confusion_matrix(y[test_idx], y_pred_rf, labels=EXERCISES)

    print("\n--- RANDOM FOREST CLASSIFIER ---", flush=True)
    print(f"Accuracy: {report_rf.accuracy * 100:.1f}%   Macro F1: {report_rf.macro_f1:.3f}", flush=True)
    print("Confusion matrix (rows = true, columns = predicted):", flush=True)
    print_confusion_matrix_table(cm_rf, EXERCISES)
    print("\nClassification Report:", flush=True)
    print(sk_classification_report(y[test_idx], y_pred_rf, target_names=EXERCISES, zero_division=0), flush=True)

    # Save RF model & plot
    save_model(rf, "models/qevd_exercise_classifier.joblib")
    save_confusion_matrix_plot(y[test_idx], y_pred_rf, "RandomForest", "analysis/qevd_rf_confusion_matrix.png")
    print("Saved trained model to models/qevd_exercise_classifier.joblib", flush=True)
    print("Saved confusion matrix plot to analysis/qevd_rf_confusion_matrix.png", flush=True)

    # 2. MLP Classifier
    scaler = StandardScaler().fit(X_eng[train_idx])
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=args.seed)
    mlp.fit(scaler.transform(X_eng[train_idx]), y[train_idx])
    y_pred_mlp = mlp.predict(scaler.transform(X_eng[test_idx]))
    report_mlp = classification_report(y[test_idx], y_pred_mlp, labels=EXERCISES)
    cm_mlp = confusion_matrix(y[test_idx], y_pred_mlp, labels=EXERCISES)

    print("\n--- MLP CLASSIFIER ---", flush=True)
    print(f"Accuracy: {report_mlp.accuracy * 100:.1f}%   Macro F1: {report_mlp.macro_f1:.3f}", flush=True)
    print("Classification Report:", flush=True)
    print(sk_classification_report(y[test_idx], y_pred_mlp, target_names=EXERCISES, zero_division=0), flush=True)

    # 3. XGBoost
    train_unique_classes = np.unique(y[train_idx])
    le = LabelEncoder().fit(train_unique_classes)
    y_train_xgb = le.transform(y[train_idx])
    xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=args.seed, eval_metric="mlogloss", verbosity=0)
    xgb.fit(X_eng[train_idx], y_train_xgb)
    
    y_pred_xgb_raw = xgb.predict(X_eng[test_idx])
    y_pred_xgb = le.inverse_transform(y_pred_xgb_raw)
    report_xgb = classification_report(y[test_idx], y_pred_xgb, labels=EXERCISES)

    print("\n--- XGBOOST CLASSIFIER ---", flush=True)
    print(f"Accuracy: {report_xgb.accuracy * 100:.1f}%   Macro F1: {report_xgb.macro_f1:.3f}", flush=True)
    print("Classification Report:", flush=True)
    print(sk_classification_report(y[test_idx], y_pred_xgb, target_names=EXERCISES, zero_division=0), flush=True)

    # 4. Temporal 1D-CNN
    cnn = TemporalCNNClassifier(epochs=30, seed=args.seed)
    cnn.fit(X_seq[train_idx], y[train_idx])
    y_pred_cnn = cnn.predict(X_seq[test_idx])
    report_cnn = classification_report(y[test_idx], y_pred_cnn, labels=EXERCISES)

    print("\n--- TEMPORAL 1D-CNN CLASSIFIER ---", flush=True)
    print(f"Accuracy: {report_cnn.accuracy * 100:.1f}%   Macro F1: {report_cnn.macro_f1:.3f}", flush=True)
    print("Classification Report:", flush=True)
    print(sk_classification_report(y[test_idx], y_pred_cnn, target_names=EXERCISES, zero_division=0), flush=True)

    print("\n" + "=" * 70, flush=True)
    print("FINAL BALANCED DATASET EXPERIMENT SUMMARY:", flush=True)
    print(f"{'Model':<18}{'Accuracy':>12}{'Macro F1':>12}", flush=True)
    print("-" * 42, flush=True)
    print(f"{'Random Forest':<18}{report_rf.accuracy * 100:>11.1f}%{report_rf.macro_f1:>12.3f}", flush=True)
    print(f"{'MLP':<18}{report_mlp.accuracy * 100:>11.1f}%{report_mlp.macro_f1:>12.3f}", flush=True)
    print(f"{'XGBoost':<18}{report_xgb.accuracy * 100:>11.1f}%{report_xgb.macro_f1:>12.3f}", flush=True)
    print(f"{'Temporal 1D-CNN':<18}{report_cnn.accuracy * 100:>11.1f}%{report_cnn.macro_f1:>12.3f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
