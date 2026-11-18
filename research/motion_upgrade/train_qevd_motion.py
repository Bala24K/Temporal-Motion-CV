"""
Comprehensive multi-seed statistical benchmark runner comparing:
  - Motion-Upgrade ST-GCN + Temporal Transformer (Learned Graph Model)
  - Random Forest (Modular Feature Matrix, 870 dims)
  - XGBoost (Modular Feature Matrix)
  - Temporal 1D-CNN (Sequence Matrix)
  - MLP Classifier (Standardized Feature Matrix)

Evaluates on QEVD-FIT-300k-Part-1 (247 extracted video clips) across 5 deterministic random seeds.
Reports Mean +- Std Dev, 95% Confidence Intervals, Calibration metrics (ECE, Brier), Latency, and Confusion Matrices.
"""
import argparse
import os
import sys
import time
import json
import pickle
import numpy as np
from collections import Counter, defaultdict

# Ensure PYTHONPATH includes Ver-1 and motion-upgrade
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
ver1_dir = os.path.join(workspace_dir, "Ver-1")
if ver1_dir not in sys.path:
    sys.path.insert(0, ver1_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report as sk_classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from evaluation.classification import classification_report
from recognition.feature_engineering import FeatureConfig, extract_modular_tabular_features, extract_modular_sequence_features

from motion.dataset import (
    EXERCISE_CLASSES,
    MotionDataset,
    build_dataset_from_pose_cache,
    group_disjoint_split,
)
from motion.model import MotionModelConfig, build_model, count_parameters
from motion.tensors import STREAM_NAMES, build_streams
from motion.calibration import calibrate, apply_temperature, expected_calibration_error, brier_score
from recognition.temporal_model import TemporalCNNClassifier

import torch
from torch.utils.data import DataLoader, Dataset


class STGCNDataset(Dataset):
    def __init__(self, samples, class_to_index, streams=("joint", "bone", "joint_motion", "bone_motion"), include_validity=True):
        self.samples = samples
        self.class_to_index = class_to_index
        self.streams = streams
        self.include_validity = include_validity

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = build_streams(sample.window, self.streams, self.include_validity)
        y = self.class_to_index[sample.label]
        phases = sample.phase_labels
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.tensor(y, dtype=torch.long),
            torch.from_numpy(np.ascontiguousarray(phases, dtype=np.int64)),
        )


def train_eval_stgcn(train_ds, val_ds, test_ds, seed=42, epochs=35, batch_size=16, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    classes = train_ds.classes
    class_to_index = train_ds.class_to_index
    
    train_loader = DataLoader(STGCNDataset(train_ds.samples, class_to_index), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(STGCNDataset(val_ds.samples, class_to_index), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(STGCNDataset(test_ds.samples, class_to_index), batch_size=batch_size, shuffle=False)
    
    # Model config matching 15 nodes, early fusion
    in_channels = 13 # 3 (coords) * 4 (streams) + 1 (validity)
    cfg = MotionModelConfig(num_classes=len(classes), streams=("joint", "bone", "joint_motion", "bone_motion"), include_validity_channel=True, transformer_layers=2, transformer_heads=4)
    model = build_model(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_ex = torch.nn.CrossEntropyLoss()
    criterion_ph = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss_ex = criterion_ex(out["exercise_logits"], y)
            loss_ph = criterion_ph(out["phase_logits"].transpose(1, 2), p)
            loss = loss_ex + 0.5 * loss_ph
            loss.backward()
            optimizer.step()
            
    # Calibration on Val split
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for x, y, _ in val_loader:
            out = model(x.to(device))
            val_logits.append(out["exercise_logits"].cpu().numpy())
            val_labels.append(y.numpy())
    val_logits = np.concatenate(val_logits)
    val_labels = np.concatenate(val_labels)
    cal_res = calibrate(val_logits, val_labels)
    temp = cal_res.temperature
    
    # Test evaluation
    test_logits, test_labels = [], []
    t_start = time.time()
    with torch.no_grad():
        for x, y, _ in test_loader:
            out = model(x.to(device))
            test_logits.append(out["exercise_logits"].cpu().numpy())
            test_labels.append(y.numpy())
    t_end = time.time()
    latency_per_sample_ms = ((t_end - t_start) / max(1, len(test_ds.samples))) * 1000.0
    
    test_logits = np.concatenate(test_logits)
    test_labels = np.concatenate(test_labels)
    
    probs = apply_temperature(test_logits, temp)
    preds = probs.argmax(axis=1)
    
    y_true_str = [classes[i] for i in test_labels]
    y_pred_str = [classes[i] for i in preds]
    rep = classification_report(y_true_str, y_pred_str, classes)
    
    ece, _ = expected_calibration_error(probs, test_labels)
    brier = brier_score(probs, test_labels)
    
    return {
        "accuracy": rep.accuracy,
        "macro_f1": rep.macro_f1,
        "precision": rep.precision,
        "recall": rep.recall,
        "f1_per_class": dict(zip(classes, rep.f1)),
        "ece": ece,
        "brier": brier,
        "latency_ms": latency_per_sample_ms,
        "y_true": y_true_str,
        "y_pred": y_pred_str,
        "cm": rep.confusion_matrix.tolist(),
        "num_params": count_parameters(model),
    }


def extract_tabular_data_from_cache(cache_path, feature_config=None, target_classes=EXERCISE_CLASSES):
    if feature_config is None:
        feature_config = FeatureConfig()
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    clip_ids = sorted(cache.keys())
    X_eng, X_seq, y, clip_list = [], [], [], []
    
    for cid in clip_ids:
        item = cache[cid]
        lbl = item["label"]
        if lbl not in target_classes:
            continue
        pose_frames = item["pose_frames"]
        if not pose_frames or len(pose_frames) < 5:
            continue
            
        f_eng = extract_modular_tabular_features(pose_frames, feature_config)
        seq_mat = extract_modular_sequence_features(pose_frames, feature_config)
        
        # Resample sequence to 16 frames for 1D-CNN baseline
        if len(seq_mat) >= 16:
            idx = np.linspace(0, len(seq_mat) - 1, 16, dtype=int)
            seq_mat_fixed = seq_mat[idx]
        else:
            pad_count = 16 - len(seq_mat)
            last_row = seq_mat[-1:]
            seq_mat_fixed = np.vstack([seq_mat, np.repeat(last_row, pad_count, axis=0)])
            
        X_eng.append(f_eng)
        X_seq.append(seq_mat_fixed)
        y.append(lbl)
        clip_list.append(cid)
        
    return np.array(X_eng, dtype=np.float32), np.array(X_seq, dtype=np.float32), np.array(y), clip_list


def eval_baseline_models(cache_path, seeds=[42, 101, 2024, 7, 999]):
    feature_config = FeatureConfig()
    X_eng, X_seq, y_all, clip_list = extract_tabular_data_from_cache(cache_path, feature_config)
    clip_to_idx = {cid: i for i, cid in enumerate(clip_list)}
    
    results = {
        "RandomForest": defaultdict(list),
        "XGBoost": defaultdict(list),
        "Temporal1DCNN": defaultdict(list),
        "MLP": defaultdict(list),
        "STGCN": defaultdict(list),
    }
    
    models_cm = {m: np.zeros((len(EXERCISE_CLASSES), len(EXERCISE_CLASSES)), dtype=int) for m in results}
    
    for seed in seeds:
        # Load dataset & group-disjoint split
        ds = build_dataset_from_pose_cache(cache_path, seed=seed)
        train_ds, val_ds, test_ds = group_disjoint_split(ds, seed=seed)
        
        # Get matching indices for tabular baseline models
        train_cids = [s.clip_id for s in train_ds.samples if s.clip_id in clip_to_idx]
        val_cids = [s.clip_id for s in val_ds.samples if s.clip_id in clip_to_idx]
        test_cids = [s.clip_id for s in test_ds.samples if s.clip_id in clip_to_idx]
        
        tr_idx = np.array([clip_to_idx[c] for c in train_cids])
        val_idx = np.array([clip_to_idx[c] for c in val_cids])
        te_idx = np.array([clip_to_idx[c] for c in test_cids])
        
        y_train = y_all[tr_idx]
        y_val = y_all[val_idx]
        y_test = y_all[te_idx]
        
        # 1. Random Forest
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
        rf.fit(X_eng[tr_idx], y_train)
        rf_probs = rf.predict_proba(X_eng[te_idx])
        t1 = time.time()
        rf_preds = rf.classes_[rf_probs.argmax(axis=1)]
        rf_rep = classification_report(y_test, rf_preds, EXERCISE_CLASSES)
        rf_lat = ((t1 - t0) / len(te_idx)) * 1000.0
        
        results["RandomForest"]["accuracy"].append(rf_rep.accuracy)
        results["RandomForest"]["macro_f1"].append(rf_rep.macro_f1)
        results["RandomForest"]["latency_ms"].append(rf_lat)
        models_cm["RandomForest"] += rf_rep.confusion_matrix
        
        # 2. XGBoost
        le = LabelEncoder().fit(EXERCISE_CLASSES)
        y_tr_num = le.transform(y_train)
        xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=seed, eval_metric="mlogloss", verbosity=0)
        xgb.fit(X_eng[tr_idx], y_tr_num)
        t0 = time.time()
        xgb_probs = xgb.predict_proba(X_eng[te_idx])
        t1 = time.time()
        xgb_preds = le.inverse_transform(xgb_probs.argmax(axis=1))
        xgb_rep = classification_report(y_test, xgb_preds, EXERCISE_CLASSES)
        xgb_lat = ((t1 - t0) / len(te_idx)) * 1000.0
        
        results["XGBoost"]["accuracy"].append(xgb_rep.accuracy)
        results["XGBoost"]["macro_f1"].append(xgb_rep.macro_f1)
        results["XGBoost"]["latency_ms"].append(xgb_lat)
        models_cm["XGBoost"] += xgb_rep.confusion_matrix
        
        # 3. MLP Classifier (with StandardScaler)
        scaler = StandardScaler().fit(X_eng[tr_idx])
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed)
        mlp.fit(scaler.transform(X_eng[tr_idx]), y_train)
        t0 = time.time()
        mlp_probs = mlp.predict_proba(scaler.transform(X_eng[te_idx]))
        t1 = time.time()
        mlp_preds = mlp.classes_[mlp_probs.argmax(axis=1)]
        mlp_rep = classification_report(y_test, mlp_preds, EXERCISE_CLASSES)
        mlp_lat = ((t1 - t0) / len(te_idx)) * 1000.0
        
        results["MLP"]["accuracy"].append(mlp_rep.accuracy)
        results["MLP"]["macro_f1"].append(mlp_rep.macro_f1)
        results["MLP"]["latency_ms"].append(mlp_lat)
        models_cm["MLP"] += mlp_rep.confusion_matrix
        
        # 4. Temporal 1D-CNN
        cnn = TemporalCNNClassifier(epochs=30, seed=seed)
        cnn.fit(X_seq[tr_idx], y_train)
        t0 = time.time()
        cnn_preds = cnn.predict(X_seq[te_idx])
        t1 = time.time()
        cnn_rep = classification_report(y_test, cnn_preds, EXERCISE_CLASSES)
        cnn_lat = ((t1 - t0) / len(te_idx)) * 1000.0
        
        results["Temporal1DCNN"]["accuracy"].append(cnn_rep.accuracy)
        results["Temporal1DCNN"]["macro_f1"].append(cnn_rep.macro_f1)
        results["Temporal1DCNN"]["latency_ms"].append(cnn_lat)
        models_cm["Temporal1DCNN"] += cnn_rep.confusion_matrix
        
        # 5. ST-GCN + Transformer (Motion-Upgrade)
        st_res = train_eval_stgcn(train_ds, val_ds, test_ds, seed=seed)
        results["STGCN"]["accuracy"].append(st_res["accuracy"])
        results["STGCN"]["macro_f1"].append(st_res["macro_f1"])
        results["STGCN"]["ece"].append(st_res["ece"])
        results["STGCN"]["brier"].append(st_res["brier"])
        results["STGCN"]["latency_ms"].append(st_res["latency_ms"])
        models_cm["STGCN"] += np.array(st_res["cm"], dtype=int)
        
    return results, models_cm


def compute_ci(data, confidence=0.95):
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    # 95% CI multiplier for t-distribution (N=5 -> df=4 -> 2.776)
    t_val = 2.776 if len(arr) == 5 else 1.96
    ci = t_val * (std / np.sqrt(len(arr))) if len(arr) > 0 else 0.0
    return mean, std, ci


def save_confusion_matrix_plot(cm, model_name, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EXERCISE_CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"QEVD Exercise Classification ({model_name})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=r"c:\TernaryAnnum\TemporalMotionAnalyzer\Ver-1\data\qevd_pose_cache.pkl")
    parser.add_argument("--out-dir", default=r"c:\TernaryAnnum\TemporalMotionAnalyzer\analysis")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [42, 101, 2024, 7, 999]
    
    print("=" * 80)
    print("RUNNING MULTI-SEED STATISTICAL BENCHMARK ON QEVD DATASET")
    print(f"Seeds: {seeds}")
    print("=" * 80)
    
    results, models_cm = eval_baseline_models(args.cache, seeds=seeds)
    
    summary_table = []
    print("\n" + "=" * 85)
    print(f"{'Model':<22}{'Accuracy (Mean ± Std)':<25}{'95% CI (Acc)':<15}{'Macro F1 (Mean ± Std)':<25}")
    print("-" * 85)
    
    report_data = {}
    for model_name, metrics in results.items():
        acc_m, acc_std, acc_ci = compute_ci(metrics["accuracy"])
        f1_m, f1_std, f1_ci = compute_ci(metrics["macro_f1"])
        lat_m = np.mean(metrics["latency_ms"])
        
        acc_str = f"{acc_m*100:.1f}% ± {acc_std*100:.1f}%"
        ci_str = f"± {acc_ci*100:.1f}%"
        f1_str = f"{f1_m:.3f} ± {f1_std:.3f}"
        
        print(f"{model_name:<22}{acc_str:<25}{ci_str:<15}{f1_str:<25}")
        
        # Save CM plot
        cm_path = os.path.join(args.out_dir, f"qevd_{model_name.lower()}_confusion_matrix.png")
        save_confusion_matrix_plot(models_cm[model_name], model_name, cm_path)
        
        report_data[model_name] = {
            "accuracy_mean": float(acc_m),
            "accuracy_std": float(acc_std),
            "accuracy_ci95": float(acc_ci),
            "macro_f1_mean": float(f1_m),
            "macro_f1_std": float(f1_std),
            "macro_f1_ci95": float(f1_ci),
            "latency_ms_mean": float(lat_m),
            "confusion_matrix_sum": models_cm[model_name].tolist(),
        }
        if "ece" in metrics:
            report_data[model_name]["ece_mean"] = float(np.mean(metrics["ece"]))
            report_data[model_name]["brier_mean"] = float(np.mean(metrics["brier"]))
            
    print("=" * 85)
    
    json_path = os.path.join(args.out_dir, "qevd_benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\nSaved benchmark results to {json_path}")
    print(f"Saved confusion matrix plots to {args.out_dir}")


if __name__ == "__main__":
    main()
