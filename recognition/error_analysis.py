"""
Detailed Prediction Error Analysis for QEVD Exercise Classification.

Saves prediction metadata for held-out test clips:
  - video_id, true_label, predicted_label, confidence, probabilities, kinematic features
Groups misclassifications by:
  - true_label vs predicted_label
  - confidence distribution (high-confidence vs low-confidence errors)
  - top confusion pairs (e.g. bicep_curl vs lateral_raise)
Exports error analysis report to analysis/error_analysis_report.json
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np

from recognition.qevd_dataset import load_qevd_dataset, train_val_test_split_by_video, EXERCISES
from recognition.feature_engineering import FeatureConfig
from sklearn.ensemble import RandomForestClassifier


def run_error_analysis(dataset_dir: str = "QEVD-FIT-300k-Part-1",
                       target_per_class: int = 25,
                       seed: int = 42,
                       output_report_path: str = "analysis/error_analysis_report.json") -> Dict:
    """
    Evaluates Random Forest on test set, logs detailed prediction outputs, and identifies top confusion pairs.
    """
    cfg = FeatureConfig()
    ds = load_qevd_dataset(dataset_dir=dataset_dir, target_per_class=target_per_class, feature_config=cfg)
    train_data, val_data, test_data = train_val_test_split_by_video(ds, seed=seed)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    rf.fit(train_data["X_tab"], train_data["y"])

    test_vids = test_data["video_ids"]
    y_true = test_data["y"]
    X_test = test_data["X_tab"]

    preds = rf.predict(X_test)
    probas = rf.predict_proba(X_test)
    classes = list(rf.classes_)

    predictions_log = []
    errors = []
    confusion_pairs: Dict[str, int] = {}

    for i in range(len(y_true)):
        vid = test_vids[i]
        true_lbl = y_true[i]
        pred_lbl = preds[i]
        prob_dict = {c: float(probas[i][classes.index(c)]) for c in classes}
        conf = float(prob_dict[pred_lbl])

        item = {
            "video_id": vid,
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "confidence": conf,
            "probabilities": prob_dict,
            "is_correct": bool(true_lbl == pred_lbl),
        }
        predictions_log.append(item)

        if true_lbl != pred_lbl:
            errors.append(item)
            pair_key = f"{true_lbl} -> {pred_lbl}"
            confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1

    total_test = len(y_true)
    total_errors = len(errors)
    acc = (total_test - total_errors) / max(1, total_test)

    # Sort confusion pairs
    top_confusion_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    report = {
        "dataset_size": len(ds["y"]),
        "test_sample_count": total_test,
        "correct_predictions": total_test - total_errors,
        "error_count": total_errors,
        "test_accuracy": float(acc),
        "top_confusion_pairs": top_confusion_pairs,
        "errors": errors,
        "all_predictions": predictions_log,
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    print("Running Prediction Error Analysis...")
    report = run_error_analysis()
    print(f"Test Accuracy: {report['test_accuracy']*100:.1f}% ({report['correct_predictions']}/{report['test_sample_count']})")
    print(f"Error Count: {report['error_count']}")
    print(f"Top Confusion Pairs: {report['top_confusion_pairs']}")
    print(f"Report saved to 'analysis/error_analysis_report.json'")


if __name__ == "__main__":
    main()
