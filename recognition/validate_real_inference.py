"""
Real QEVD Video Clip Inference Validation Script.

Selects 20 real QEVD video clips from the disk cache, runs complete inference
using ExerciseRecognizer model bundle (models/exercise_recognizer_bundle.joblib),
validates probability constraints (0 <= p <= 1, sum(p) == 1), measures timing,
and saves the output to analysis/real_video_inference_report.json.
"""

import json
import os
import time
from typing import Dict, List
import numpy as np

from recognition.exercise_recognizer import ExerciseRecognizer, ExerciseRecognitionResult
from recognition.qevd_dataset import load_or_build_pose_cache, EXERCISES


def run_real_inference_validation(bundle_path: str = "models/exercise_recognizer_bundle.joblib",
                                   output_path: str = "analysis/real_video_inference_report.json",
                                   min_clips: int = 20) -> Dict:
    """
    Evaluates ExerciseRecognizer on 20 real video clips from cache.
    """
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle '{bundle_path}' not found. Run benchmark first.")

    recognizer = ExerciseRecognizer.load_bundle(bundle_path)
    cache = load_or_build_pose_cache(target_per_class=25)

    vids = sorted(list(cache.keys()))[:min_clips]

    results = []
    success_count = 0
    failure_count = 0
    inference_times = []
    valid_prob_count = 0

    print(f"\nRunning Real QEVD Inference Validation on {len(vids)} clips using bundle '{bundle_path}'...")

    for vid in vids:
        item = cache[vid]
        pose_frames = item["pose_frames"]
        heuristic_label = item["label"]

        t0 = time.time()
        res: ExerciseRecognitionResult = recognizer.predict_sequence(pose_frames)
        t1 = time.time()

        elapsed_ms = (t1 - t0) * 1000.0
        inference_times.append(elapsed_ms)

        # Validate probability constraints
        probs = list(res.probabilities.values())
        probs_bounded = all(0.0 <= p <= 1.0 for p in probs)
        prob_sum_valid = abs(sum(probs) - 1.0) < 1e-3

        if probs_bounded and prob_sum_valid:
            valid_prob_count += 1

        is_success = res.frame_count > 0 and probs_bounded and prob_sum_valid
        if is_success:
            success_count += 1
        else:
            failure_count += 1

        res_dict = res.to_dict()
        res_dict["video_id"] = vid
        res_dict["heuristic_label"] = heuristic_label
        res_dict["inference_time_ms"] = elapsed_ms
        res_dict["probability_bounded"] = probs_bounded
        res_dict["probability_sum_valid"] = prob_sum_valid

        results.append(res_dict)

    avg_latency = np.mean(inference_times)
    total_frames = sum(r["frame_count"] for r in results)
    total_time_s = sum(inference_times) / 1000.0
    effective_fps = (total_frames / total_time_s) if total_time_s > 0 else 0.0

    report = {
        "total_clips_evaluated": len(vids),
        "successful_inferences": success_count,
        "failed_inferences": failure_count,
        "success_rate": float(success_count / len(vids)),
        "valid_probability_ratio": float(valid_prob_count / len(vids)),
        "average_inference_ms_per_clip": float(avg_latency),
        "total_processed_frames": int(total_frames),
        "effective_inference_fps": float(effective_fps),
        "clip_results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation Complete! Success Rate: {report['success_rate']*100:.1f}%, Avg Latency: {avg_latency:.2f} ms/clip, Effective FPS: {effective_fps:.1f}")
    print(f"Report saved to '{output_path}'")
    return report


def main():
    run_real_inference_validation()


if __name__ == "__main__":
    main()
