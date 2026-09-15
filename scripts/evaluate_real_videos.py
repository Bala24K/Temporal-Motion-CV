"""
Real-Time Fitness CV Evaluation Harness for Real Recorded Videos.

Runs the complete online FitnessCVEngine frame-by-frame on recorded .mp4 files
(or synthetic video pose streams) to measure:
  - Expected Reps vs Detected Reps
  - False Positives & Missed Reps
  - Exercise Classification Accuracy & Model Score
  - Frame Processing Latency (ms)
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional
import cv2

# Add parent directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pose.detector import PoseDetector
from pose.landmarks import PoseSequence
from recognition.fitness_pipeline import FitnessCVEngine


def evaluate_video_file(video_path: str,
                        expected_exercise: str,
                        expected_reps: int,
                        engine: FitnessCVEngine,
                        detector: PoseDetector) -> Dict:
    """Evaluates a single recorded video clip against ground truth rep count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: '{video_path}'")

    pose_seq = PoseSequence(max_len=45)
    latencies = []
    frame_count = 0
    valid_pose_frames = 0
    final_res = {}

    print(f"Evaluating '{os.path.basename(video_path)}' (Expected: {expected_exercise}, {expected_reps} reps)...")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_count += 1
        t0 = time.time()

        pf = detector.detect(frame)
        if pf is not None and len(pf.landmarks) >= 5:
            pose_seq.push(pf)
            valid_pose_frames += 1
        else:
            if (t0 - engine.last_valid_pose_time) > 0.50:
                pose_seq.clear()

        final_res = engine.process_frame_sequence(pose_seq.frames, now=t0)
        t_elapsed = (time.time() - t0) * 1000.0
        latencies.append(t_elapsed)

    cap.release()

    detected_reps = final_res.get("rep_count", 0)
    detected_ex = final_res.get("exercise_id", "none")
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    rep_diff = detected_reps - expected_reps
    false_positives = max(0, rep_diff)
    missed_reps = max(0, -rep_diff)

    return {
        "video": os.path.basename(video_path),
        "expected_exercise": expected_exercise,
        "detected_exercise": detected_ex,
        "exercise_correct": (detected_ex.lower() == expected_exercise.lower()),
        "expected_reps": expected_reps,
        "detected_reps": detected_reps,
        "false_positives": false_positives,
        "missed_reps": missed_reps,
        "total_frames": frame_count,
        "valid_pose_frames": valid_pose_frames,
        "avg_latency_ms": round(avg_latency, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fitness CV Pipeline on Recorded Videos")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--exercise", type=str, default="bicep_curl", help="Expected exercise ground truth")
    parser.add_argument("--reps", type=int, default=10, help="Expected rep count ground truth")
    parser.add_argument("--model-bundle", type=str, default="models/exercise_recognizer_bundle.joblib")
    parser.add_argument("--pose-model", type=str, default="models/pose_landmarker.task")
    args = parser.parse_args()

    if not args.video or not os.path.exists(args.video):
        print("Usage: python scripts/evaluate_real_videos.py --video path/to/clip.mp4 --exercise bicep_curl --reps 10")
        sys.exit(1)

    engine = FitnessCVEngine(bundle_path=args.model_bundle)
    detector = PoseDetector(model_path=args.pose_model)

    res = evaluate_video_file(args.video, args.exercise, args.reps, engine, detector)

    print("\n" + "=" * 50)
    print("  ONLINE PIPELINE EVALUATION REPORT")
    print("=" * 50)
    for k, v in res.items():
        print(f"  {k:25s}: {v}")
    print("=" * 50)

    detector.close()


if __name__ == "__main__":
    main()
