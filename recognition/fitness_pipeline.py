"""
Unified Real-Time Fitness-CV Engine tying together:
  1. Exercise Recognition (ExerciseRecognizer)
  2. Exercise-Specific Rep Counting (RepStateMachine / ExerciseAnalyzers)
  3. Real-Time Form Analysis (FormThresholds & Biomechanical Rules)
"""

from collections import Counter, deque
import os
import time
from typing import Dict, List, Optional
import numpy as np

from pose.landmarks import PoseFrame, is_valid_person_pose
from recognition.exercise_recognizer import ExerciseRecognizer, ExerciseRecognitionResult
from exercises.bicep_curl import BicepCurlAnalyzer
from exercises.squat import SquatAnalyzer
from exercises.shoulder_press import ShoulderPressAnalyzer
from exercises.lateral_raise import LateralRaiseAnalyzer
from exercises.base import ExerciseAnalyzer, ok, warn


class FitnessCVEngine:
    """
    Complete Fitness-CV Engine providing real-time exercise classification,
    repetition counting, movement phase tracking, and form feedback.
    """

    MIN_WARMUP_FRAMES = 5
    CONSENSUS_WINDOW = 5
    REQUIRED_CONSENSUS_COUNT = 3
    MIN_CONSENSUS_SCORE = 0.55
    TRACKING_LOSS_TIMEOUT_SEC = 0.50

    def __init__(self, bundle_path: Optional[str] = "models/exercise_recognizer_bundle.joblib",
                 recognizer: Optional[ExerciseRecognizer] = None):
        if recognizer is not None:
            self.recognizer = recognizer
        elif bundle_path and os.path.exists(bundle_path):
            self.recognizer = ExerciseRecognizer.load_bundle(bundle_path)
        else:
            self.recognizer = None

        self.analyzers: Dict[str, ExerciseAnalyzer] = {
            "bicep_curl": BicepCurlAnalyzer(),
            "squat": SquatAnalyzer(),
            "shoulder_press": ShoulderPressAnalyzer(),
            "lateral_raise": LateralRaiseAnalyzer(),
        }
        self.active_exercise: Optional[str] = None
        self.prediction_history: deque = deque(maxlen=self.CONSENSUS_WINDOW)
        self.last_valid_pose_time: float = 0.0

    def process_frame_sequence(self, pose_frames: List[PoseFrame], now: Optional[float] = None) -> Dict:
        """
        Processes a rolling window of PoseFrames to classify exercise, count reps,
        and generate structured form analysis feedback.
        """
        now = now if now is not None else time.time()

        # 1. Empty or invalid tracking check
        if not pose_frames:
            self.prediction_history.clear()
            return self._tracking_lost_output("No person detected in frame")

        latest_frame = pose_frames[-1]

        # 2. Robust pose validity check
        current_ex_id = self.active_exercise or "bicep_curl"
        pose_ok = is_valid_person_pose(latest_frame, exercise_id=current_ex_id, min_visibility=0.05)

        if pose_ok:
            self.last_valid_pose_time = now
        else:
            # Grace period check (500ms)
            if self.last_valid_pose_time > 0 and (now - self.last_valid_pose_time) > self.TRACKING_LOSS_TIMEOUT_SEC:
                self.prediction_history.clear()
                return self._tracking_lost_output("Person tracking lost")

        # 3. Warmup requirement check
        if len(pose_frames) < self.MIN_WARMUP_FRAMES:
            return {
                "exercise": "ANALYZING...",
                "exercise_id": "analyzing",
                "confidence": 0.0,
                "model_score": 0.0,
                "rep_count": 0,
                "phase": "CALIBRATING",
                "current_angle": None,
                "rom": None,
                "tempo": None,
                "form_feedback": [{"status": "good", "message": "Collecting movement history..."}],
                "valid_pose": True,
            }

        # 4. Temporal Exercise Classification & Consensus
        model_score = 1.0
        if self.recognizer is not None:
            rec_res: ExerciseRecognitionResult = self.recognizer.predict_sequence(pose_frames)
            model_score = float(rec_res.confidence)
            self.prediction_history.append((rec_res.exercise, model_score))

            # Evaluate rolling consensus across last CONSENSUS_WINDOW predictions
            if len(self.prediction_history) >= self.CONSENSUS_WINDOW:
                ex_counts = Counter([ex for ex, score in self.prediction_history])
                top_ex, top_count = ex_counts.most_common(1)[0]
                top_scores = [score for ex, score in self.prediction_history if ex == top_ex]
                avg_score = sum(top_scores) / len(top_scores)

                if top_count >= self.REQUIRED_CONSENSUS_COUNT and avg_score >= self.MIN_CONSENSUS_SCORE:
                    if self.active_exercise != top_ex:
                        self.active_exercise = top_ex
        else:
            if self.active_exercise is None:
                self.active_exercise = "bicep_curl"

        target_ex = self.active_exercise or "bicep_curl"
        analyzer = self.analyzers.get(target_ex, self.analyzers["bicep_curl"])

        # 5. Process latest frame through active exercise analyzer
        analysis_res = analyzer.process(latest_frame, now=now)

        # 6. Format structured form feedback output
        raw_flags = analysis_res.get("form_flags", [])
        structured_feedback = []
        for flag in raw_flags:
            if isinstance(flag, dict):
                lvl = flag.get("level", "ok")
                msg = flag.get("message", "")
                status = "good" if lvl == "ok" else "warning"
                structured_feedback.append({"status": status, "message": msg})

        if not structured_feedback:
            structured_feedback.append({"status": "good", "message": "Controlled movement"})

        # Calibrate display model score (cap raw RF probability)
        calibrated_score = min(0.96, model_score) if model_score > 0.85 else model_score

        return {
            "exercise": target_ex.replace("_", " ").title(),
            "exercise_id": target_ex,
            "confidence": float(calibrated_score),
            "model_score": float(calibrated_score),
            "rep_count": analysis_res.get("rep_count", 0),
            "phase": analysis_res.get("phase", "MOVING"),
            "current_angle": analysis_res.get("current_angle"),
            "rom": analysis_res.get("rom"),
            "tempo": analysis_res.get("tempo"),
            "form_feedback": structured_feedback,
            "valid_pose": pose_ok,
            "just_completed_rep": analysis_res.get("just_completed_rep"),
        }

    def _tracking_lost_output(self, message: str) -> Dict:
        return {
            "exercise": "NO PERSON DETECTED",
            "exercise_id": "none",
            "confidence": 0.0,
            "model_score": 0.0,
            "rep_count": 0,
            "phase": "TRACKING LOST",
            "current_angle": None,
            "rom": None,
            "tempo": None,
            "form_feedback": [{"status": "warning", "message": message}],
            "valid_pose": False,
        }

    def reset_state(self):
        """Resets rep counts, prediction history, and state machines."""
        for analyzer in self.analyzers.values():
            analyzer.reset()
        self.active_exercise = None
        self.prediction_history.clear()
        self.last_valid_pose_time = 0.0

    def reset(self):
        self.reset_state()
