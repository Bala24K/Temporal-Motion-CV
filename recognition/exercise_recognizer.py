"""
Production-ready Exercise Recognizer Inference API and Model Serialization Bundle.

Pipeline flow:
  Video / Camera Feed
      ↓
  MediaPipe Pose Detector
      ↓
  Landmark Normalization (Mid-hip center & torso scaling)
      ↓
  Temporal Feature Extraction (FeatureConfig modular groups)
      ↓
  Exercise Classifier (Random Forest / Logistic Regression / MLP / 1D-CNN)
      ↓
  Exercise Recognition Result (exercise, confidence, probabilities)

Includes full bundle serialization (model, scaler, feature config, classes, seed).
"""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np

from pose.detector import PoseDetector
from pose.landmarks import PoseFrame
from recognition.feature_engineering import (
    FeatureConfig,
    extract_modular_sequence_features,
    extract_modular_tabular_features,
)
from recognition.qevd_dataset import (
    extract_video_pose_data,
    load_qevd_dataset,
    train_val_test_split_by_video,
    EXERCISES,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class ExerciseRecognitionResult:
    """Encapsulates exercise classification outcome and confidence scores."""
    exercise: str
    confidence: float
    probabilities: Dict[str, float]
    frame_count: int
    valid_pose_ratio: float

    def to_dict(self) -> Dict:
        return {
            "exercise": self.exercise,
            "confidence": float(self.confidence),
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            "frame_count": int(self.frame_count),
            "valid_pose_ratio": float(self.valid_pose_ratio),
        }


class ExerciseRecognizer:
    """
    Reusable, self-contained Exercise Recognizer.
    Can be instantiated directly or loaded from a saved model bundle file (.joblib).
    """

    def __init__(self,
                 model: object,
                 classes: Optional[List[str]] = None,
                 feature_config: Optional[FeatureConfig] = None,
                 scaler: Optional[StandardScaler] = None,
                 seed: int = 42,
                 model_name: str = "RandomForest"):
        self.model = model
        self.classes = classes if classes is not None else list(EXERCISES)
        self.feature_config = feature_config if feature_config is not None else FeatureConfig()
        self.scaler = scaler
        self.seed = seed
        self.model_name = model_name

    def predict_sequence(self, pose_frames: List[PoseFrame]) -> ExerciseRecognitionResult:
        """
        Runs exercise recognition directly on a sequence of PoseFrames.
        """
        if not pose_frames:
            # Fallback for empty sequence
            uniform_prob = 1.0 / max(1, len(self.classes))
            return ExerciseRecognitionResult(
                exercise=self.classes[0],
                confidence=0.0,
                probabilities={c: uniform_prob for c in self.classes},
                frame_count=0,
                valid_pose_ratio=0.0,
            )

        f_tab = extract_modular_tabular_features(pose_frames, self.feature_config).reshape(1, -1)

        if self.scaler is not None:
            f_tab = self.scaler.transform(f_tab)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(f_tab)[0]
        else:
            pred = self.model.predict(f_tab)[0]
            proba = np.array([1.0 if c == pred else 0.0 for c in self.classes])

        # Match class labels
        if hasattr(self.model, "classes_"):
            model_classes = list(self.model.classes_)
            prob_dict = {c: float(proba[model_classes.index(c)]) if c in model_classes else 0.0 for c in self.classes}
        else:
            prob_dict = {c: float(p) for c, p in zip(self.classes, proba)}

        top_exercise = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top_exercise]

        return ExerciseRecognitionResult(
            exercise=top_exercise,
            confidence=confidence,
            probabilities=prob_dict,
            frame_count=len(pose_frames),
            valid_pose_ratio=1.0,
        )

    def predict_exercise(self, video_path: str,
                         detector: Optional[PoseDetector] = None,
                         model_path: str = "models/pose_landmarker.task") -> ExerciseRecognitionResult:
        """
        Runs full end-to-end exercise recognition on a video file.
        """
        close_detector = False
        if detector is None:
            detector = PoseDetector(model_path=model_path)
            close_detector = True

        res = extract_video_pose_data(video_path, detector)
        if close_detector:
            detector.close()

        if res is None:
            uniform_prob = 1.0 / max(1, len(self.classes))
            return ExerciseRecognitionResult(
                exercise=self.classes[0],
                confidence=0.0,
                probabilities={c: uniform_prob for c in self.classes},
                frame_count=0,
                valid_pose_ratio=0.0,
            )

        pose_frames, _ = res
        return self.predict_sequence(pose_frames)

    def save_bundle(self, filepath: str):
        """
        Serializes the complete recognizer state into a single .joblib bundle file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            "model": self.model,
            "classes": self.classes,
            "feature_config": self.feature_config,
            "scaler": self.scaler,
            "seed": self.seed,
            "model_name": self.model_name,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load_bundle(cls, filepath: str) -> "ExerciseRecognizer":
        """
        Deserializes a saved model bundle file into an ExerciseRecognizer instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model bundle file '{filepath}' not found.")
        bundle = joblib.load(filepath)
        return cls(
            model=bundle["model"],
            classes=bundle.get("classes", list(EXERCISES)),
            feature_config=bundle.get("feature_config", FeatureConfig()),
            scaler=bundle.get("scaler"),
            seed=bundle.get("seed", 42),
            model_name=bundle.get("model_name", "RandomForest"),
        )


def train_and_save_recognizer(dataset_dir: str = "QEVD-FIT-300k-Part-1",
                              target_per_class: int = 25,
                              output_bundle_path: str = "models/exercise_recognizer_bundle.joblib",
                              seed: int = 42) -> ExerciseRecognizer:
    """
    Trains a Random Forest classifier on balanced QEVD data and saves the recognizer bundle.
    """
    cfg = FeatureConfig(use_raw_landmarks=True, use_joint_angles=True, use_velocity=True,
                        use_acceleration=True, use_symmetry=True, use_range_of_motion=True)
    full_ds = load_qevd_dataset(dataset_dir=dataset_dir, target_per_class=target_per_class, feature_config=cfg)
    train_data, val_data, test_data = train_val_test_split_by_video(full_ds, seed=seed)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(train_data["X_tab"])

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    model.fit(X_tr_scaled, train_data["y"])

    recognizer = ExerciseRecognizer(
        model=model,
        classes=list(EXERCISES),
        feature_config=cfg,
        scaler=scaler,
        seed=seed,
        model_name="RandomForest",
    )
    recognizer.save_bundle(output_bundle_path)
    return recognizer
