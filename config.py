"""
Centralized Application Configuration for Fitness CV.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Configuration settings for camera, pipeline, UI, and recording."""

    camera_id: int = 0
    video_file: Optional[str] = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    horizontal_flip: bool = True
    exercise_mode: str = "AUTO"  # "AUTO", "BICEP CURL", "SQUAT", "SHOULDER PRESS", "LATERAL RAISE"
    debug_overlay: bool = False
    debug_signals: bool = False
    record_session: bool = False
    output_dir: str = "sessions"
    recordings_dir: str = "recordings"
    logs_dir: str = "logs"
    model_bundle_path: str = "models/exercise_recognizer_bundle.joblib"
    pose_model_path: str = "models/pose_landmarker.task"

    EXERCISE_OPTIONS = [
        "AUTO",
        "BICEP CURL",
        "SQUAT",
        "SHOULDER PRESS",
        "LATERAL RAISE",
    ]

    @property
    def is_video_source(self) -> bool:
        return self.video_file is not None and len(self.video_file.strip()) > 0
