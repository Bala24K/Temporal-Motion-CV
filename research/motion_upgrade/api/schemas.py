"""
Pydantic schemas for the Motion ST-GCN inference API.

Mirrors the data structures produced by NeuralExerciseRecognizer and
HybridMovementEngine, including the richer outputs the learned model
provides: per-frame phases, rep segmentation, temporal attention,
abstention, and calibrated confidence.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Shared sub-models ────────────────────────────────────────────────


class LandmarkInput(BaseModel):
    """A single pose landmark matching pose.landmarks.Landmark."""
    x: float = Field(..., description="Normalized x-coordinate (0-1)")
    y: float = Field(..., description="Normalized y-coordinate (0-1)")
    z: float = Field(0.0, description="Relative depth estimate")
    visibility: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")


class PoseFrameInput(BaseModel):
    """One frame of pose landmarks, keyed by anatomical name."""
    landmarks: Dict[str, LandmarkInput] = Field(
        ...,
        description=(
            "Landmark dictionary keyed by anatomical name "
            "(e.g. 'left_shoulder', 'right_elbow'). "
            "At minimum include shoulders, elbows, wrists, hips, knees, ankles."
        ),
    )
    timestamp: Optional[float] = Field(
        None,
        description="Frame timestamp in seconds (optional, auto-generated if absent)",
    )


class FormFeedbackItem(BaseModel):
    """A single form feedback entry from the kinematic pipeline."""
    status: str = Field(..., description="'good' or 'warning'")
    message: str = Field(..., description="Human-readable feedback text")


class RepSegmentOutput(BaseModel):
    """A learned rep boundary from phase segmentation."""
    start_frame: int = Field(..., description="Start frame index of the rep")
    end_frame: int = Field(..., description="End frame index of the rep")
    peak_frame: int = Field(..., description="Frame index of peak contraction")


# ── Request schemas ──────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """POST /predict — classify an exercise from a pose sequence."""
    sequence: List[PoseFrameInput] = Field(
        ...,
        min_length=1,
        description="Ordered sequence of pose frames (temporal window)",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence_not_empty(cls, v: List[PoseFrameInput]) -> List[PoseFrameInput]:
        if not v:
            raise ValueError("Pose sequence must contain at least 1 frame")
        return v


class AnalyzeRequest(BaseModel):
    """POST /analyze — full pipeline analysis with phases, reps, form."""
    sequence: List[PoseFrameInput] = Field(
        ...,
        min_length=1,
        description="Ordered sequence of pose frames (temporal window)",
    )
    exercise_mode: Optional[str] = Field(
        None,
        description=(
            "Force a specific exercise mode instead of auto-detection. "
            "Options: 'bicep_curl', 'squat', 'shoulder_press', 'lateral_raise'"
        ),
    )


# ── Response schemas ─────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = Field(..., description="Service status: 'ok', 'degraded', or 'error'")
    model_loaded: bool = Field(..., description="Whether the ST-GCN model is loaded and ready")
    model_name: Optional[str] = Field(None, description="Model architecture name")
    model_path: Optional[str] = Field(None, description="Path to the checkpoint file")


class ModelInfoResponse(BaseModel):
    """GET /model response."""
    model_name: str = Field(..., description="Model architecture name")
    model_path: str = Field(..., description="Path to the checkpoint")
    classes: List[str] = Field(..., description="Supported exercise class labels")
    streams: List[str] = Field(..., description="Input stream types")
    window_frames: int = Field(..., description="Temporal window size")
    temperature: float = Field(..., description="Calibration temperature")
    min_confidence: float = Field(..., description="Abstention confidence threshold")
    device: str = Field(..., description="Inference device")
    metadata: Dict[str, Any] = Field(..., description="Checkpoint metadata")


class PredictResponse(BaseModel):
    """POST /predict response."""
    exercise: str = Field(..., alias="class", description="Predicted exercise class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence")
    probabilities: Dict[str, float] = Field(
        ..., description="Per-class probability distribution (temperature-scaled)"
    )
    frame_count: int = Field(..., description="Number of frames in the input sequence")
    valid_pose_ratio: float = Field(..., description="Fraction of frames with valid pose data")
    abstained: bool = Field(..., description="Whether the model abstained due to low confidence")
    latency_ms: float = Field(..., description="Total inference latency in milliseconds")

    model_config = {"populate_by_name": True}


class AnalyzeResponse(BaseModel):
    """POST /analyze response — full hybrid engine analysis."""
    exercise: str = Field(..., description="Detected/forced exercise name")
    exercise_id: str = Field(..., description="Internal exercise identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence")
    confidence_is_calibrated: bool = Field(..., description="Whether confidence is temperature-scaled")
    abstained: bool = Field(..., description="Whether the model abstained")
    rep_count: int = Field(..., description="Authoritative repetition count")
    learned_rep_count: int = Field(..., description="Rep count from learned phase segmentation")
    state_machine_rep_count: int = Field(..., description="Rep count from kinematic state machine")
    rep_count_agreement: bool = Field(..., description="Whether learned and state machine agree")
    phase: str = Field(..., description="Current movement phase (kinematic)")
    learned_phase: Optional[str] = Field(None, description="Learned phase from ST-GCN")
    phase_sequence: Optional[List[str]] = Field(None, description="Per-frame phase labels")
    current_angle: Optional[float] = Field(None, description="Primary joint angle (degrees)")
    rom: Optional[float] = Field(None, description="Range of motion (degrees)")
    tempo: Optional[float] = Field(None, description="Movement tempo")
    form_feedback: List[FormFeedbackItem] = Field(
        ..., description="Structured form analysis feedback"
    )
    valid_pose: bool = Field(..., description="Whether the pose was valid for analysis")
    probabilities: Dict[str, float] = Field(
        ..., description="Per-class probability distribution"
    )
    reps: Optional[List[RepSegmentOutput]] = Field(None, description="Learned rep segments")
    latency_ms: float = Field(..., description="Total analysis latency in milliseconds")
    timing: Dict[str, float] = Field(
        ..., description="Breakdown of preprocessing and inference time"
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""
    error: str = Field(..., description="Error type/code")
    detail: str = Field(..., description="Human-readable error description")
