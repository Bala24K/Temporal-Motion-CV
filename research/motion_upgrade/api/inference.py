"""
Inference service — thin wrapper calling the EXISTING motion-upgrade pipeline.

This module does NOT reimplement any preprocessing, feature extraction,
model loading, or prediction logic. It constructs the same PoseFrame
objects the local application uses and passes them to:

  - NeuralExerciseRecognizer.predict_sequence()  →  /predict
  - NeuralExerciseRecognizer.analyze()            →  /analyze (neural branch)
  - HybridMovementEngine                          →  /analyze (fused kinematic + neural)

local inference == API inference  (for equivalent inputs)
"""

import logging
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from pose.landmarks import Landmark, PoseFrame
from motion.recognizer import NeuralExerciseRecognizer, MotionAnalysisResult
from motion.fusion import HybridMovementEngine, FusionConfig

from api.schemas import LandmarkInput, PoseFrameInput

logger = logging.getLogger("api.inference")


# ── Conversion helpers ───────────────────────────────────────────────


def _to_landmark(lm_input: LandmarkInput) -> Landmark:
    """Converts an API LandmarkInput to the existing Landmark dataclass."""
    return Landmark(
        x=lm_input.x,
        y=lm_input.y,
        z=lm_input.z,
        visibility=lm_input.visibility,
    )


def to_pose_frames(frames_input: Sequence[PoseFrameInput]) -> List[PoseFrame]:
    """Converts a sequence of API PoseFrameInput to the internal PoseFrame list.

    This is the ONLY conversion layer between the API request format and
    the existing ML pipeline's internal representation.
    """
    pose_frames: List[PoseFrame] = []
    for i, frame_in in enumerate(frames_input):
        pf = PoseFrame()
        pf.landmarks = {
            name: _to_landmark(lm) for name, lm in frame_in.landmarks.items()
        }
        pf.timestamp = frame_in.timestamp if frame_in.timestamp is not None else float(i) / 7.5
        pose_frames.append(pf)
    return pose_frames


# ── Inference service class ──────────────────────────────────────────


class InferenceService:
    """Stateless inference facade wrapping the motion-upgrade ML pipeline.

    Holds a loaded NeuralExerciseRecognizer (for /predict) and creates
    a HybridMovementEngine per /analyze request (to avoid cross-request
    state bleed from the kinematic state machines).
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self._model_path = model_path
        self._device = device
        self._recognizer: Optional[NeuralExerciseRecognizer] = None
        self._load_error: Optional[str] = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Loads ST-GCN checkpoint from disk. Called once at startup."""
        try:
            logger.info("Loading ST-GCN checkpoint from '%s' (device=%s)...", self._model_path, self._device)
            self._recognizer = NeuralExerciseRecognizer.load(self._model_path, device=self._device)

            # Warmup: pay the lazy init cost upfront
            warmup_ms = self._recognizer.warmup() * 1000.0
            logger.info(
                "Model loaded: %s (%d classes: %s) | warmup: %.1f ms",
                self._recognizer.model_name,
                len(self._recognizer.classes),
                ", ".join(self._recognizer.classes),
                warmup_ms,
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Failed to load model: %s", exc)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ── Model info ───────────────────────────────────────────────────

    def get_model_info(self) -> Dict:
        """Returns metadata about the loaded model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        rec = self._recognizer
        return {
            "model_name": rec.model_name,
            "model_path": self._model_path,
            "classes": rec.classes,
            "streams": list(rec.streams),
            "window_frames": rec.window_frames,
            "temperature": rec.temperature,
            "min_confidence": rec.min_confidence,
            "device": rec.device,
            "metadata": rec.metadata,
        }

    # ── Prediction (NeuralExerciseRecognizer.predict_sequence) ───────

    def predict(self, frames_input: Sequence[PoseFrameInput]) -> Dict:
        """Runs exercise classification via the ST-GCN model.

        Uses the SAME NeuralExerciseRecognizer.predict_sequence() and
        NeuralExerciseRecognizer.analyze() the local pipeline uses.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        t_start = time.perf_counter()

        # Convert API input -> internal PoseFrame list
        pose_frames = to_pose_frames(frames_input)
        t_preprocess = time.perf_counter()

        # Call the EXISTING inference pipeline — full analyze for richer output
        analysis = self._recognizer.analyze(pose_frames)
        t_inference = time.perf_counter()

        total_ms = (t_inference - t_start) * 1000.0

        if analysis is None:
            # No valid window could be built
            return {
                "class": "unknown",
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self._recognizer.classes},
                "frame_count": len(pose_frames),
                "valid_pose_ratio": 0.0,
                "abstained": True,
                "latency_ms": round(total_ms, 2),
            }

        return {
            "class": analysis.exercise,
            "confidence": float(analysis.confidence),
            "probabilities": {k: float(v) for k, v in analysis.probabilities.items()},
            "frame_count": len(pose_frames),
            "valid_pose_ratio": float(analysis.valid_pose_ratio),
            "abstained": analysis.abstained,
            "latency_ms": round(total_ms, 2),
        }

    # ── Full analysis (HybridMovementEngine path) ────────────────────

    def analyze(
        self,
        frames_input: Sequence[PoseFrameInput],
        exercise_mode: Optional[str] = None,
    ) -> Dict:
        """Runs the full hybrid fusion pipeline.

        Creates a fresh HybridMovementEngine per request, feeds every
        pose frame through update_pose(), and feeds the neural recognition
        through update_recognition(), then returns the fused state().

        This mirrors what the live pipeline does across its threads, but
        serialised into a single request.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        t_start = time.perf_counter()

        # Convert API input -> internal PoseFrame list
        pose_frames = to_pose_frames(frames_input)
        t_preprocess = time.perf_counter()

        # Fresh engine per request — avoids cross-request state bleed
        engine = HybridMovementEngine(config=FusionConfig())

        # Apply exercise mode override
        if exercise_mode:
            mapped_ex = exercise_mode.lower().replace(" ", "_")
            engine.lock_exercise(mapped_ex)

        # Run neural recognition on the full window
        analysis = self._recognizer.analyze(pose_frames)

        # Feed recognition result into the fusion engine
        engine.update_recognition(analysis)

        # Feed each pose frame through the kinematic analyzers
        for pf in pose_frames:
            engine.update_pose(pf, now=pf.timestamp)

        # Get the fused state
        now = pose_frames[-1].timestamp if pose_frames else time.time()
        state = engine.state(now=now)
        t_inference = time.perf_counter()

        preprocess_ms = (t_preprocess - t_start) * 1000.0
        inference_ms = (t_inference - t_preprocess) * 1000.0
        total_ms = (t_inference - t_start) * 1000.0

        # Build rep segments from the neural analysis
        reps_out = None
        phase_seq = None
        if analysis is not None:
            reps_out = [
                {"start_frame": r.start, "end_frame": r.end, "peak_frame": r.peak}
                for r in analysis.reps
            ]
            phase_seq = analysis.phase_sequence

        return {
            "exercise": state.get("exercise", ""),
            "exercise_id": state.get("exercise_id", ""),
            "confidence": float(state.get("confidence", 0.0)),
            "confidence_is_calibrated": state.get("confidence_is_calibrated", False),
            "abstained": state.get("abstained", False),
            "rep_count": int(state.get("rep_count", 0)),
            "learned_rep_count": int(state.get("learned_rep_count", 0)),
            "state_machine_rep_count": int(state.get("state_machine_rep_count", 0)),
            "rep_count_agreement": state.get("rep_count_agreement", True),
            "phase": state.get("phase", ""),
            "learned_phase": state.get("learned_phase"),
            "phase_sequence": phase_seq,
            "current_angle": state.get("current_angle"),
            "rom": state.get("rom"),
            "tempo": state.get("tempo"),
            "form_feedback": state.get("form_feedback", []),
            "valid_pose": state.get("valid_pose", False),
            "probabilities": state.get("probabilities", {}),
            "reps": reps_out,
            "latency_ms": round(total_ms, 2),
            "timing": {
                "preprocessing_ms": round(preprocess_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }
