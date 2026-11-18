"""
Inference adapter: the learned model behind the interface the rest of the
application already speaks.

`recognition.fitness_pipeline.FitnessCVEngine` accepts any object with a
`predict_sequence(pose_frames) -> result` method exposing `.exercise`,
`.confidence` and `.probabilities`. Matching that interface here means the
UI, the exercise analyzers, the rep state machines, the session recorder
and their tests keep working untouched:

    engine = FitnessCVEngine(recognizer=NeuralExerciseRecognizer.load("models/motion_stgcn.pt"))

Three things this does that the Random Forest path did not:

  CALIBRATED CONFIDENCE
      The displayed number is the temperature-scaled probability fitted
      on the validation split, not `min(0.96, raw_score)`. It can be
      checked against outcomes.

  ABSTENTION
      Below `min_confidence`, or when the pose was mostly interpolated,
      the result is `other` rather than a forced choice among four
      exercises. A live system that confidently calls someone scratching
      their head a shoulder press loses trust faster than one that says
      it does not know.

  PHASES AND REPS
      The same forward pass yields per-frame phase probabilities, so
      `predict_full` returns learned rep boundaries alongside the
      exercise. The deterministic state machine remains available as the
      fallback and as the baseline it is measured against.
"""
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from pose.landmarks import PoseFrame
from motion.calibration import apply_temperature
from motion.model import MotionModelConfig, build_model
from motion.phase import NUM_PHASES, PHASE_NAMES, RepSegment, segment_reps_from_phases
from motion.tensors import STREAM_NAMES, build_streams, build_window

try:  # keep isinstance compatibility with the existing pipeline where possible
    from recognition.exercise_recognizer import ExerciseRecognitionResult
except Exception:  # pragma: no cover - only when the legacy path can't import
    @dataclass
    class ExerciseRecognitionResult:  # type: ignore[no-redef]
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


@dataclass
class MotionAnalysisResult:
    """The full output of one forward pass, for callers that want more
    than the top label."""
    exercise: str
    confidence: float
    probabilities: Dict[str, float]
    phase_probabilities: np.ndarray        # (T, NUM_PHASES)
    phase_sequence: List[str]              # argmax phase per frame
    reps: List[RepSegment]
    attention: np.ndarray                  # (T_feat,) temporal saliency
    valid_pose_ratio: float
    abstained: bool

    @property
    def rep_count(self) -> int:
        return len(self.reps)


class NeuralExerciseRecognizer:
    """
    Wraps a trained checkpoint. Construct via `load`; the checkpoint
    carries its own architecture config, class list, stream selection and
    calibration temperature, so a model file cannot be loaded with
    mismatched inference settings.
    """

    def __init__(
        self,
        model,
        classes: Sequence[str],
        streams: Sequence[str],
        window_frames: int,
        temperature: float = 1.0,
        include_validity_channel: bool = True,
        min_confidence: float = 0.45,
        min_valid_ratio: float = 0.35,
        other_class: str = "other",
        device: str = "cpu",
        metadata: Optional[Dict] = None,
    ):
        self.model = model
        self.classes = list(classes)
        self.streams = tuple(streams)
        self.window_frames = int(window_frames)
        self.temperature = float(temperature)
        self.include_validity_channel = bool(include_validity_channel)
        self.min_confidence = float(min_confidence)
        self.min_valid_ratio = float(min_valid_ratio)
        self.other_class = other_class
        self.device = device
        self.metadata = metadata or {}
        self.model_name = "MotionSTGCN"

    @classmethod
    def load(cls, path: str, device: str = "cpu", **overrides) -> "NeuralExerciseRecognizer":
        import torch

        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path!r}")
        ckpt = torch.load(path, map_location=device, weights_only=False)

        cfg = MotionModelConfig.from_dict(ckpt["config"])
        model = build_model(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()

        if not ckpt.get("release_ready", False):
            # Loud but non-fatal: a development checkpoint should be
            # usable in a demo, just never mistaken for a validated one.
            prov = ckpt.get("label_provenance", {})
            print(
                f"[NeuralExerciseRecognizer] loaded a development checkpoint "
                f"({path}): {prov.get('note', 'label provenance unrecorded')}"
            )

        return cls(
            model=model,
            classes=ckpt["classes"],
            streams=ckpt.get("streams", STREAM_NAMES),
            window_frames=ckpt.get("window_frames", 48),
            temperature=ckpt.get("temperature", 1.0),
            include_validity_channel=cfg.include_validity_channel,
            device=device,
            metadata={
                "label_provenance": ckpt.get("label_provenance", {}),
                "metrics": ckpt.get("metrics", {}),
                "release_ready": ckpt.get("release_ready", False),
            },
            **overrides,
        )

    def _empty_result(self, n_frames: int) -> ExerciseRecognitionResult:
        uniform = 1.0 / max(1, len(self.classes))
        return ExerciseRecognitionResult(
            exercise=self.other_class if self.other_class in self.classes else self.classes[0],
            confidence=0.0,
            probabilities={c: uniform for c in self.classes},
            frame_count=n_frames,
            valid_pose_ratio=0.0,
        )

    def analyze(self, pose_frames: Sequence[PoseFrame]) -> Optional[MotionAnalysisResult]:
        """
        One forward pass over a window of pose frames.

        Returns None when the window cannot be built at all — no frame had
        the landmarks needed for normalisation. Callers should treat that
        as "no person", not as a low-confidence prediction.
        """
        import torch

        if not pose_frames:
            return None

        window = build_window(pose_frames, target_len=self.window_frames)
        if window is None:
            return None

        x = build_streams(window, self.streams, self.include_validity_channel)
        tensor = torch.from_numpy(x[None, ...]).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        logits = out["exercise_logits"].cpu().numpy()
        probs = apply_temperature(logits, self.temperature)[0]
        phase_logits = out["phase_logits"].cpu().numpy()[0]
        attention = out["attention"].cpu().numpy()[0]

        exp = np.exp(phase_logits - phase_logits.max(axis=1, keepdims=True))
        phase_probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

        prob_dict = {c: float(p) for c, p in zip(self.classes, probs)}
        top = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top]

        abstained = confidence < self.min_confidence or window.valid_ratio < self.min_valid_ratio
        if abstained and self.other_class in self.classes:
            top = self.other_class

        return MotionAnalysisResult(
            exercise=top,
            confidence=confidence,
            probabilities=prob_dict,
            phase_probabilities=phase_probs,
            phase_sequence=[PHASE_NAMES[i] for i in phase_probs.argmax(axis=1)],
            reps=segment_reps_from_phases(phase_probs),
            attention=attention,
            valid_pose_ratio=window.valid_ratio,
            abstained=abstained,
        )

    def predict_sequence(self, pose_frames: Sequence[PoseFrame]) -> ExerciseRecognitionResult:
        """Drop-in replacement for `ExerciseRecognizer.predict_sequence`."""
        result = self.analyze(pose_frames)
        if result is None:
            return self._empty_result(len(pose_frames))
        return ExerciseRecognitionResult(
            exercise=result.exercise,
            confidence=float(result.confidence),
            probabilities=result.probabilities,
            frame_count=len(pose_frames),
            valid_pose_ratio=float(result.valid_pose_ratio),
        )

    def predict_full(self, pose_frames: Sequence[PoseFrame]) -> Optional[MotionAnalysisResult]:
        """Alias for `analyze`, named to read clearly at call sites that
        want phases and reps as well as the label."""
        return self.analyze(pose_frames)

    def warmup(self, n_frames: Optional[int] = None) -> float:
        """
        Run one forward pass on zeros to pay the lazy-initialisation cost
        up front, returning the elapsed seconds.

        Worth calling before the camera starts: the first inference in a
        fresh process is routinely an order of magnitude slower than the
        rest, and paid during the first live frames it looks like a stall.
        """
        import time
        import torch

        n = n_frames or self.window_frames
        from motion.tensors import stream_channel_count
        c = stream_channel_count(self.streams, self.include_validity_channel)
        dummy = torch.zeros(1, c, n, self.model.cfg.num_joints, device=self.device)
        start = time.perf_counter()
        with torch.no_grad():
            self.model(dummy)
        return time.perf_counter() - start
