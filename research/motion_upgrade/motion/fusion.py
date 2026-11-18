"""
The fusion layer: learned recognition plus deterministic kinematics.

    neural branch          kinematic branch
    -------------          ----------------
    what movement is this  how well was it performed
    phase probabilities    ROM, symmetry, tempo, alignment
    rep boundaries         rep boundaries (state machine)
           \\                    /
            +------ fused -----+
                     |
             what the user sees

The division of labour is deliberate and not a hedge. The network is
better at "this is a squat" because that is a pattern-recognition problem
over a whole movement. The kinematic layer is better at "your left knee
travelled 8 degrees further than your right" because that is a
measurement, and a measurement should be measured, not inferred. A
network asked to output a knee angle gives you a number with no error
bars and no way to explain itself.

REP COUNTING IS REPORTED TWICE
    The state machine remains authoritative by default. It is causal —
    it decides at the moment the rep completes, from the current frame —
    whereas the learned segmenter looks at a whole window and can revise
    the recent past, which on screen means a rep counter that goes
    backwards. The learned count is reported alongside, and the
    disagreement between them is exposed as `rep_count_agreement`. That
    disagreement is the experiment: when it closes on real data, the
    learned counter can take over by flipping `rep_source`.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from exercises.base import ExerciseAnalyzer
from exercises.bicep_curl import BicepCurlAnalyzer
from exercises.lateral_raise import LateralRaiseAnalyzer
from exercises.shoulder_press import ShoulderPressAnalyzer
from exercises.squat import SquatAnalyzer
from pose.landmarks import PoseFrame, is_valid_person_pose


@dataclass
class FusionConfig:
    """
    Consensus and hysteresis settings.

    The defaults mirror `FitnessCVEngine` so behaviour is comparable
    across the two recognisers rather than confounded by different
    smoothing.
    """
    consensus_window: int = 5
    required_consensus: int = 3
    min_consensus_confidence: float = 0.55
    tracking_loss_timeout: float = 0.5
    rep_source: str = "state_machine"   # 'state_machine' | 'learned'
    default_exercise: str = "bicep_curl"


class HybridMovementEngine:
    """
    Consumes the two asynchronous streams the pipeline produces — a pose
    frame at pose rate, a recognition result at inference rate — and
    maintains one coherent view of the session.

    Wire it to `AsyncMotionPipeline`:

        engine = HybridMovementEngine()
        pipeline = AsyncMotionPipeline(
            source=camera, detector=detector, recognizer=recognizer,
            on_pose=engine.update_pose,
            on_recognition=engine.update_recognition,
        )

    Thread safety: `update_pose` runs on the pose thread and
    `update_recognition` on the inference thread, so the shared fields
    they touch are the active exercise and the last results. Those are
    single assignments of immutable values; the analyzers themselves are
    only ever touched by the pose thread.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.analyzers: Dict[str, ExerciseAnalyzer] = {
            "bicep_curl": BicepCurlAnalyzer(),
            "squat": SquatAnalyzer(),
            "shoulder_press": ShoulderPressAnalyzer(),
            "lateral_raise": LateralRaiseAnalyzer(),
        }
        self.active_exercise: Optional[str] = None
        self.locked_exercise: Optional[str] = None   # set by a manual UI override

        self._history: Deque[Tuple[str, float]] = deque(maxlen=self.config.consensus_window)
        self._last_recognition: Optional[Any] = None
        self._last_kinematics: Dict[str, Any] = {}
        self._last_pose_time: float = 0.0
        self._switch_count: int = 0

    # ------------------------------------------------------- input: per frame

    def update_pose(self, frame: PoseFrame, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the active exercise's analyzer on one frame. Called at pose
        rate; must stay cheap, which it is — this is arithmetic on 15
        points, not a model.
        """
        now = now if now is not None else time.time()
        target = self.locked_exercise or self.active_exercise or self.config.default_exercise
        analyzer = self.analyzers.get(target, self.analyzers[self.config.default_exercise])

        pose_ok = is_valid_person_pose(frame, exercise_id=target, min_visibility=0.05)
        if pose_ok:
            self._last_pose_time = now

        result = analyzer.process(frame, now=now)
        result["valid_pose"] = pose_ok
        result["exercise_id"] = target
        self._last_kinematics = result
        return result

    # ------------------------------------------------- input: per recognition

    def update_recognition(self, result: Any) -> None:
        """
        Fold one window-level recognition into the consensus.

        An abstention — the model below its confidence floor, or the pose
        mostly interpolated — is recorded but never switches the active
        exercise. Switching on a low-confidence frame is how a system ends
        up flipping between squat and press while someone stands still.
        """
        self._last_recognition = result
        if result is None:
            return

        abstained = getattr(result, "abstained", False)
        exercise = getattr(result, "exercise", None)
        confidence = float(getattr(result, "confidence", 0.0))

        if abstained or exercise not in self.analyzers:
            self._history.append(("__abstain__", confidence))
            return

        self._history.append((exercise, confidence))
        if self.locked_exercise is not None:
            return  # a manual override wins over the model
        if len(self._history) < self.config.consensus_window:
            return

        counts = Counter(name for name, _ in self._history)
        top, top_count = counts.most_common(1)[0]
        if top == "__abstain__" or top not in self.analyzers:
            return

        scores = [c for name, c in self._history if name == top]
        if (top_count >= self.config.required_consensus
                and sum(scores) / len(scores) >= self.config.min_consensus_confidence):
            if self.active_exercise != top:
                self.active_exercise = top
                self._switch_count += 1

    # ---------------------------------------------------------------- control

    def lock_exercise(self, exercise: Optional[str]) -> None:
        """Pin the exercise (the UI's manual mode), or pass None to
        return control to the model."""
        if exercise is not None and exercise not in self.analyzers:
            raise ValueError(f"unknown exercise {exercise!r}; expected one of {sorted(self.analyzers)}")
        self.locked_exercise = exercise

    def reset(self) -> None:
        for analyzer in self.analyzers.values():
            analyzer.reset()
        self.active_exercise = None
        self._history.clear()
        self._last_recognition = None
        self._last_kinematics = {}
        self._last_pose_time = 0.0
        self._switch_count = 0

    # ----------------------------------------------------------------- output

    def state(self, now: Optional[float] = None) -> Dict[str, Any]:
        """
        The fused view, using the same keys `FitnessCVEngine` returns so
        `ui/dashboard.py` and `ui/overlay.py` need no changes, plus the
        new fields the learned branch makes available.
        """
        now = now if now is not None else time.time()
        target = self.locked_exercise or self.active_exercise

        if target is None:
            return self._warmup_state()

        tracking_lost = (
            self._last_pose_time > 0.0
            and (now - self._last_pose_time) > self.config.tracking_loss_timeout
        )
        if tracking_lost:
            return self._tracking_lost_state("Person tracking lost")

        kinematics = self._last_kinematics or {}
        recognition = self._last_recognition

        state_machine_reps = int(kinematics.get("rep_count", 0))
        learned_reps = len(getattr(recognition, "reps", []) or []) if recognition else 0
        authoritative = (
            learned_reps if self.config.rep_source == "learned" else state_machine_reps
        )

        form_feedback = self._format_form_flags(kinematics.get("form_flags", []))

        return {
            # --- existing contract, unchanged ---
            "exercise": target.replace("_", " ").title(),
            "exercise_id": target,
            "confidence": float(getattr(recognition, "confidence", 0.0)),
            "model_score": float(getattr(recognition, "confidence", 0.0)),
            "rep_count": authoritative,
            "phase": kinematics.get("phase", "MOVING"),
            "current_angle": kinematics.get("current_angle"),
            "rom": kinematics.get("rom"),
            "tempo": kinematics.get("tempo"),
            "form_feedback": form_feedback,
            "valid_pose": bool(kinematics.get("valid_pose", False)),
            "just_completed_rep": kinematics.get("just_completed_rep"),
            # --- new: what the learned branch adds ---
            "learned_phase": self._current_learned_phase(recognition),
            "learned_rep_count": learned_reps,
            "state_machine_rep_count": state_machine_reps,
            "rep_count_agreement": learned_reps == state_machine_reps,
            "rep_source": self.config.rep_source,
            "probabilities": dict(getattr(recognition, "probabilities", {}) or {}),
            "abstained": bool(getattr(recognition, "abstained", False)),
            "valid_pose_ratio": float(getattr(recognition, "valid_pose_ratio", 0.0)),
            "exercise_switches": self._switch_count,
            # The confidence above is temperature-calibrated by the
            # recogniser, not capped for display. Flagged explicitly so
            # nothing downstream re-applies the old min(0.96, x) heuristic.
            "confidence_is_calibrated": recognition is not None,
        }

    @staticmethod
    def _current_learned_phase(recognition: Any) -> Optional[str]:
        """The phase at the most recent frame of the analysed window."""
        sequence = getattr(recognition, "phase_sequence", None)
        return sequence[-1] if sequence else None

    @staticmethod
    def _format_form_flags(flags: List[Any]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for flag in flags or []:
            if isinstance(flag, dict):
                level = flag.get("level", "ok")
                out.append({
                    "status": "good" if level == "ok" else "warning",
                    "message": flag.get("message", ""),
                })
        return out or [{"status": "good", "message": "Controlled movement"}]

    @staticmethod
    def _warmup_state() -> Dict[str, Any]:
        return {
            "exercise": "ANALYZING...", "exercise_id": "analyzing",
            "confidence": 0.0, "model_score": 0.0, "rep_count": 0,
            "phase": "CALIBRATING", "current_angle": None, "rom": None, "tempo": None,
            "form_feedback": [{"status": "good", "message": "Collecting movement history..."}],
            "valid_pose": True, "learned_phase": None, "learned_rep_count": 0,
            "state_machine_rep_count": 0, "rep_count_agreement": True,
            "rep_source": "state_machine", "probabilities": {}, "abstained": False,
            "valid_pose_ratio": 0.0, "exercise_switches": 0,
            "confidence_is_calibrated": False,
        }

    @staticmethod
    def _tracking_lost_state(message: str) -> Dict[str, Any]:
        state = HybridMovementEngine._warmup_state()
        state.update({
            "exercise": "NO PERSON DETECTED", "exercise_id": "none",
            "phase": "TRACKING LOST", "valid_pose": False,
            "form_feedback": [{"status": "warning", "message": message}],
        })
        return state
