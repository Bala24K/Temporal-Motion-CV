"""
Fusion tests.

The contract being protected is that `HybridMovementEngine.state()`
returns everything `FitnessCVEngine.process_frame_sequence()` returned, so
the UI, the overlay and the session recorder keep working when the
recogniser underneath changes. That is asserted against the real
FitnessCVEngine output rather than a hand-copied list of keys, which
would drift.
"""
import time
from typing import List, Optional

import pytest

from motion.fusion import FusionConfig, HybridMovementEngine
from tests.motion_fixtures import curl_clip, neutral_clip, squat_clip


class Recognition:
    """Stands in for a MotionAnalysisResult."""

    def __init__(self, exercise="bicep_curl", confidence=0.9, abstained=False,
                 reps=0, phase="CONCENTRIC"):
        self.exercise = exercise
        self.confidence = confidence
        self.probabilities = {exercise: confidence}
        self.abstained = abstained
        self.reps = [object()] * reps
        self.phase_sequence = [phase]
        self.valid_pose_ratio = 1.0


def feed(engine: HybridMovementEngine, result: Recognition, times: int = 5) -> None:
    for _ in range(times):
        engine.update_recognition(result)


def test_state_before_any_recognition_is_the_warmup_view():
    state = HybridMovementEngine().state()
    assert state["exercise_id"] == "analyzing"
    assert state["rep_count"] == 0
    assert state["confidence_is_calibrated"] is False


def test_state_keys_match_the_existing_engine_contract():
    # Everything the current pipeline returns must still be present, or
    # ui/dashboard.py breaks on a KeyError at runtime rather than here.
    from recognition.fitness_pipeline import FitnessCVEngine

    legacy = FitnessCVEngine(bundle_path=None).process_frame_sequence(curl_clip(20))

    engine = HybridMovementEngine()
    feed(engine, Recognition())
    for frame in curl_clip(20):
        engine.update_pose(frame)
    fused = engine.state()

    missing = set(legacy.keys()) - set(fused.keys())
    assert not missing, f"the fused state dropped keys the UI reads: {sorted(missing)}"


def test_consensus_is_required_before_switching_exercise():
    engine = HybridMovementEngine(FusionConfig(consensus_window=5, required_consensus=3))
    for _ in range(4):
        engine.update_recognition(Recognition("squat"))
        assert engine.active_exercise is None, "must not switch on a partial window"
    engine.update_recognition(Recognition("squat"))
    assert engine.active_exercise == "squat"


def test_low_confidence_predictions_do_not_switch():
    engine = HybridMovementEngine(FusionConfig(min_consensus_confidence=0.55))
    feed(engine, Recognition("squat", confidence=0.3), times=8)
    assert engine.active_exercise is None


def test_abstention_never_switches_the_active_exercise():
    engine = HybridMovementEngine()
    feed(engine, Recognition("bicep_curl"))
    assert engine.active_exercise == "bicep_curl"

    feed(engine, Recognition("squat", abstained=True), times=8)
    assert engine.active_exercise == "bicep_curl", "an abstention is not a vote"
    assert engine.state()["abstained"] is True


def test_a_flickering_minority_prediction_does_not_flip_the_exercise():
    engine = HybridMovementEngine(FusionConfig(consensus_window=5, required_consensus=3))
    feed(engine, Recognition("squat"))
    assert engine.active_exercise == "squat"

    engine.update_recognition(Recognition("shoulder_press"))
    engine.update_recognition(Recognition("squat"))
    engine.update_recognition(Recognition("shoulder_press"))
    assert engine.active_exercise == "squat"


def test_sustained_change_does_switch_and_is_counted():
    engine = HybridMovementEngine()
    feed(engine, Recognition("squat"))
    feed(engine, Recognition("lateral_raise"), times=6)
    assert engine.active_exercise == "lateral_raise"
    assert engine.state()["exercise_switches"] == 2


def test_manual_lock_overrides_the_model():
    engine = HybridMovementEngine()
    engine.lock_exercise("squat")
    feed(engine, Recognition("bicep_curl"), times=10)
    assert engine.state()["exercise_id"] == "squat"

    engine.lock_exercise(None)
    feed(engine, Recognition("bicep_curl"), times=6)
    assert engine.state()["exercise_id"] == "bicep_curl"


def test_lock_rejects_an_unknown_exercise():
    with pytest.raises(ValueError, match="unknown exercise"):
        HybridMovementEngine().lock_exercise("deadlift")


def test_unknown_predicted_exercise_is_ignored_rather_than_crashing():
    engine = HybridMovementEngine()
    feed(engine, Recognition("burpee"), times=10)
    assert engine.active_exercise is None


def test_kinematics_run_on_the_active_exercise_analyzer():
    engine = HybridMovementEngine()
    feed(engine, Recognition("squat"))
    result = engine.update_pose(squat_clip(10)[5])
    assert result["exercise_id"] == "squat"
    assert "rep_count" in result


def test_both_rep_counts_are_reported_with_an_agreement_flag():
    engine = HybridMovementEngine()
    feed(engine, Recognition("bicep_curl", reps=3))
    for frame in curl_clip(30):
        engine.update_pose(frame)
    state = engine.state()

    assert state["learned_rep_count"] == 3
    assert "state_machine_rep_count" in state
    assert state["rep_count"] == state["state_machine_rep_count"], "state machine is authoritative by default"
    assert state["rep_count_agreement"] == (state["learned_rep_count"] == state["state_machine_rep_count"])


def test_rep_source_can_be_switched_to_the_learned_counter():
    engine = HybridMovementEngine(FusionConfig(rep_source="learned"))
    feed(engine, Recognition("bicep_curl", reps=7))
    for frame in curl_clip(20):
        engine.update_pose(frame)
    state = engine.state()
    assert state["rep_count"] == 7
    assert state["rep_source"] == "learned"


def test_learned_phase_is_the_most_recent_frame_of_the_window():
    engine = HybridMovementEngine()
    feed(engine, Recognition("squat", phase="ECCENTRIC"))
    engine.update_pose(squat_clip(10)[3])
    assert engine.state()["learned_phase"] == "ECCENTRIC"


def test_tracking_loss_is_reported_after_the_grace_period():
    engine = HybridMovementEngine(FusionConfig(tracking_loss_timeout=0.5))
    feed(engine, Recognition("bicep_curl"))
    now = time.time()
    engine.update_pose(curl_clip(10)[0], now=now)

    assert engine.state(now=now + 0.2)["exercise_id"] == "bicep_curl"
    lost = engine.state(now=now + 1.0)
    assert lost["exercise_id"] == "none"
    assert lost["phase"] == "TRACKING LOST"
    assert lost["valid_pose"] is False


def test_confidence_is_passed_through_uncapped():
    # The old pipeline capped the displayed score at 0.96. The calibrated
    # probability must reach the UI unmodified, or calibration is pointless.
    engine = HybridMovementEngine()
    feed(engine, Recognition("bicep_curl", confidence=0.995))
    engine.update_pose(curl_clip(10)[0])
    state = engine.state()
    assert state["confidence"] == pytest.approx(0.995)
    assert state["confidence_is_calibrated"] is True


def test_form_feedback_is_always_populated():
    engine = HybridMovementEngine()
    feed(engine, Recognition("bicep_curl"))
    engine.update_pose(neutral_clip(5)[0])
    feedback = engine.state()["form_feedback"]
    assert isinstance(feedback, list) and feedback
    assert set(feedback[0]) == {"status", "message"}


def test_reset_clears_counts_and_the_active_exercise():
    engine = HybridMovementEngine()
    feed(engine, Recognition("bicep_curl", reps=4))
    for frame in curl_clip(30):
        engine.update_pose(frame)

    engine.reset()
    assert engine.active_exercise is None
    state = engine.state()
    assert state["exercise_id"] == "analyzing"
    assert state["rep_count"] == 0
    assert state["exercise_switches"] == 0


def test_none_recognition_is_tolerated():
    engine = HybridMovementEngine()
    engine.update_recognition(None)
    assert engine.state()["exercise_id"] == "analyzing"
