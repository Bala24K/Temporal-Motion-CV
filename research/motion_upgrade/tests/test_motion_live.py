"""
Live-stack tests.

These check the wiring rather than the vision: that a pose frame reaches
the kinematic branch, that a recognition reaches the consensus logic, and
that the fused state carries both the UI contract and the pipeline health
numbers. The stack is assembled from injected parts, so none of this
needs a camera, MediaPipe or a checkpoint.
"""
import time

import pytest

from motion.live import LiveStack, build_live_stack, load_recognizer
from motion.runtime import PipelineConfig
from tests.test_motion_runtime import FakeDetector, FakeRecognizer, FakeSource
from tests.motion_fixtures import curl_clip


def make_stack(recognizer=None, **kwargs) -> LiveStack:
    return build_live_stack(
        FakeSource(),
        FakeDetector(curl_clip(40)),
        recognizer,
        pipeline_config=PipelineConfig(inference_interval=0.03, min_frames_for_inference=3),
        **kwargs,
    )


def test_stack_connects_pose_frames_to_the_kinematic_branch():
    stack = make_stack()
    with stack:
        time.sleep(0.4)
        state = stack.state()
    # The engine has not been told an exercise yet, so it reports warmup —
    # but the analyzers must have been running all along.
    assert stack.pipeline.state().poses_detected > 0
    assert "rep_count" in state


def test_stack_connects_recognitions_to_the_consensus_logic():
    stack = make_stack(FakeRecognizer(delay=0.005))
    with stack:
        time.sleep(0.6)
    assert stack.engine.active_exercise == "bicep_curl"


def test_state_carries_both_the_ui_contract_and_pipeline_health():
    stack = make_stack(FakeRecognizer(delay=0.005))
    with stack:
        time.sleep(0.5)
        state = stack.state()

    for key in ("exercise", "rep_count", "phase", "form_feedback", "confidence"):
        assert key in state, f"UI contract key {key} missing"
    for key in ("capture_fps", "pose_fps", "inference_fps", "frames_dropped"):
        assert key in state, f"pipeline health key {key} missing"
    assert state["capture_fps"] > 0


def test_stack_stops_cleanly_via_the_context_manager():
    source = FakeSource()
    stack = build_live_stack(source, FakeDetector(curl_clip(10)))
    with stack:
        time.sleep(0.15)
    assert source.stopped is True


def test_missing_checkpoint_degrades_to_kinematics_only():
    # A moved model file must not stop the demo starting.
    assert load_recognizer("models/definitely_not_here.pt") is None
    assert load_recognizer(None) is None


def test_stack_runs_without_a_recognizer():
    stack = make_stack(None)
    with stack:
        time.sleep(0.3)
        state = stack.state()
    assert state["exercise_id"] == "analyzing"
    assert stack.pipeline.state().poses_detected > 0
