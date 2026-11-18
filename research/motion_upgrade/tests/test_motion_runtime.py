"""
Runtime tests.

Threading is the part of a live system that is normally untested, because
testing it usually means owning a webcam. Injecting the source, the
detector and the recogniser removes that excuse: every stage here runs
with fakes, including the drop behaviour and the shutdown path, which are
exactly the bits that fail in a demo.

Timing assertions are kept loose — this runs on shared CI hardware — and
the strict assertions are on invariants that hold at any speed: no
deadlock, no stale state, bounded memory, clean stop.
"""
import threading
import time
from typing import List, Optional

import numpy as np
import pytest

from motion.runtime import (
    AsyncMotionPipeline,
    LatencyMonitor,
    LatestSlot,
    PipelineConfig,
    PoseRingBuffer,
)
from pose.landmarks import PoseFrame
from tests.motion_fixtures import curl_clip, neutral_clip


class FakeSource:
    """A camera that never blocks and counts how often it was read."""

    def __init__(self, delay: float = 0.002, fail_after: Optional[int] = None):
        self.delay = delay
        self.fail_after = fail_after
        self.reads = 0
        self.stopped = False

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise RuntimeError("camera disconnected")
        time.sleep(self.delay)
        return np.zeros((8, 8, 3), dtype=np.uint8), time.time()

    def stop(self):
        self.stopped = True


class FakeDetector:
    """Replays a fixture clip, optionally failing to detect a person."""

    def __init__(self, frames: List[PoseFrame], detect_every: int = 1, delay: float = 0.003):
        self.frames = frames
        self.detect_every = detect_every
        self.delay = delay
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        time.sleep(self.delay)
        if self.calls % self.detect_every != 0:
            return None
        return self.frames[self.calls % len(self.frames)]


class FakeRecognition:
    def __init__(self, exercise="bicep_curl", confidence=0.9):
        self.exercise = exercise
        self.confidence = confidence
        self.probabilities = {exercise: confidence}
        self.phase_sequence = ["CONCENTRIC"]
        self.reps = []
        self.abstained = False
        self.valid_pose_ratio = 1.0


class FakeRecognizer:
    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.calls = 0
        self.window_sizes: List[int] = []

    def analyze(self, window):
        self.calls += 1
        self.window_sizes.append(len(window))
        time.sleep(self.delay)
        return FakeRecognition()


def run_pipeline(pipeline: AsyncMotionPipeline, seconds: float = 0.6):
    """
    Run for a fixed duration, then stop BEFORE snapshotting.

    Reading state while the threads are live races against them by
    construction — the counters are incremented after each stage's work
    completes, so a mid-flight snapshot can legitimately be one behind.
    That is fine at runtime and useless in a test, so the tests that
    assert exact counts measure a stopped pipeline.
    """
    pipeline.start()
    time.sleep(seconds)
    pipeline.stop()
    return pipeline.state()


# ------------------------------------------------------------ LatencyMonitor


def test_latency_monitor_percentiles():
    monitor = LatencyMonitor()
    for ms in range(1, 101):
        monitor.record("pose", ms / 1000.0)
    stats = monitor.stats("pose")
    assert stats["n"] == 100
    assert stats["p50_ms"] == pytest.approx(50.5, abs=1.0)
    assert stats["p95_ms"] == pytest.approx(95.0, abs=2.0)
    assert stats["max_ms"] == pytest.approx(100.0, abs=0.1)


def test_latency_monitor_reports_p95_not_just_mean():
    # The case the percentile exists for: a mostly-fast pipeline with a
    # long tail reads as healthy on the mean alone.
    monitor = LatencyMonitor()
    for _ in range(90):
        monitor.record("inference", 0.010)
    for _ in range(10):
        monitor.record("inference", 0.400)
    stats = monitor.stats("inference")
    assert stats["mean_ms"] < 60, "the mean says this pipeline is fine"
    assert stats["p95_ms"] > 300, "the p95 says it stutters every twentieth frame"


def test_latency_monitor_window_is_bounded():
    monitor = LatencyMonitor(window=50)
    for i in range(500):
        monitor.record("capture", i / 1000.0)
    assert monitor.stats("capture")["n"] == 50


def test_latency_timer_context_manager_records():
    monitor = LatencyMonitor()
    with monitor.timer("stage"):
        time.sleep(0.01)
    assert monitor.stats("stage")["n"] == 1
    assert monitor.stats("stage")["mean_ms"] >= 5


def test_unknown_stage_returns_zeros_not_an_error():
    assert LatencyMonitor().stats("nope")["n"] == 0


def test_latency_report_is_printable():
    monitor = LatencyMonitor()
    monitor.record("pose", 0.02)
    assert "pose" in monitor.report()


# ----------------------------------------------------------------- LatestSlot


def test_latest_slot_overwrites_and_counts_drops():
    slot = LatestSlot()
    slot.put("a")
    slot.put("b")
    slot.put("c")
    assert slot.dropped == 2, "an unread item must be counted as dropped, not queued"
    assert slot.get(timeout=0.01) == "c", "the consumer must get the freshest item"
    assert slot.get(timeout=0.01) is None


def test_latest_slot_get_times_out_instead_of_hanging():
    start = time.perf_counter()
    assert LatestSlot().get(timeout=0.05) is None
    assert time.perf_counter() - start < 1.0


def test_latest_slot_peek_does_not_consume():
    slot = LatestSlot()
    slot.put("x")
    assert slot.peek() == "x"
    assert slot.peek() == "x"
    assert slot.get(timeout=0.01) == "x"


def test_latest_slot_wakes_a_waiting_reader():
    slot = LatestSlot()
    received = []

    def reader():
        received.append(slot.get(timeout=2.0))

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.05)
    slot.put("ping")
    thread.join(timeout=2.0)
    assert received == ["ping"]


# -------------------------------------------------------------- ring buffer


def test_ring_buffer_is_bounded_and_keeps_the_newest():
    buffer = PoseRingBuffer(max_len=10)
    frames = neutral_clip(30)
    for frame in frames:
        buffer.push(frame)
    snapshot = buffer.snapshot()
    assert len(snapshot) == 10
    assert snapshot[-1] is frames[-1]


def test_ring_buffer_snapshot_is_a_copy():
    # The inference thread must see a stable window while the pose thread
    # keeps appending.
    buffer = PoseRingBuffer(max_len=5)
    for frame in neutral_clip(3):
        buffer.push(frame)
    snapshot = buffer.snapshot()
    buffer.push(neutral_clip(1)[0])
    assert len(snapshot) == 3


def test_ring_buffer_is_thread_safe_under_contention():
    buffer = PoseRingBuffer(max_len=32)
    frames = neutral_clip(50)
    errors: List[Exception] = []

    def writer():
        for _ in range(200):
            for frame in frames:
                buffer.push(frame)

    def reader():
        try:
            for _ in range(500):
                assert len(buffer.snapshot()) <= 32
        except Exception as exc:  # pragma: no cover - only on a real race
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors


# ------------------------------------------------------------------ pipeline


def test_pipeline_runs_all_three_stages():
    source = FakeSource()
    detector = FakeDetector(curl_clip(40))
    recognizer = FakeRecognizer()
    pipeline = AsyncMotionPipeline(
        source, detector, recognizer,
        config=PipelineConfig(inference_interval=0.05, min_frames_for_inference=3),
    )
    state = run_pipeline(pipeline, seconds=0.7)

    assert state.frames_captured > 0
    assert state.poses_detected > 0
    assert state.inferences > 0
    assert state.recognition is not None
    assert state.tracking_ok is True


def test_pipeline_stops_cleanly_and_releases_the_source():
    source = FakeSource()
    pipeline = AsyncMotionPipeline(source, FakeDetector(curl_clip(20)))
    pipeline.start()
    time.sleep(0.2)
    start = time.perf_counter()
    pipeline.stop(timeout=2.0)
    assert time.perf_counter() - start < 3.0, "stop must not hang on a sleeping thread"
    assert source.stopped is True
    assert all(not t.is_alive() for t in pipeline._threads)


def test_pipeline_cannot_be_started_twice():
    pipeline = AsyncMotionPipeline(FakeSource(), FakeDetector(curl_clip(10)))
    pipeline.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            pipeline.start()
    finally:
        pipeline.stop()


def test_pipeline_works_as_a_context_manager():
    source = FakeSource()
    with AsyncMotionPipeline(source, FakeDetector(curl_clip(10))) as pipeline:
        time.sleep(0.15)
        assert pipeline.state().frames_captured > 0
    assert source.stopped is True


def test_slow_inference_does_not_starve_capture():
    # The property the whole design exists for: a 120 ms model must not
    # drag the camera down with it.
    source = FakeSource(delay=0.002)
    slow = FakeRecognizer(delay=0.12)
    pipeline = AsyncMotionPipeline(
        source, FakeDetector(curl_clip(30), delay=0.002), slow,
        config=PipelineConfig(inference_interval=0.01, min_frames_for_inference=3),
    )
    state = run_pipeline(pipeline, seconds=0.8)

    assert state.frames_captured > 20, "capture kept running while inference was slow"
    assert state.frames_captured > state.inferences * 3


def test_frames_are_dropped_rather_than_queued():
    # A slow pose stage must lose frames, not accumulate a backlog that
    # shows the past at a healthy-looking frame rate.
    pipeline = AsyncMotionPipeline(
        FakeSource(delay=0.001), FakeDetector(curl_clip(20), delay=0.03),
        config=PipelineConfig(inference_interval=1.0),
    )
    state = run_pipeline(pipeline, seconds=0.6)
    assert state.frames_dropped > 0
    assert state.frames_captured > state.poses_detected


def test_pipeline_survives_a_failing_camera():
    source = FakeSource(fail_after=5)
    pipeline = AsyncMotionPipeline(source, FakeDetector(curl_clip(10)))
    state = run_pipeline(pipeline, seconds=0.3)
    assert any("capture" in e for e in pipeline.errors)
    assert isinstance(state.frames_captured, int)  # still responsive


def test_pipeline_survives_a_failing_recognizer():
    class Exploding:
        def analyze(self, window):
            raise ValueError("bad tensor")

    pipeline = AsyncMotionPipeline(
        FakeSource(), FakeDetector(curl_clip(20)), Exploding(),
        config=PipelineConfig(inference_interval=0.02, min_frames_for_inference=2),
    )
    run_pipeline(pipeline, seconds=0.4)
    assert any("inference" in e for e in pipeline.errors)


def test_undetected_poses_are_counted_not_crashed_on():
    pipeline = AsyncMotionPipeline(
        FakeSource(), FakeDetector(curl_clip(20), detect_every=3),
        config=PipelineConfig(inference_interval=1.0),
    )
    state = run_pipeline(pipeline, seconds=0.5)
    assert state.pose_failures > 0
    assert state.poses_detected > 0


def test_inference_waits_for_enough_frames():
    recognizer = FakeRecognizer()
    pipeline = AsyncMotionPipeline(
        FakeSource(), FakeDetector(curl_clip(20), delay=0.02), recognizer,
        config=PipelineConfig(inference_interval=0.02, min_frames_for_inference=1000),
    )
    run_pipeline(pipeline, seconds=0.3)
    assert recognizer.calls == 0, "must not run on a half-empty window"


def test_pipeline_without_a_recognizer_still_produces_poses():
    # Kinematics-only mode: useful as a fallback when no checkpoint exists.
    pipeline = AsyncMotionPipeline(FakeSource(), FakeDetector(curl_clip(20)), recognizer=None)
    state = run_pipeline(pipeline, seconds=0.3)
    assert state.poses_detected > 0
    assert state.recognition is None


def test_on_pose_hook_runs_at_pose_rate():
    seen: List[PoseFrame] = []
    pipeline = AsyncMotionPipeline(
        FakeSource(), FakeDetector(curl_clip(20)),
        config=PipelineConfig(inference_interval=1.0),
        on_pose=lambda frame: (seen.append(frame), {"rep_count": len(seen)})[1],
    )
    state = run_pipeline(pipeline, seconds=0.4)
    assert len(seen) == state.poses_detected
    assert state.kinematics.get("rep_count") == len(seen)


def test_on_recognition_hook_receives_results():
    received: List[object] = []
    pipeline = AsyncMotionPipeline(
        FakeSource(), FakeDetector(curl_clip(20)), FakeRecognizer(),
        config=PipelineConfig(inference_interval=0.03, min_frames_for_inference=2),
        on_recognition=received.append,
    )
    run_pipeline(pipeline, seconds=0.4)
    assert len(received) > 0


def test_state_snapshot_never_blocks_and_is_internally_consistent():
    pipeline = AsyncMotionPipeline(
        FakeSource(), FakeDetector(curl_clip(30)), FakeRecognizer(delay=0.05),
        config=PipelineConfig(inference_interval=0.02, min_frames_for_inference=2),
    )
    pipeline.start()
    try:
        for _ in range(50):
            start = time.perf_counter()
            state = pipeline.state()
            assert time.perf_counter() - start < 0.05, "state() must not wait on a worker"
            assert state.poses_detected >= 0
            assert state.frames_captured >= state.poses_detected - 1
    finally:
        pipeline.stop()


def test_profile_reports_every_stage():
    pipeline = AsyncMotionPipeline(
        FakeSource(), FakeDetector(curl_clip(20)), FakeRecognizer(),
        config=PipelineConfig(inference_interval=0.03, min_frames_for_inference=2),
        on_pose=lambda frame: {},
    )
    pipeline.start()
    time.sleep(0.5)
    profile = pipeline.profile()
    pipeline.stop()

    for stage in ("capture", "pose", "inference"):
        assert stage in profile["stages"], f"{stage} timings missing from the profile"
    assert profile["estimated_serial_latency_ms"] > 0
    assert 0.0 <= profile["drop_rate"] <= 1.0
    assert 0.0 <= profile["pose_failure_rate"] <= 1.0
