"""
Asynchronous real-time pipeline.

The current application does capture -> pose -> recognise -> draw in one
loop, so every stage's latency adds up and the slowest one sets the frame
rate. A neural recogniser makes that worse: it is heavier than a Random
Forest, and run inline it would stall the camera on every window.

The fix is not "make the model faster". It is to let the stages run at
their own rates and never let a slow one block a fast one:

    capture thread    ~30 fps   always the freshest frame
         |  latest-wins slot (no queue)
    pose thread       ~15-30    MediaPipe, one frame at a time
         |  ring buffer of PoseFrames  +  per-frame kinematics
    inference thread  ~5-10     the ST-GCN over a window
         |  latest-result slot
    consumer (UI)     ~30 fps   never blocks on any of the above

LATEST-WINS, NOT A QUEUE
    Every handoff between stages holds exactly one item and overwrites it.
    A queue would absorb a burst and then hand the pose stage frames that
    are already a second old — the pipeline would report a healthy fps
    while showing the past. Dropping is the correct behaviour for live
    video, and the number of dropped frames is counted and reported
    rather than hidden.

WHAT THIS MODULE DOES NOT DO
    It does not import cv2, mediapipe or torch. The camera source, the
    pose detector and the recogniser are injected, which is what makes
    the whole pipeline testable with fakes — including the threading,
    which is otherwise the least-tested and most fragile part of a live
    system.
"""
from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Sequence

import numpy as np

from pose.landmarks import PoseFrame


class FrameSource(Protocol):
    """Anything shaped like `camera.capture.Camera`."""

    def read(self): ...
    def stop(self) -> None: ...


class PoseEstimator(Protocol):
    """Anything shaped like `pose.detector.PoseDetector`."""

    def detect(self, frame) -> Optional[PoseFrame]: ...


class LatencyMonitor:
    """
    Per-stage timing with percentiles.

    Mean latency hides the problem. A pipeline averaging 30 ms with a p95
    of 180 ms feels broken to use, and only the percentile says so — which
    is why the plan asks for p95 specifically and why this records a
    rolling window of samples rather than a running mean.
    """

    def __init__(self, window: int = 300):
        self._window = window
        self._samples: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            buf = self._samples.setdefault(stage, deque(maxlen=self._window))
            buf.append(float(seconds) * 1000.0)  # milliseconds

    def timer(self, stage: str) -> "_StageTimer":
        return _StageTimer(self, stage)

    def stats(self, stage: str) -> Dict[str, float]:
        with self._lock:
            samples = list(self._samples.get(stage, ()))
        if not samples:
            return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        arr = np.asarray(samples)
        return {
            "n": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            stages = list(self._samples.keys())
        return {stage: self.stats(stage) for stage in stages}

    def report(self) -> str:
        lines = ["stage                  n    mean     p50     p95     max"]
        for stage, s in sorted(self.all_stats().items()):
            lines.append(
                f"{stage:<20} {s['n']:>4} {s['mean_ms']:>7.1f} "
                f"{s['p50_ms']:>7.1f} {s['p95_ms']:>7.1f} {s['max_ms']:>7.1f}"
            )
        return "\n".join(lines)


class _StageTimer:
    """Context manager returned by `LatencyMonitor.timer`."""

    def __init__(self, monitor: LatencyMonitor, stage: str):
        self._monitor = monitor
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self._monitor.record(self._stage, time.perf_counter() - self._start)
        return False


class LatestSlot:
    """
    A one-item handoff between threads. Writing overwrites whatever was
    there and counts the drop; reading takes the item and leaves the slot
    empty.
    """

    def __init__(self):
        self._item: Any = None
        self._has_item = False
        self._dropped = 0
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)

    def put(self, item: Any) -> None:
        with self._ready:
            if self._has_item:
                self._dropped += 1
            self._item = item
            self._has_item = True
            self._ready.notify()

    def get(self, timeout: Optional[float] = None) -> Any:
        """Block until an item is available, or return None on timeout."""
        with self._ready:
            if not self._has_item:
                self._ready.wait(timeout)
            if not self._has_item:
                return None
            item, self._item, self._has_item = self._item, None, False
            return item

    def peek(self) -> Any:
        """Read without consuming — for the consumer thread, which must
        never block and must tolerate seeing the same result twice."""
        with self._lock:
            return self._item

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped


class PoseRingBuffer:
    """
    Thread-safe rolling window of PoseFrames.

    `snapshot` copies the list under the lock so the inference thread can
    work on a stable window while the pose thread keeps appending — a
    window mutating mid-inference produces tensors whose time axis isn't
    monotonic, which is close to impossible to debug from a loss curve.
    """

    def __init__(self, max_len: int = 64):
        self.max_len = max_len
        self._frames: Deque[PoseFrame] = deque(maxlen=max_len)
        self._lock = threading.Lock()

    def push(self, frame: PoseFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self) -> List[PoseFrame]:
        with self._lock:
            return list(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


@dataclass
class PipelineConfig:
    """Rates and sizes. All timing is in seconds."""
    window_frames: int = 48
    min_frames_for_inference: int = 12
    inference_interval: float = 0.15      # ~6-7 recognitions per second
    max_capture_fps: Optional[float] = None   # throttle; None means as fast as the source
    pose_timeout: float = 0.5             # how long to wait for a frame before idling
    tracking_loss_timeout: float = 0.5    # matches FitnessCVEngine's grace period
    latency_window: int = 300
    log_stage_timings: bool = True


@dataclass
class PipelineState:
    """
    A consistent snapshot for the consumer. Built under one lock so the
    UI can never render a rep count from one moment beside an exercise
    label from another.
    """
    raw_frame: Any = None                      # freshest camera frame, for display
    pose_frame: Optional[PoseFrame] = None
    recognition: Optional[Any] = None          # MotionAnalysisResult or None
    kinematics: Dict[str, Any] = field(default_factory=dict)
    frames_captured: int = 0
    frames_dropped: int = 0
    poses_detected: int = 0
    pose_failures: int = 0
    inferences: int = 0
    capture_fps: float = 0.0
    pose_fps: float = 0.0
    inference_fps: float = 0.0
    tracking_ok: bool = False
    last_pose_time: float = 0.0


class AsyncMotionPipeline:
    """
    Runs capture, pose estimation and recognition on separate threads.

        pipeline = AsyncMotionPipeline(source=camera, detector=detector,
                                       recognizer=recognizer,
                                       on_pose=engine.update_pose)
        pipeline.start()
        while running:
            state = pipeline.state()      # never blocks
            draw(state)
        pipeline.stop()

    `on_pose` is called on the pose thread for every detected frame — the
    hook for per-frame kinematics, which are cheap enough to run at pose
    rate and are what the form feedback reads. `on_recognition` is called
    on the inference thread when a new window result arrives.
    """

    def __init__(
        self,
        source: FrameSource,
        detector: PoseEstimator,
        recognizer: Optional[Any] = None,
        config: Optional[PipelineConfig] = None,
        on_pose: Optional[Callable[[PoseFrame], Dict[str, Any]]] = None,
        on_recognition: Optional[Callable[[Any], None]] = None,
    ):
        self.source = source
        self.detector = detector
        self.recognizer = recognizer
        self.config = config or PipelineConfig()
        self.on_pose = on_pose
        self.on_recognition = on_recognition

        self.latency = LatencyMonitor(window=self.config.latency_window)
        self.buffer = PoseRingBuffer(max_len=self.config.window_frames)

        self._frame_slot = LatestSlot()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._state_lock = threading.Lock()
        self._state = PipelineState()
        self._errors: List[str] = []

        self._capture_times: Deque[float] = deque(maxlen=60)
        self._pose_times: Deque[float] = deque(maxlen=60)
        self._inference_times: Deque[float] = deque(maxlen=30)

    # ---------------------------------------------------------------- threads

    def _capture_loop(self) -> None:
        # A real camera paces itself at its own frame rate. A synthetic or
        # file source does not, and left uncapped it spins at thousands of
        # frames per second, making the reported drop rate meaningless.
        min_interval = (
            1.0 / self.config.max_capture_fps if self.config.max_capture_fps else 0.0
        )
        next_frame_at = 0.0
        while not self._stop.is_set():
            if min_interval:
                wait = next_frame_at - time.perf_counter()
                if wait > 0 and self._stop.wait(wait):
                    return
                next_frame_at = time.perf_counter() + min_interval
            try:
                with self.latency.timer("capture"):
                    result = self.source.read()
            except Exception as exc:  # a dead camera must not take the process down
                self._record_error(f"capture: {exc}")
                time.sleep(0.05)
                continue

            frame = result[0] if isinstance(result, tuple) else result
            if frame is None:
                # End of a video file, or a dropped USB frame. Either way
                # there is nothing to do but yield and retry.
                time.sleep(0.005)
                continue

            timestamp = result[1] if isinstance(result, tuple) and len(result) > 1 else time.time()
            self._frame_slot.put((frame, timestamp))
            self._capture_times.append(time.perf_counter())
            with self._state_lock:
                # Held by reference for the consumer to draw on. The pose
                # thread may be working on an older frame; showing the
                # newest one keeps the video smooth while the skeleton
                # lags by a frame or two, which reads far better than a
                # video feed stuttering in lockstep with MediaPipe.
                self._state.raw_frame = frame
                self._state.frames_captured += 1

    def _pose_loop(self) -> None:
        while not self._stop.is_set():
            item = self._frame_slot.get(timeout=self.config.pose_timeout)
            if item is None:
                continue
            frame, _timestamp = item

            try:
                with self.latency.timer("pose"):
                    pose_frame = self.detector.detect(frame)
            except Exception as exc:
                self._record_error(f"pose: {exc}")
                continue

            # Stage throughput is recorded whether or not a person was
            # found: pose_fps answers "how fast is this stage running",
            # and a detector that fails quickly is still running quickly.
            # Conflating it with the detection rate reports 0 fps for a
            # pipeline that is in fact keeping up perfectly.
            self._pose_times.append(time.perf_counter())

            if pose_frame is None:
                with self._state_lock:
                    self._state.pose_failures += 1
                    self._state.tracking_ok = (
                        time.time() - self._state.last_pose_time
                    ) <= self.config.tracking_loss_timeout
                continue

            self.buffer.push(pose_frame)

            kinematics: Dict[str, Any] = {}
            if self.on_pose is not None:
                try:
                    with self.latency.timer("kinematics"):
                        kinematics = self.on_pose(pose_frame) or {}
                except Exception as exc:
                    self._record_error(f"kinematics: {exc}")

            now = time.time()
            with self._state_lock:
                self._state.pose_frame = pose_frame
                self._state.kinematics = kinematics
                self._state.poses_detected += 1
                self._state.last_pose_time = now
                self._state.tracking_ok = True

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            # Sleep in short slices so stop() is responsive rather than
            # waiting out a full interval.
            deadline = time.perf_counter() + self.config.inference_interval
            while time.perf_counter() < deadline:
                if self._stop.wait(0.01):
                    return

            if self.recognizer is None:
                continue
            window = self.buffer.snapshot()
            if len(window) < self.config.min_frames_for_inference:
                continue

            try:
                with self.latency.timer("inference"):
                    result = self.recognizer.analyze(window)
            except Exception as exc:
                self._record_error(f"inference: {exc}")
                continue

            if result is None:
                continue

            self._inference_times.append(time.perf_counter())
            with self._state_lock:
                self._state.recognition = result
                self._state.inferences += 1

            if self.on_recognition is not None:
                try:
                    self.on_recognition(result)
                except Exception as exc:
                    self._record_error(f"on_recognition: {exc}")

    # ------------------------------------------------------------- lifecycle

    def start(self) -> "AsyncMotionPipeline":
        if self._threads:
            raise RuntimeError("pipeline already started")
        self._stop.clear()
        targets = [("capture", self._capture_loop), ("pose", self._pose_loop),
                   ("inference", self._inference_loop)]
        for name, target in targets:
            thread = threading.Thread(target=target, name=f"motion-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """
        Signal every thread and wait. Daemon threads mean a hung stage
        cannot prevent the process exiting, but joining with a timeout
        still gives MediaPipe a chance to release the camera cleanly.
        """
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()
        try:
            self.source.stop()
        except Exception as exc:
            self._record_error(f"source.stop: {exc}")

    def __enter__(self) -> "AsyncMotionPipeline":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    # ---------------------------------------------------------------- output

    @staticmethod
    def _rate(times: Deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return float((len(times) - 1) / span) if span > 1e-9 else 0.0

    def state(self) -> PipelineState:
        """A consistent snapshot. Never blocks on a worker thread."""
        with self._state_lock:
            snapshot = PipelineState(
                raw_frame=self._state.raw_frame,
                pose_frame=self._state.pose_frame,
                recognition=self._state.recognition,
                kinematics=dict(self._state.kinematics),
                frames_captured=self._state.frames_captured,
                frames_dropped=self._frame_slot.dropped,
                poses_detected=self._state.poses_detected,
                pose_failures=self._state.pose_failures,
                inferences=self._state.inferences,
                tracking_ok=self._state.tracking_ok,
                last_pose_time=self._state.last_pose_time,
            )
        snapshot.capture_fps = self._rate(self._capture_times)
        snapshot.pose_fps = self._rate(self._pose_times)
        snapshot.inference_fps = self._rate(self._inference_times)
        return snapshot

    def _record_error(self, message: str) -> None:
        with self._state_lock:
            self._errors.append(f"{time.time():.3f} {message}")
            if len(self._errors) > 50:
                del self._errors[:-50]

    @property
    def errors(self) -> List[str]:
        with self._state_lock:
            return list(self._errors)

    def profile(self) -> Dict[str, Any]:
        """
        Everything needed to answer "where does the time go" — the
        measurement the optimisation plan asks for before any quantising
        or ONNX conversion. Optimising an unprofiled pipeline usually
        means speeding up the stage that was never the bottleneck.
        """
        state = self.state()
        stages = self.latency.all_stats()
        end_to_end = sum(s["mean_ms"] for s in stages.values())
        return {
            "stages": stages,
            "estimated_serial_latency_ms": end_to_end,
            "capture_fps": state.capture_fps,
            "pose_fps": state.pose_fps,
            "inference_fps": state.inference_fps,
            "frames_captured": state.frames_captured,
            "frames_dropped": state.frames_dropped,
            "drop_rate": (state.frames_dropped / state.frames_captured) if state.frames_captured else 0.0,
            "poses_detected": state.poses_detected,
            "pose_failure_rate": (
                state.pose_failures / max(1, state.poses_detected + state.pose_failures)
            ),
            "inferences": state.inferences,
            "errors": self.errors[-5:],
        }
