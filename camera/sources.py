"""
Camera Source Abstraction Layer & Device Enumeration.

Provides a unified interface for multiple video frame sources:
  - LocalWebcamSource: Built-in or standard USB webcam
  - ExternalCameraSource: iPhone / external camera via Windows virtual device (Camo, DroidCam, EpocCam)
  - VideoFileSource: Prerecorded .mp4 video file stream
  - MockCameraSource: Synthetic frame stream for headless automated unit tests

Features:
  - Device enumeration (list_available_cameras)
  - Camera diagnostics test mode (run_camera_test)
  - Threaded recency-preferring capture loop (drops stale frames under load)
  - Measured capture FPS and frame latency monitoring
"""

from abc import ABC, abstractmethod
from collections import deque
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np


class CameraSource(ABC):
    """Abstract Base Class for all video frame input sources."""

    @abstractmethod
    def start(self) -> "CameraSource":
        """Initializes and starts the frame capture stream."""
        pass

    @abstractmethod
    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Reads the most recent frame and timestamp.
        Returns: (frame_matrix_or_None, timestamp_in_seconds)
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops capture and releases underlying video devices."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Returns capture performance metrics (FPS, frame count, latency)."""
        pass


class BaseThreadedCameraSource(CameraSource):
    """
    Threaded base capture implementation.
    Reads frames on a background thread into a size-1 slot to prefer recency over completeness.
    """

    def __init__(self, source_id: Union[int, str] = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.source_id = source_id
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = 0.0

        self._frame_times = deque(maxlen=30)

    def _init_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source_id)
        if isinstance(self.source_id, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def start(self) -> "CameraSource":
        self._cap = self._init_capture()
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera device/source '{self.source_id}'. "
                "Check that the camera device is connected and not in use by another application."
            )

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        while self._running:
            if self._cap is None:
                break
            # Flush hardware driver buffer to get the freshest physical frame
            ok = self._cap.grab()
            if not ok:
                time.sleep(0.005)
                continue
            ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                continue

            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = now
                self._frame_count += 1
                self._frame_times.append(now)

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_ts

    def get_metrics(self) -> Dict[str, float]:
        with self._lock:
            if len(self._frame_times) >= 2:
                span = self._frame_times[-1] - self._frame_times[0]
                fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            else:
                fps = 0.0

            return {
                "capture_fps": float(fps),
                "frame_count": int(self._frame_count),
                "latest_timestamp": float(self._latest_ts),
            }

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None


class LocalWebcamSource(BaseThreadedCameraSource):
    """Camera source for built-in or standard USB webcams."""
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__(source_id=camera_id, width=width, height=height, fps=fps)


class ExternalCameraSource(BaseThreadedCameraSource):
    """
    Camera source for external cameras (e.g. iPhone connected via Camo, DroidCam, EpocCam, or USB video card).
    Appears to Windows as a normal camera device index.
    """
    def __init__(self, camera_id: int = 1, width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__(source_id=camera_id, width=width, height=height, fps=fps)


class VideoFileSource(CameraSource):
    """Camera source for reading prerecorded .mp4 video files in a loop or single pass."""

    def __init__(self, video_path: str, loop: bool = True):
        self.video_path = video_path
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time = 0.0

    def start(self) -> "CameraSource":
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file '{self.video_path}' not found.")
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file '{self.video_path}'.")
        self._start_time = time.time()
        return self

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        if self._cap is None:
            return None, 0.0

        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None, 0.0

        self._frame_count += 1
        return frame, time.time()

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_metrics(self) -> Dict[str, float]:
        return {
            "capture_fps": 30.0,
            "frame_count": self._frame_count,
            "latest_timestamp": time.time(),
        }


class MockCameraSource(CameraSource):
    """Synthetic camera source generating dummy frames for automated testing."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self._frame_count = 0

    def start(self) -> "CameraSource":
        self._running = True
        return self

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        if not self._running:
            return None, 0.0

        self._frame_count += 1
        now = time.time()
        # Generate dummy synthetic BGR image frame
        frame = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        cv2.putText(frame, f"Mock Frame {self._frame_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame, now

    def stop(self) -> None:
        self._running = False

    def get_metrics(self) -> Dict[str, float]:
        return {
            "capture_fps": float(self.fps),
            "frame_count": self._frame_count,
            "latest_timestamp": time.time(),
        }


def list_available_cameras(max_devices: int = 10) -> List[Dict]:
    """
    Scans camera indices 0 through max_devices-1 to enumerate available Windows video input devices.
    Returns list of camera info dictionaries: [{"id": 0, "resolution": "1280x720", ...}]
    """
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

    found_cameras = []
    print(f"Scanning camera indices 0 to {max_devices - 1}...")

    for dev_id in range(max_devices):
        cap = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(dev_id)

        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                found_cameras.append({
                    "id": dev_id,
                    "width": w,
                    "height": h,
                    "fps": fps,
                    "name": f"Camera Device {dev_id} ({w}x{h})"
                })
            cap.release()

    return found_cameras


def run_camera_test(camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
    """
    Runs an interactive camera diagnostic test: opens camera, displays live feed,
    shows resolution, measured FPS, and frame latency overlay, exiting on 'q' or Esc.
    """
    print(f"\n--- CAMERA DIAGNOSTIC TEST (Device ID: {camera_id}) ---")
    print(f"Requesting resolution: {width}x{height} @ {fps} FPS")

    try:
        cam = ExternalCameraSource(camera_id=camera_id, width=width, height=height, fps=fps).start()
    except Exception as e:
        print(f"Error: Unable to start camera {camera_id}: {e}")
        return

    print("Camera feed opened successfully! Press 'q' or ESC to exit diagnostic test.\n")

    frame_counter = 0
    t_start = time.time()

    try:
        while True:
            loop_start = time.time()
            frame, cap_ts = cam.read()

            if frame is None:
                time.sleep(0.005)
                continue

            frame_counter += 1
            metrics = cam.get_metrics()
            curr_fps = metrics.get("capture_fps", 0.0)
            latency_ms = (time.time() - loop_start) * 1000.0

            h, w, _ = frame.shape
            info_str = f"Camera ID: {camera_id} | Res: {w}x{h} | FPS: {curr_fps:.1f} | Latency: {latency_ms:.1f}ms"

            cv2.putText(frame, info_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, info_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(f"Camera Diagnostic Test - Device {camera_id}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera diagnostic test finished.")
