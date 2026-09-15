"""
Unit tests for CameraSource abstractions, device enumeration, and MockCameraSource.
Runs headlessly without requiring physical camera hardware.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from camera.sources import (
    CameraSource,
    BaseThreadedCameraSource,
    LocalWebcamSource,
    ExternalCameraSource,
    VideoFileSource,
    MockCameraSource,
    list_available_cameras,
)
from camera.capture import Camera


def test_mock_camera_source_lifecycle():
    """Verify MockCameraSource start, read, metrics, and stop."""
    source = MockCameraSource(width=640, height=480, fps=30)
    source.start()

    frame, ts = source.read()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert isinstance(frame, np.ndarray)
    assert ts > 0.0

    metrics = source.get_metrics()
    assert metrics["capture_fps"] == 30.0
    assert metrics["frame_count"] == 1
    assert metrics["latest_timestamp"] > 0.0

    source.stop()
    frame_after, ts_after = source.read()
    assert frame_after is None
    assert ts_after == 0.0


def test_local_and_external_camera_source_instantiation():
    """Verify class hierarchy and default parameter initialization."""
    local = LocalWebcamSource(camera_id=0, width=1280, height=720, fps=30)
    assert isinstance(local, CameraSource)
    assert isinstance(local, BaseThreadedCameraSource)
    assert local.source_id == 0
    assert local.width == 1280
    assert local.height == 720

    ext = ExternalCameraSource(camera_id=1, width=1920, height=1080, fps=60)
    assert isinstance(ext, CameraSource)
    assert isinstance(ext, BaseThreadedCameraSource)
    assert ext.source_id == 1
    assert ext.width == 1920
    assert ext.fps == 60


def test_video_file_source_file_not_found():
    """Verify VideoFileSource raises FileNotFoundError for missing path."""
    source = VideoFileSource("non_existent_video.mp4")
    with pytest.raises(FileNotFoundError):
        source.start()


@patch("cv2.VideoCapture")
def test_list_available_cameras_mocked(mock_vc):
    """Verify list_available_cameras scans devices and parses properties correctly."""
    # Mock VideoCapture instance for dev_id=0 (success) and dev_id=1 (fail)
    cap_mock_0 = MagicMock()
    cap_mock_0.isOpened.return_value = True
    cap_mock_0.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    cap_mock_0.get.side_effect = lambda prop: {
        3: 1280,  # CAP_PROP_FRAME_WIDTH
        4: 720,   # CAP_PROP_FRAME_HEIGHT
        5: 30.0,  # CAP_PROP_FPS
    }.get(prop, 0)

    cap_mock_fail = MagicMock()
    cap_mock_fail.isOpened.return_value = False

    def video_capture_side_effect(device_id, *args, **kwargs):
        if device_id == 0:
            return cap_mock_0
        return cap_mock_fail

    mock_vc.side_effect = video_capture_side_effect

    cameras = list_available_cameras(max_devices=3)
    assert len(cameras) == 1
    assert cameras[0]["id"] == 0
    assert cameras[0]["width"] == 1280
    assert cameras[0]["height"] == 720


def test_legacy_camera_wrapper_with_mock_source():
    """Verify legacy Camera class correctly wraps underlying CameraSource."""
    mock_src = MockCameraSource(width=640, height=480, fps=30)
    cam = Camera(source=mock_src)

    cam.start()
    frame = cam.get_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)

    metrics = cam.get_metrics()
    assert "capture_fps" in metrics
    cam.stop()
