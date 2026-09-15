"""
Threaded webcam capture wrapper preserving backward compatibility.
Integrates with camera/sources.py abstraction layer.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from camera.sources import (
    CameraSource,
    LocalWebcamSource,
    ExternalCameraSource,
    VideoFileSource,
    MockCameraSource,
)


class Camera:
    """Backward compatible Camera wrapper delegating to CameraSource."""

    def __init__(self, source: Union[int, str, CameraSource] = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self._camera_impl: CameraSource

        if isinstance(source, CameraSource):
            self._camera_impl = source
        elif isinstance(source, str):
            if source.endswith((".mp4", ".avi", ".mov", ".mkv")):
                self._camera_impl = VideoFileSource(video_path=source)
            else:
                try:
                    dev_id = int(source)
                    self._camera_impl = ExternalCameraSource(camera_id=dev_id, width=width, height=height, fps=fps)
                except ValueError:
                    self._camera_impl = VideoFileSource(video_path=source)
        else:
            self._camera_impl = LocalWebcamSource(camera_id=source, width=width, height=height, fps=fps)

    def start(self) -> "Camera":
        self._camera_impl.start()
        return self

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        return self._camera_impl.read()

    def get_frame(self) -> Optional[np.ndarray]:
        frame, _ = self.read()
        return frame

    def get_metrics(self) -> Dict[str, float]:
        return self._camera_impl.get_metrics()

    @property
    def frame_count(self) -> int:
        return int(self._camera_impl.get_metrics().get("frame_count", 0))

    def stop(self) -> None:
        self._camera_impl.stop()
