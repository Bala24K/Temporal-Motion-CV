"""
Wraps MediaPipe's Pose Landmarker and converts its output into our own
PoseFrame representation (see pose/landmarks.py). Nothing outside this
file should import mediapipe directly — that's the whole point of having
our own Landmark/PoseFrame types: swap the model later without touching
biomechanics/, exercises/, or ui/.

Setup (on the machine that actually has a webcam — not needed to run the
test suite):
    pip install mediapipe opencv-python

    # Download the pose landmarker model once:
    mkdir -p models
    curl -L -o models/pose_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

The "lite" variant is the right starting point for the 20+ FPS / low
latency target in the V1 contract — try "full" only if lite's accuracy
turns out to be the bottleneck, not before.
"""
import time
from typing import Optional

import numpy as np

from pose.landmarks import PoseFrame, Landmark, MEDIAPIPE_INDEX


class PoseDetector:
    def __init__(self, model_path: str = "models/pose_landmarker.task",
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        # Imported lazily so the rest of the codebase (and the test suite)
        # doesn't require mediapipe to be installed just to import this module.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._start_time = time.time()

    def detect(self, bgr_frame: np.ndarray) -> Optional[PoseFrame]:
        """bgr_frame: a frame as read by OpenCV (BGR channel order)."""
        rgb = bgr_frame[:, :, ::-1]  # BGR -> RGB, cheap view, no copy needed for mp.Image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        raw = result.pose_landmarks[0]
        frame = PoseFrame(timestamp=time.time())
        for name, idx in MEDIAPIPE_INDEX.items():
            lm = raw[idx]
            frame.landmarks[name] = Landmark(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=getattr(lm, "visibility", 1.0),
            )
        return frame

    def close(self):
        self._landmarker.close()
