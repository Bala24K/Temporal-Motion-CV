"""Pose-estimator interface + adapters + benchmark harness.

Every estimator returns the same `Frame` ({joint: [x, y, visibility]}, normalised image coords), so the
engine never knows which model produced it. Adapters for RTMPose / YOLO-Pose import their libraries lazily
and raise `PoseUnavailable` with install instructions. Only the pure conversion functions
(`from_coco17`, `from_mediapipe33`) and the benchmark are unit-tested here; the library-backed adapters
need those libraries and real video, which this sandbox lacks - benchmark them on YOUR footage before
trusting them (`python -m ml.benchmark_pose`).
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .skeleton import IMAGE_KEY, J, REAL_JOINTS, Frame, image_aspect

COCO17 = {"nose": 0, "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9,
          "right_wrist": 10, "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14,
          "left_ankle": 15, "right_ankle": 16}
MP33 = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24, "left_knee": 25,
        "right_knee": 26, "left_ankle": 27, "right_ankle": 28}


class PoseUnavailable(RuntimeError):
    pass


def from_coco17(kpts: np.ndarray, scores: np.ndarray, width: int = 1, height: int = 1,
                image_size: Optional[Tuple[int, int]] = None) -> Frame:
    """(17,2) keypoints + (17,) scores -> Frame (contract: per-axis normalised coords + `_image`).
    Pixel keypoints: pass width/height. Already-normalised keypoints (e.g. YOLO `xyn`): pass `image_size`."""
    f: Dict[str, list] = {n: [float(kpts[i, 0]) / width, float(kpts[i, 1]) / height, float(scores[i])]
                          for n, i in COCO17.items()}
    f[IMAGE_KEY] = list(image_size) if image_size else [width, height]
    return f


def from_mediapipe33(landmarks: Sequence, image_size: Optional[Tuple[int, int]] = None) -> Frame:
    """Sequence of objects with .x .y .visibility (MediaPipe) -> Frame. `image_size` = (width, height) in pixels;
    required for real images (MediaPipe x/y are normalised per axis), omit only for isotropic test data."""
    f: Dict[str, list] = {n: [float(landmarks[i].x), float(landmarks[i].y), float(getattr(landmarks[i], "visibility", 1.0))]
                          for n, i in MP33.items()}
    if image_size:
        f[IMAGE_KEY] = list(image_size)
    return f


class PoseEstimator(ABC):
    name = "base"
    version = "0"

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, t_ms: float) -> Optional[Frame]:
        """Return landmarks for the most prominent person, or None."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class MediaPipeEstimator(PoseEstimator):
    """MediaPipe Tasks PoseLandmarker. Needs the .task model file (path via arg or MEDIAPIPE_POSE_MODEL)."""
    name = "mediapipe"

    def __init__(self, model_path: Optional[str] = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
        except Exception as e:                       # pragma: no cover - needs mediapipe
            raise PoseUnavailable("pip install mediapipe") from e
        path = model_path or os.getenv("MEDIAPIPE_POSE_MODEL")
        if not path or not os.path.exists(path):
            raise PoseUnavailable("Download pose_landmarker_*.task and set MEDIAPIPE_POSE_MODEL")
        self._mp = mp
        opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=path),
                                            running_mode=vision.RunningMode.VIDEO, num_poses=1)
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self.version = os.path.basename(path)

    def detect(self, image_bgr, t_ms):                # pragma: no cover - needs mediapipe
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), int(t_ms))
        h, w = image_bgr.shape[:2]
        return from_mediapipe33(res.pose_landmarks[0], (w, h)) if res.pose_landmarks else None

    def close(self):                                  # pragma: no cover
        self._lm.close()


class YoloPoseEstimator(PoseEstimator):
    name = "yolo-pose"

    def __init__(self, weights: str = "yolov8n-pose.pt"):
        try:
            from ultralytics import YOLO
        except Exception as e:                        # pragma: no cover
            raise PoseUnavailable("pip install ultralytics") from e
        self._m, self.version = YOLO(weights), weights

    def detect(self, image_bgr, t_ms):                # pragma: no cover
        r = self._m(image_bgr, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.xyn) == 0:
            return None
        i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        h, w = image_bgr.shape[:2]
        return from_coco17(r.keypoints.xyn[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy(), image_size=(w, h))


class RTMPoseEstimator(PoseEstimator):
    name = "rtmpose"

    def __init__(self, mode: str = "lightweight"):
        try:
            from rtmlib import Body
        except Exception as e:                        # pragma: no cover
            raise PoseUnavailable("pip install rtmlib onnxruntime") from e
        self._b, self.version = Body(mode=mode, backend="onnxruntime", device="cpu"), f"rtmlib-{mode}"

    def detect(self, image_bgr, t_ms):                # pragma: no cover
        kp, sc = self._b(image_bgr)
        if len(kp) == 0:
            return None
        h, w = image_bgr.shape[:2]
        return from_coco17(kp[0], sc[0], w, h)


REGISTRY: Dict[str, Callable[..., PoseEstimator]] = {
    "mediapipe": MediaPipeEstimator, "yolo-pose": YoloPoseEstimator, "rtmpose": RTMPoseEstimator}


def make_estimator(name: str, **kw) -> PoseEstimator:
    if name not in REGISTRY:
        raise KeyError(f"unknown estimator {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)


# ----------------------------------------------------------------------------- benchmark
@dataclass
class BenchResult:
    name: str
    frames: int
    detection_rate: float
    fps: float
    latency_ms_p50: float
    latency_ms_p95: float
    jitter: float                      # mean |2nd difference| of joints, in torso lengths (noise proxy)
    error: Optional[float] = None      # mean joint error vs ground truth, in torso lengths
    pck: Optional[float] = None        # share of joints within 0.1 torso lengths of ground truth

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _xy(frame: Frame) -> np.ndarray:
    xy = np.array([frame[j][:2] if j in frame else (np.nan, np.nan) for j in REAL_JOINTS], float)
    xy[:, 0] *= image_aspect(frame)          # isotropic units, so errors and jitter are comparable across aspects
    return xy


def _torso(xy: np.ndarray) -> float:
    sc = (xy[REAL_JOINTS.index("left_shoulder")] + xy[REAL_JOINTS.index("right_shoulder")]) / 2
    hc = (xy[REAL_JOINTS.index("left_hip")] + xy[REAL_JOINTS.index("right_hip")]) / 2
    return float(np.linalg.norm(sc - hc))


def benchmark(estimator: PoseEstimator, samples: Iterable[Tuple[np.ndarray, Optional[Frame]]],
              fps_nominal: float = 30.0) -> BenchResult:
    """samples: (image_bgr, ground_truth_frame_or_None). Measures speed, dropouts, jitter, accuracy."""
    lat, preds, errs, hits, n = [], [], [], [], 0
    for i, (img, gt) in enumerate(samples):
        n += 1
        t0 = time.perf_counter()
        out = estimator.detect(img, i * 1000.0 / fps_nominal)
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(None if out is None else _xy(out))
        if out is not None and gt is not None:
            g, p = _xy(gt), _xy(out)
            d = np.linalg.norm(g - p, axis=1) / max(_torso(g), 1e-6)
            d = d[np.isfinite(d)]
            if len(d):
                errs.append(d.mean())
                hits.append((d < 0.1).mean())
    ok = [p for p in preds if p is not None]
    jit = []
    for a, b, c in zip(preds, preds[1:], preds[2:]):
        if a is not None and b is not None and c is not None:
            t = max(_torso(b), 1e-6)
            jit.append(np.nanmean(np.linalg.norm(a - 2 * b + c, axis=1)) / t)
    lat_s = sorted(lat) or [0.0]
    return BenchResult(estimator.name, n, len(ok) / max(n, 1), 1000.0 / max(np.mean(lat), 1e-9) if lat else 0.0,
                       lat_s[len(lat_s) // 2], lat_s[min(len(lat_s) - 1, int(len(lat_s) * 0.95))],
                       float(np.mean(jit)) if jit else float("nan"),
                       float(np.mean(errs)) if errs else None, float(np.mean(hits)) if hits else None)
