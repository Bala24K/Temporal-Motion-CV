"""Video -> landmarks ingestion (raw video is deleted after extraction by default: privacy)."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from motion_engine.pose import PoseEstimator
from motion_engine.skeleton import Frame
from .schema import ClipMeta
from .store import DatasetStore

MIN_USABLE_RATIO = 0.6
MIN_SECONDS = 1.0


class IngestError(ValueError):
    pass


def extract_poses(video_path: str, estimator: PoseEstimator, max_fps: float = 30.0) -> Tuple[List[Optional[Frame]], List[float], float]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IngestError("could not open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max_fps)))
    frames, ts, i = [], [], 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                t = i / src_fps
                frames.append(estimator.detect(img, t * 1000.0))
                ts.append(t)
            i += 1
    finally:
        cap.release()
    return frames, ts, src_fps / step


def ingest_video(store: DatasetStore, video_path: str, meta: ClipMeta, estimator: PoseEstimator,
                 delete_video: bool = True) -> ClipMeta:
    try:
        frames, ts, fps = extract_poses(video_path, estimator)
        usable = [f for f in frames if f is not None]
        if not frames or len(usable) / len(frames) < MIN_USABLE_RATIO:
            raise IngestError(f"pose found in only {len(usable)}/{len(frames)} frames")
        if (ts[-1] - ts[0]) < MIN_SECONDS:
            raise IngestError("clip shorter than 1 second")
        # keep timing aligned: carry the last good pose over dropout frames (flagged via quality below)
        filled, last = [], usable[0]
        for f in frames:
            last = f if f is not None else last
            filled.append(last)
        meta = meta.model_copy(update={"fps": float(fps), "pose_model": f"{estimator.name}:{estimator.version}"})
        out = store.add_clip(filled, ts, meta)
        return out.model_copy(update={"quality": len(usable) / len(frames)})
    finally:
        if delete_video and os.path.exists(video_path):
            os.remove(video_path)
