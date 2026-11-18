"""Landmark-frame contract enforced at the API boundary (WebSocket and REST).

A real frame must say which image it came from (`_image` = [width, height] in pixels) so the engine can convert the
pose model's per-axis normalised coordinates into isotropic geometry (see skeleton.IMAGE_KEY). Without it every
angle and ratio is silently distorted, which is the root cause of the calibration failures found in
docs/POSE_CALIBRATION_AUDIT.md (RC-1). So frames without it are rejected instead of guessed at.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from .skeleton import IMAGE_KEY, REAL_JOINTS

MAX_SIDE_PX = 16384


def frame_error(frame: object) -> Optional[str]:
    """None if the frame satisfies the contract, else a human-readable reason."""
    if not isinstance(frame, Mapping):
        return "frame must be an object of landmarks"
    img = frame.get(IMAGE_KEY)
    if not isinstance(img, Sequence) or len(img) < 2:
        return f"frame is missing '{IMAGE_KEY}': [width, height] of the source image in pixels"
    try:
        w, h = float(img[0]), float(img[1])
    except (TypeError, ValueError):
        return f"'{IMAGE_KEY}' must be two numbers"
    if not (0 < w <= MAX_SIDE_PX and 0 < h <= MAX_SIDE_PX):
        return f"'{IMAGE_KEY}' dimensions out of range"
    for name in REAL_JOINTS:
        v = frame.get(name)
        if v is None:
            continue
        if not isinstance(v, Sequence) or len(v) < 2:
            return f"landmark {name!r} must be [x, y, visibility]"
        try:
            if not all(math.isfinite(float(c)) for c in v[:3]):
                return f"landmark {name!r} has a non-finite value"
        except (TypeError, ValueError):
            return f"landmark {name!r} must contain numbers"
    return None


def frames_error(frames: Sequence[object]) -> Optional[str]:
    for i, f in enumerate(frames):
        err = frame_error(f)
        if err:
            return f"frame {i}: {err}"
    return None
