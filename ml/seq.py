"""Sequence tensors for the skeleton encoder (numpy only, fully tested).

positions: (T, 15, 2) hip-centred, torso-scaled joint coordinates. A model input window is (C=4, W, V=15):
channels = x, y, dx, dy.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from motion_engine.skeleton import JOINTS, J, fill_gaps, frames_to_array, normalize, Frame

V = len(JOINTS)
SWAP = np.arange(V)
for _l, _r in [("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"), ("left_wrist", "right_wrist"),
               ("left_hip", "right_hip"), ("left_knee", "right_knee"), ("left_ankle", "right_ankle")]:
    SWAP[J[_l]], SWAP[J[_r]] = J[_r], J[_l]
EDGES = [("nose", "shoulder_center"), ("shoulder_center", "left_shoulder"), ("shoulder_center", "right_shoulder"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("shoulder_center", "hip_center"), ("hip_center", "left_hip"),
         ("hip_center", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle")]


def adjacency() -> np.ndarray:
    """Symmetric-normalised adjacency with self loops, (V, V)."""
    A = np.eye(V)
    for a, b in EDGES:
        A[J[a], J[b]] = A[J[b], J[a]] = 1.0
    d = A.sum(1)
    return A / np.sqrt(d[:, None] * d[None, :])


def clip_positions(frames: Sequence[Frame], profile=None) -> np.ndarray:
    arr = normalize(fill_gaps(frames_to_array(frames)), profile)
    xy = np.nan_to_num(arr[:, :, :2], nan=0.0)
    return xy.astype(np.float32)


def channels(xy: np.ndarray) -> np.ndarray:
    """(W, V, 2) -> (4, W, V): positions + frame-to-frame velocity."""
    v = np.zeros_like(xy)
    v[1:] = xy[1:] - xy[:-1]
    return np.concatenate([xy, v], axis=-1).transpose(2, 0, 1).astype(np.float32)


def windows(xy: np.ndarray, window: int = 90, stride: int = 45) -> List[np.ndarray]:
    if len(xy) < window:
        return []
    return [xy[s:s + window] for s in range(0, len(xy) - window + 1, stride)]


def mirror(xy: np.ndarray) -> np.ndarray:
    out = xy[:, SWAP].copy()
    out[..., 0] *= -1
    return out


def resample_time(xy: np.ndarray, n: int) -> np.ndarray:
    t_old, t_new = np.linspace(0, 1, len(xy)), np.linspace(0, 1, n)
    return np.stack([np.stack([np.interp(t_new, t_old, xy[:, j, c]) for c in range(2)], -1) for j in range(V)], 1).astype(np.float32)


def augment(xy_clip: np.ndarray, window: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop at a random speed, then scale / small rotation / mirror / noise / joint dropout."""
    speed = float(rng.uniform(0.8, 1.25))
    span = int(min(len(xy_clip), max(window // 2, round(window * speed))))
    s = int(rng.integers(0, max(1, len(xy_clip) - span + 1)))
    w = resample_time(xy_clip[s:s + span], window)
    ang = np.deg2rad(rng.uniform(-8, 8))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    w = (w @ R.T) * float(rng.uniform(0.92, 1.08))
    if rng.random() < 0.5:
        w = mirror(w)
    w = w + rng.normal(0, 0.01, w.shape).astype(np.float32)
    drop = rng.random(V) < 0.05
    w[:, drop] = 0.0
    return w.astype(np.float32)
