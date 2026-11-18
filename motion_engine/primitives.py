"""Movement primitives: the reusable vocabulary exercises are composed from.

A *signal spec* is plain data (JSON-friendly), e.g.
    {"primitive": "joint_angle", "joints": ["shoulder", "elbow", "wrist"], "side": "mean"}
and `compute_signal(arr, spec)` turns a (T, J, 3) array into a (T,) series.
Exercises never contain code; they reference primitives by name.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .skeleton import J

SIDES = ("left", "right", "mean", "max", "min", "visible")
VISIBLE_MARGIN = 0.15      # "visible": both sides averaged unless one side is clearly better seen
VALID_JOINT_STEMS = {"shoulder", "elbow", "wrist", "hip", "knee", "ankle"}
VALID_JOINTS = set(J) | VALID_JOINT_STEMS


def _name(stem: str, side: str) -> str:
    return stem if stem in J else f"{side}_{stem}"


def _angle(arr, a, b, c) -> np.ndarray:
    """Interior angle (deg) at b between b->a and b->c, per frame."""
    v1 = arr[:, J[a], :2] - arr[:, J[b], :2]
    v2 = arr[:, J[c], :2] - arr[:, J[b], :2]
    cos = np.einsum("ti,ti->t", v1, v2) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _combine(series: List[np.ndarray], side: str) -> np.ndarray:
    st = np.stack(series)
    if side == "max":
        return np.nanmax(st, 0)
    if side == "min":
        return np.nanmin(st, 0)
    return np.nanmean(st, 0)


def _per_side(arr, spec, fn) -> np.ndarray:
    side = spec.get("side", "mean")
    if side == "visible":
        return _visible_side(arr, spec, fn)
    sides = ["left", "right"] if side in ("mean", "max", "min") else [side]
    return _combine([fn(s) for s in sides], side if side in ("max", "min") else "mean")


def _visible_side(arr, spec, fn) -> np.ndarray:
    """Per frame: mean of both sides when both are clearly visible, else the better-visible side. Side-on, the far
    limb is occluded and its landmarks are guesses; averaging it in (side:"mean") corrupts the signal."""
    L, R = fn("left"), fn("right")
    stems = [j for j in spec.get("joints", []) if j not in J]
    if not stems:
        return np.nanmean(np.stack([L, R]), 0)
    vis = {s: np.nanmin(np.stack([arr[:, J[f"{s}_{j}"], 2] for j in stems]), 0) for s in ("left", "right")}
    vl, vr = np.nan_to_num(vis["left"], nan=0.0), np.nan_to_num(vis["right"], nan=0.0)
    both = np.abs(vl - vr) < VISIBLE_MARGIN
    with np.errstate(invalid="ignore"):
        out = np.where(vl >= vr, L, R)
        out = np.where(both & np.isfinite(L) & np.isfinite(R), (L + R) / 2, out)
    return np.where(np.isfinite(out), out, np.where(np.isfinite(L), L, R))


def joint_angle(arr, spec):
    a, b, c = spec["joints"]
    return _per_side(arr, spec, lambda s: _angle(arr, _name(a, s), _name(b, s), _name(c, s)))


def segment_vertical_angle(arr, spec):
    """Angle (deg) between segment a->b and straight down (image +y). 0 = hanging,
    90 = horizontal, 180 = straight up. Used for trunk lean and arm elevation."""
    a, b = spec["joints"]

    def fn(s):
        v = arr[:, J[_name(b, s)], :2] - arr[:, J[_name(a, s)], :2]
        return np.degrees(np.arctan2(np.abs(v[:, 0]), v[:, 1]))  # signed by y: up => >90
    return _per_side(arr, spec, fn)


def relative_height(arr, spec):
    """Vertical offset of joint a above joint b in torso lengths (+ = a is higher)."""
    a, b = spec["joints"]
    return _per_side(arr, spec, lambda s: (arr[:, J[_name(b, s)], 1] - arr[:, J[_name(a, s)], 1]))


def joint_distance(arr, spec):
    a, b = spec["joints"]
    return _per_side(arr, spec, lambda s: np.linalg.norm(arr[:, J[_name(a, s)], :2] - arr[:, J[_name(b, s)], :2], axis=1))


def joint_x_offset(arr, spec):
    """Horizontal offset a - b (torso lengths); used for alignment / sway."""
    a, b = spec["joints"]
    return _per_side(arr, spec, lambda s: arr[:, J[_name(a, s)], 0] - arr[:, J[_name(b, s)], 0])


def trunk_lean(arr, spec):
    """Trunk angle from vertical (deg) using shoulder_center and hip_center."""
    v = arr[:, J["shoulder_center"], :2] - arr[:, J["hip_center"], :2]
    return np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))


def velocity(arr, spec, fps: float = 30.0):
    base = compute_signal(arr, spec["of"], fps)
    return np.gradient(base) * fps


PRIMITIVES: Dict[str, Callable] = {
    "joint_angle": joint_angle,
    "segment_vertical_angle": segment_vertical_angle,
    "relative_height": relative_height,
    "joint_distance": joint_distance,
    "joint_x_offset": joint_x_offset,
    "trunk_lean": trunk_lean,
    "velocity": velocity,
}


def validate_spec(spec: dict, path: str = "signal") -> List[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errs: List[str] = []
    prim = spec.get("primitive")
    if prim not in PRIMITIVES:
        return [f"{path}: unknown primitive {prim!r}"]
    if prim == "velocity":
        return validate_spec(spec.get("of", {}), path + ".of")
    need = {"joint_angle": 3, "segment_vertical_angle": 2, "relative_height": 2,
            "joint_distance": 2, "joint_x_offset": 2, "trunk_lean": 0}[prim]
    joints = spec.get("joints", [])
    if len(joints) != need:
        errs.append(f"{path}: {prim} needs {need} joints, got {len(joints)}")
    errs += [f"{path}: unknown joint {j!r}" for j in joints if j not in VALID_JOINTS]
    if spec.get("side", "mean") not in SIDES:
        errs.append(f"{path}: bad side {spec.get('side')!r}")
    return errs


def compute_signal(arr: np.ndarray, spec: dict, fps: float = 30.0) -> np.ndarray:
    fn = PRIMITIVES[spec["primitive"]]
    return fn(arr, spec, fps) if spec["primitive"] == "velocity" else fn(arr, spec)


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average that tolerates NaN (interpolated) and keeps length."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    bad = np.isnan(x)
    if bad.all():
        return x
    if bad.any():
        x = np.interp(np.arange(len(x)), np.flatnonzero(~bad), x[~bad])
    k = max(1, min(window, len(x)))
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")
