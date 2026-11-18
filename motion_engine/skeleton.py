"""Skeleton representation, body profile, and personalised normalisation.

Same 15-node topology as research/motion_upgrade/motion/graph.py (13 MediaPipe
landmarks + hip_center + shoulder_center), kept dependency-free (numpy only).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

JOINTS = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "hip_center", "shoulder_center",
]
J = {n: i for i, n in enumerate(JOINTS)}
REAL_JOINTS = JOINTS[:13]

# Landmark frame: {"left_elbow": (x, y, visibility), ..., "_image": (width, height)} with image coords (y down).
#
# Coordinate contract: pose models (MediaPipe, YOLO/RTMPose `xyn`) return x as a fraction of image WIDTH and y as a
# fraction of image HEIGHT. Those units are anisotropic for any non-square frame, so distances and angles computed on
# them are wrong (a 45 deg elbow reads ~29 deg on a 16:9 camera). Every real frame therefore carries `_image` = [w, h]
# and `frame_to_array` converts to isotropic units (multiples of image height) exactly once. Frames without `_image`
# are treated as already isotropic: that is the synthetic generator's convention. The API boundary rejects real frames
# that lack `_image` (see motion_engine/contract.py).
Frame = Mapping[str, Sequence[float]]
IMAGE_KEY = "_image"


def image_aspect(frame: Frame) -> float:
    """Width / height of the frame's source image (1.0 when the frame is already isotropic)."""
    img = frame.get(IMAGE_KEY)
    if img is None or len(img) < 2:
        return 1.0
    w, h = float(img[0]), float(img[1])
    return w / h if w > 0 and h > 0 else 1.0

SEGMENTS = {
    "upper_arm": (("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")),
    "forearm": (("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")),
    "thigh": (("left_hip", "left_knee"), ("right_hip", "right_knee")),
    "shin": (("left_knee", "left_ankle"), ("right_knee", "right_ankle")),
}


def frame_to_array(frame: Frame, min_visibility: float = 0.3) -> np.ndarray:
    """(J, 3) array of x, y, visibility in ISOTROPIC units (x scaled by the image aspect, see IMAGE_KEY).
    Missing/low-visibility joints get NaN xy."""
    out = np.full((len(JOINTS), 3), np.nan)
    aspect = image_aspect(frame)
    for name in REAL_JOINTS:
        v = frame.get(name)
        if v is None:
            continue
        vis = float(v[2]) if len(v) > 2 else 1.0
        out[J[name], 2] = vis
        if vis >= min_visibility:
            out[J[name], :2] = (float(v[0]) * aspect, float(v[1]))
    for mid, (a, b) in (("hip_center", ("left_hip", "right_hip")),
                        ("shoulder_center", ("left_shoulder", "right_shoulder"))):
        out[J[mid], :2] = (out[J[a], :2] + out[J[b], :2]) / 2
        out[J[mid], 2] = np.nanmin([out[J[a], 2], out[J[b], 2]]) if not np.isnan(out[J[a], 2] + out[J[b], 2]) else np.nan
    return out


def frames_to_array(frames: Sequence[Frame], min_visibility: float = 0.3) -> np.ndarray:
    """(T, J, 3). NaNs mark gaps; callers decide how to handle them."""
    return np.stack([frame_to_array(f, min_visibility) for f in frames]) if frames else np.zeros((0, len(JOINTS), 3))


def fill_gaps(xy: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """Linear-interpolate short NaN runs per joint/axis; longer gaps stay NaN."""
    out = xy.copy()
    T = len(out)
    t = np.arange(T)
    for j in range(out.shape[1]):
        for c in range(2):
            col = out[:, j, c]
            bad = np.isnan(col)
            if not bad.any() or bad.all():
                continue
            filled = np.interp(t, t[~bad], col[~bad])
            # keep NaN where the gap is longer than max_gap
            run = 0
            for i in range(T):
                run = run + 1 if bad[i] else 0
                if bad[i] and run > max_gap:
                    k = i
                    while k >= 0 and bad[k]:
                        filled[k] = np.nan
                        k -= 1
            out[:, j, c] = filled
    return out


@dataclass
class BodyProfile:
    """Approximate, body-specific geometry. NOT an anatomical reconstruction:
    ratios are measured from a few calibration frames in image space and are
    meant for normalisation only."""
    height_cm: Optional[float] = None
    torso_len: float = 0.0            # shoulder_center -> hip_center, image units
    shoulder_width: float = 0.0
    hip_width: float = 0.0
    upper_arm: float = 0.0
    forearm: float = 0.0
    thigh: float = 0.0
    shin: float = 0.0
    n_frames: int = 0
    quality: float = 0.0              # 0..1 overall calibration quality (stability x plausibility x stage coverage)
    stability: float = 0.0            # 0..1 consistency of the front-stage measurement alone
    # --- staged calibration (v2) ---
    calibration_version: int = 1
    stages: list = field(default_factory=list)       # e.g. ["front", "side", "arms_up"]
    shoulder_ratio: float = 0.0       # shoulder width / torso length, measured facing the camera
    hip_ratio: float = 0.0
    cm_per_torso: float = 0.0         # approximate physical scale (needs height)
    plausibility: float = 0.0         # 0..1 agreement with typical human proportions
    # --- calibration v3 ---
    confidence: float = 0.0           # 0..1 landmark visibility during calibration
    coverage: float = 0.0             # 0..1 stages captured (front 0.6, +side 0.8, +arms 1.0)
    status: str = ""                  # good | usable | retry   (quality = min(stability, confidence, coverage))
    warnings: list = field(default_factory=list)
    method: str = "2d"                # "2d" or "2d+3d" (3D world landmarks cross-checked)
    ratio_agreement_3d: Optional[float] = None
    setup: dict = field(default_factory=dict)      # camera-setup fingerprint (see calibration.setup_fingerprint)

    @property
    def calibrated(self) -> bool:
        """Usable for normalisation. Depends on measurement stability, not on how many optional stages were done,
        so a clean front-only calibration counts (overall `quality` still reflects coverage). Older profiles
        without `stability` fall back to `quality`."""
        stab = self.stability if self.stability > 0 else self.quality
        return self.n_frames >= 10 and self.torso_len > 0 and stab >= 0.5

    def scale(self) -> float:
        """Length used to normalise coordinates (torso length, falls back to 1)."""
        return self.torso_len if self.torso_len > 0 else 1.0

    def ratios(self) -> Dict[str, float]:
        s = self.scale()
        return {k: round(getattr(self, k) / s, 4) for k in
                ("shoulder_width", "hip_width", "upper_arm", "forearm", "thigh", "shin")}

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "BodyProfile":
        return BodyProfile(**{k: v for k, v in (d or {}).items() if k in BodyProfile.__dataclass_fields__})


def _seg_len(arr: np.ndarray, a: str, b: str) -> np.ndarray:
    return np.linalg.norm(arr[:, J[a], :2] - arr[:, J[b], :2], axis=1)


def measure_body(frames: Sequence[Frame], height_cm: Optional[float] = None) -> BodyProfile:
    """Median segment lengths over calibration frames; quality = 1 - robust CV."""
    arr = frames_to_array(frames)
    if len(arr) == 0:
        return BodyProfile(height_cm=height_cm)
    torso = _seg_len(arr, "shoulder_center", "hip_center")
    ok = ~np.isnan(torso)
    if ok.sum() < 3:
        return BodyProfile(height_cm=height_cm, n_frames=int(ok.sum()))

    def med(pairs):
        vals = np.concatenate([_seg_len(arr, a, b) for a, b in pairs])
        vals = vals[~np.isnan(vals)]
        return float(np.median(vals)) if len(vals) else 0.0

    t = torso[ok]
    cv = float(np.median(np.abs(t - np.median(t))) / (np.median(t) + 1e-9))
    stability = float(np.clip(1.0 - 4 * cv, 0.0, 1.0))
    return BodyProfile(
        stability=stability,
        height_cm=height_cm,
        torso_len=float(np.median(t)),
        shoulder_width=med([("left_shoulder", "right_shoulder")]),
        hip_width=med([("left_hip", "right_hip")]),
        upper_arm=med(SEGMENTS["upper_arm"]), forearm=med(SEGMENTS["forearm"]),
        thigh=med(SEGMENTS["thigh"]), shin=med(SEGMENTS["shin"]),
        n_frames=int(ok.sum()),
        quality=stability,
    )


def yaw_cosine(arr: np.ndarray, profile: BodyProfile) -> np.ndarray:
    """Per-frame cos(yaw) from shoulder width vs the calibrated front-facing ratio. Clamped to [0.5, 1]
    (beyond ~60 deg the estimate is too noisy to correct). Only meaningful for upright poses."""
    torso = np.linalg.norm(arr[:, J["shoulder_center"], :2] - arr[:, J["hip_center"], :2], axis=1)
    sw = np.linalg.norm(arr[:, J["left_shoulder"], :2] - arr[:, J["right_shoulder"], :2], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (sw / torso) / profile.shoulder_ratio
    return np.clip(np.where(np.isfinite(c), c, 1.0), 0.5, 1.0)


def normalize(arr: np.ndarray, profile: Optional[BodyProfile] = None, compensate_yaw: bool = False) -> np.ndarray:
    """Centre on hip_center per frame and scale by the person's torso length.
    Uses the calibrated profile scale if available, else the per-frame torso.
    `compensate_yaw` un-foreshortens x for FRONTAL-plane motion (lateral raise, press): valid only
    when the exercise's motion lies in the frontal plane and the profile has a front-facing shoulder ratio."""
    out = arr.copy()
    yaw = yaw_cosine(arr, profile) if (compensate_yaw and profile is not None and profile.shoulder_ratio > 0) else None
    xy = out[:, :, :2]
    root = xy[:, J["hip_center"], :][:, None, :]
    xy -= root
    if profile is not None and profile.calibrated:
        s = np.full(len(out), profile.scale())
    else:
        s = np.linalg.norm(arr[:, J["shoulder_center"], :2] - arr[:, J["hip_center"], :2], axis=1)
        s = np.where((s > 1e-6) & ~np.isnan(s), s, np.nan)
    out[:, :, :2] = xy / s[:, None, None]
    if yaw is not None:
        out[:, :, 0] = out[:, :, 0] / yaw[:, None]
    return out
