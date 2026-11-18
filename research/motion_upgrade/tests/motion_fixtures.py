"""
Synthetic skeletons for the motion tests.

Generated analytically rather than recorded, so a test can assert an
exact expected outcome: if the elbow flexion profile is a known sinusoid
with three cycles, the rep counter must find three reps, and any other
answer is a bug rather than a judgement call about real data.

These are geometric fixtures, not a claim that synthetic data is a
substitute for recorded movement — see the existing
`recognition/pose_sequence_synthesis.py` docstring for that distinction.
"""
from typing import Dict, List, Optional

import numpy as np

from pose.landmarks import Landmark, PoseFrame
from motion.graph import JOINT_INDEX, JOINT_NAMES, NUM_JOINTS

# A neutral standing pose in MediaPipe-style image coordinates: x right,
# y DOWN, both roughly in 0..1. Matching the real coordinate convention
# matters because the phase logic reads height as a negated y difference.
NEUTRAL: Dict[str, tuple] = {
    "nose":            (0.50, 0.12),
    "left_shoulder":   (0.42, 0.28),
    "right_shoulder":  (0.58, 0.28),
    "left_elbow":      (0.40, 0.42),
    "right_elbow":     (0.60, 0.42),
    "left_wrist":      (0.39, 0.56),
    "right_wrist":     (0.61, 0.56),
    "left_hip":        (0.45, 0.55),
    "right_hip":       (0.55, 0.55),
    "left_knee":       (0.45, 0.75),
    "right_knee":      (0.55, 0.75),
    "left_ankle":      (0.45, 0.94),
    "right_ankle":     (0.55, 0.94),
}


def make_pose_frame(points: Dict[str, tuple], visibility: float = 0.95,
                    timestamp: float = 0.0) -> PoseFrame:
    """Build a PoseFrame from a name -> (x, y) mapping."""
    frame = PoseFrame(timestamp=timestamp)
    for name, (x, y) in points.items():
        frame.landmarks[name] = Landmark(x=float(x), y=float(y), z=0.0, visibility=visibility)
    return frame


def neutral_clip(n_frames: int = 20, jitter: float = 0.0, seed: int = 0) -> List[PoseFrame]:
    """A stationary person, optionally with small positional noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        pts = {}
        for name, (x, y) in NEUTRAL.items():
            if jitter:
                x += float(rng.normal(0.0, jitter))
                y += float(rng.normal(0.0, jitter))
            pts[name] = (x, y)
        frames.append(make_pose_frame(pts, timestamp=t / 7.5))
    return frames


def curl_clip(n_frames: int = 48, cycles: float = 2.0, amplitude_deg: float = 120.0,
              side: str = "left") -> List[PoseFrame]:
    """
    A bicep curl: the wrist rotates about a fixed elbow so that elbow
    flexion follows a raised cosine with `cycles` complete repetitions.
    """
    frames = []
    forearm = 0.14
    for t in range(n_frames):
        phase = 2.0 * np.pi * cycles * t / max(1, n_frames - 1)
        flexion = np.radians(amplitude_deg) * (1.0 - np.cos(phase)) / 2.0

        pts = dict(NEUTRAL)
        ex, ey = NEUTRAL[f"{side}_elbow"]
        # Direction "straight down" rotated by the flexion angle, so the
        # shoulder-elbow-wrist angle is exactly 180 - flexion degrees.
        sign = -1.0 if side == "left" else 1.0
        dx = sign * np.sin(flexion)
        dy = np.cos(flexion)
        pts[f"{side}_wrist"] = (ex + forearm * dx, ey + forearm * dy)
        frames.append(make_pose_frame(pts, timestamp=t / 7.5))
    return frames


def squat_clip(n_frames: int = 48, cycles: float = 2.0, depth: float = 0.12) -> List[PoseFrame]:
    """
    A squat: hips and shoulders descend while the knees track forward, so
    knee flexion rises and falls `cycles` times. Ankles stay planted.
    """
    frames = []
    for t in range(n_frames):
        phase = 2.0 * np.pi * cycles * t / max(1, n_frames - 1)
        drop = depth * (1.0 - np.cos(phase)) / 2.0

        pts = dict(NEUTRAL)
        for name in ("nose", "left_shoulder", "right_shoulder", "left_elbow",
                     "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"):
            x, y = NEUTRAL[name]
            pts[name] = (x, y + drop)
        # Knees move forward (in x) and down less than the hips, which is
        # what produces knee flexion rather than the whole body sinking.
        for side, direction in (("left", -1.0), ("right", 1.0)):
            x, y = NEUTRAL[f"{side}_knee"]
            pts[f"{side}_knee"] = (x + direction * drop * 0.9, y + drop * 0.25)
        frames.append(make_pose_frame(pts, timestamp=t / 7.5))
    return frames


def coords_from_frames(frames: List[PoseFrame]) -> np.ndarray:
    """
    (T, J, 3) coordinates in the graph's joint order, without
    normalisation — for tests that want to exercise the phase logic on
    known geometry directly.
    """
    coords = np.zeros((len(frames), NUM_JOINTS, 3), dtype=np.float32)
    for t, frame in enumerate(frames):
        for name, lm in frame.landmarks.items():
            idx = JOINT_INDEX.get(name)
            if idx is not None:
                coords[t, idx] = (lm.x, lm.y, lm.z)
        lh, rh = frame.get("left_hip"), frame.get("right_hip")
        ls, rs = frame.get("left_shoulder"), frame.get("right_shoulder")
        if lh and rh:
            coords[t, JOINT_INDEX["hip_center"]] = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2, 0.0)
        if ls and rs:
            coords[t, JOINT_INDEX["shoulder_center"]] = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2, 0.0)
    return coords


def make_fake_cache(path: str, per_class: int = 4, n_frames: int = 40,
                    clips_per_group: int = 1) -> Dict:
    """
    Write a pickle in the same shape as `data/qevd_pose_cache.pkl`, so the
    dataset builder can be tested without the real 6.8 MB cache or any
    video decoding.
    """
    import pickle

    builders = {
        "bicep_curl": lambda i: curl_clip(n_frames, cycles=2.0),
        "squat": lambda i: squat_clip(n_frames, cycles=2.0),
        "shoulder_press": lambda i: press_clip(n_frames, cycles=2.0),
        "lateral_raise": lambda i: raise_clip(n_frames, cycles=2.0),
    }
    cache = {}
    for label, build in builders.items():
        for i in range(per_class):
            clip_id = f"{label}_{i:03d}.mp4"
            cache[clip_id] = {
                "video_id": clip_id,
                "pose_frames": build(i),
                "label": label,
            }
    with open(path, "wb") as fh:
        pickle.dump(cache, fh)
    return cache


def press_clip(n_frames: int = 48, cycles: float = 2.0, travel: float = 0.30) -> List[PoseFrame]:
    """A shoulder press: both wrists travel from shoulder height to overhead."""
    frames = []
    for t in range(n_frames):
        phase = 2.0 * np.pi * cycles * t / max(1, n_frames - 1)
        lift = travel * (1.0 - np.cos(phase)) / 2.0
        pts = dict(NEUTRAL)
        for side in ("left", "right"):
            sx, sy = NEUTRAL[f"{side}_shoulder"]
            ex, ey = NEUTRAL[f"{side}_elbow"]
            pts[f"{side}_elbow"] = (ex, ey - lift * 0.6)
            pts[f"{side}_wrist"] = (sx, sy - lift)
        frames.append(make_pose_frame(pts, timestamp=t / 7.5))
    return frames


def raise_clip(n_frames: int = 48, cycles: float = 2.0, span: float = 0.16) -> List[PoseFrame]:
    """A lateral raise: arms abduct sideways to shoulder height, elbows near-straight."""
    frames = []
    for t in range(n_frames):
        phase = 2.0 * np.pi * cycles * t / max(1, n_frames - 1)
        amount = (1.0 - np.cos(phase)) / 2.0
        pts = dict(NEUTRAL)
        for side, direction in (("left", -1.0), ("right", 1.0)):
            sx, sy = NEUTRAL[f"{side}_shoulder"]
            pts[f"{side}_elbow"] = (sx + direction * span * amount, sy + 0.14 * (1 - amount))
            pts[f"{side}_wrist"] = (sx + direction * 2 * span * amount, sy + 0.28 * (1 - amount))
        frames.append(make_pose_frame(pts, timestamp=t / 7.5))
    return frames
