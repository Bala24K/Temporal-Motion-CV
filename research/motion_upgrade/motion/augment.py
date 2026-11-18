"""
Augmentation for skeleton windows.

With a few hundred clips, augmentation is doing more work than any
architecture choice. Every transform here is one that leaves the exercise
label genuinely unchanged — which rules out some standard ones:

  * Horizontal flip must permute the left/right joint indices as well as
    negating x. Negating x alone puts the subject's right wrist where the
    left wrist node is, and the model learns anatomy that doesn't exist.
  * Temporal speed jitter is safe for recognition but changes tempo, so
    it is only ever applied to the learned branch's input. The kinematic
    branch measures tempo from real timestamps and never sees augmented
    data.
  * No vertical flip. An upside-down squat is not an exercise, and the
    y axis carries the gravity direction the whole phase logic depends on.

All transforms operate on a SkeletonWindow and return a new one; nothing
is modified in place, so a sample can be augmented differently each epoch
without the dataset drifting.
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from motion.graph import mirror_permutation
from motion.tensors import SkeletonWindow, resample_mask, resample_time

_MIRROR_PERM = mirror_permutation()


@dataclass
class AugmentConfig:
    """
    Probabilities and magnitudes. Defaults are deliberately moderate: on
    a small corpus, aggressive geometric augmentation mostly teaches the
    model to undo the augmentation.
    """
    mirror_prob: float = 0.5
    rotate_prob: float = 0.5
    rotate_max_deg: float = 12.0
    scale_prob: float = 0.5
    scale_range: tuple = (0.9, 1.1)
    translate_prob: float = 0.3
    translate_std: float = 0.04
    speed_prob: float = 0.5
    speed_range: tuple = (0.8, 1.25)
    noise_prob: float = 0.5
    noise_std: float = 0.012
    joint_dropout_prob: float = 0.3
    joint_dropout_max: int = 2
    time_mask_prob: float = 0.2
    time_mask_max_frames: int = 4

    @classmethod
    def off(cls) -> "AugmentConfig":
        """No-op config, for validation and test loaders."""
        return cls(
            mirror_prob=0.0, rotate_prob=0.0, scale_prob=0.0, translate_prob=0.0,
            speed_prob=0.0, noise_prob=0.0, joint_dropout_prob=0.0, time_mask_prob=0.0,
        )


def mirror(window: SkeletonWindow) -> SkeletonWindow:
    """Left/right flip: negate x and permute the joint axis."""
    coords = window.coords.copy()
    coords[:, :, 0] *= -1.0
    coords = coords[:, _MIRROR_PERM, :]
    valid = window.valid[:, _MIRROR_PERM]
    return replace(window, coords=coords.astype(np.float32), valid=valid)


def rotate_xy(window: SkeletonWindow, degrees: float) -> SkeletonWindow:
    """
    In-plane rotation about the origin (the pelvis, post-normalisation).

    Small angles only. Normalisation already levels the shoulder line, so
    this is simulating residual camera roll and pose-estimator jitter in
    that alignment, not a genuinely different viewpoint. Real viewpoint
    change is a 3D problem this cannot fake.
    """
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    coords = window.coords.copy()
    xy = coords[:, :, :2].reshape(-1, 2) @ rot.T
    coords[:, :, :2] = xy.reshape(coords.shape[0], coords.shape[1], 2)
    return replace(window, coords=coords.astype(np.float32))


def scale(window: SkeletonWindow, factor: float) -> SkeletonWindow:
    """Uniform scale — residual body-size variation after torso normalisation."""
    return replace(window, coords=(window.coords * float(factor)).astype(np.float32))


def translate(window: SkeletonWindow, offset: np.ndarray) -> SkeletonWindow:
    """
    Whole-body translation.

    After normalisation the pelvis sits at the origin, so this is
    simulating error in the normalisation basis itself rather than the
    subject standing elsewhere in frame.
    """
    coords = window.coords + offset.astype(np.float32)[None, None, :]
    return replace(window, coords=coords.astype(np.float32))


def time_scale(window: SkeletonWindow, factor: float) -> SkeletonWindow:
    """
    Speed jitter: resample to a different effective duration, then back to
    the original length. Frames outside the original range are clipped, so
    a slowed clip covers less of the movement and a sped-up one covers
    more of it — which is the point.
    """
    T = window.num_frames
    if T < 2 or abs(factor - 1.0) < 1e-3:
        return window
    stretched = max(2, int(round(T * float(factor))))
    coords = resample_time(window.coords, stretched)
    valid = resample_mask(window.valid, stretched)
    frame_valid = resample_mask(window.frame_valid, stretched)

    if stretched > T:  # sped up: take a random contiguous slice
        start = 0 if stretched == T else np.random.randint(0, stretched - T + 1)
        coords, valid, frame_valid = coords[start:start + T], valid[start:start + T], frame_valid[start:start + T]
    else:              # slowed: resample back up
        coords = resample_time(coords, T)
        valid = resample_mask(valid, T)
        frame_valid = resample_mask(frame_valid, T)

    return replace(
        window,
        coords=coords.astype(np.float32),
        valid=valid,
        frame_valid=frame_valid,
        source_fps=window.source_fps * float(factor),
    )


def add_noise(window: SkeletonWindow, std: float, rng: np.random.Generator) -> SkeletonWindow:
    """
    Per-coordinate Gaussian jitter, applied only to detected joints.

    Adding noise to interpolated joints would make a fabricated value look
    like a noisy measurement, which is precisely the distinction the
    validity channel exists to preserve.
    """
    noise = rng.normal(0.0, std, size=window.coords.shape).astype(np.float32)
    noise *= window.valid[:, :, None].astype(np.float32)
    return replace(window, coords=(window.coords + noise).astype(np.float32))


def joint_dropout(window: SkeletonWindow, n_joints: int, rng: np.random.Generator) -> SkeletonWindow:
    """
    Simulate occlusion: hold a few joints at their first value for the
    whole window and mark them invalid.

    This is the single most useful augmentation for live webcam use,
    where a wrist leaving frame is routine and the model otherwise sees
    an input distribution it never met in training.
    """
    if n_joints <= 0 or window.coords.shape[1] == 0:
        return window
    J = window.coords.shape[1]
    victims = rng.choice(J, size=min(n_joints, J), replace=False)
    coords = window.coords.copy()
    valid = window.valid.copy()
    for j in victims:
        coords[:, j, :] = coords[0, j, :]
        valid[:, j] = False
    return replace(window, coords=coords, valid=valid)


def time_mask(window: SkeletonWindow, max_frames: int, rng: np.random.Generator) -> SkeletonWindow:
    """
    Freeze a short run of frames — the pose estimator stalling or the
    capture thread dropping frames, which the async runtime makes a
    routine event rather than an anomaly.
    """
    T = window.num_frames
    if T < 3 or max_frames <= 0:
        return window
    span = int(rng.integers(1, min(max_frames, T - 1) + 1))
    start = int(rng.integers(0, T - span))
    coords = window.coords.copy()
    frame_valid = window.frame_valid.copy()
    coords[start:start + span] = coords[start]
    frame_valid[start:start + span] = False
    return replace(window, coords=coords, frame_valid=frame_valid)


def augment_window(
    window: SkeletonWindow,
    config: AugmentConfig,
    rng: Optional[np.random.Generator] = None,
) -> SkeletonWindow:
    """
    Apply the configured random transforms in a fixed order.

    Order matters: geometric transforms first (they compose cleanly),
    then temporal, then corruptions. Adding noise before a rotation would
    have the rotation smooth the noise away.
    """
    rng = rng or np.random.default_rng()
    out = window

    if rng.random() < config.mirror_prob:
        out = mirror(out)
    if rng.random() < config.rotate_prob:
        out = rotate_xy(out, float(rng.uniform(-config.rotate_max_deg, config.rotate_max_deg)))
    if rng.random() < config.scale_prob:
        out = scale(out, float(rng.uniform(*config.scale_range)))
    if rng.random() < config.translate_prob:
        out = translate(out, rng.normal(0.0, config.translate_std, size=3))
    if rng.random() < config.speed_prob:
        out = time_scale(out, float(rng.uniform(*config.speed_range)))
    if rng.random() < config.noise_prob:
        out = add_noise(out, config.noise_std, rng)
    if rng.random() < config.joint_dropout_prob:
        out = joint_dropout(out, int(rng.integers(1, config.joint_dropout_max + 1)), rng)
    if rng.random() < config.time_mask_prob:
        out = time_mask(out, config.time_mask_max_frames, rng)

    return out
