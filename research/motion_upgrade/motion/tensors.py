"""
Skeleton sequence tensors — the representation change at the centre of
this version.

The previous recognition path flattened a window into ~870 engineered
numbers and handed them to a Random Forest. Here the window stays in its
natural shape:

    (T, J, C)   T frames, J joints, C coordinate channels

and the joints keep their identity, so a graph convolution can use the
skeleton topology (`motion/graph.py`). The engineered kinematics are not
deleted — they move to a parallel branch whose job is explaining the
movement, not classifying it.

WHAT THIS MODULE IS RESPONSIBLE FOR
    1. PoseFrame list -> (T, J, 3) array in a body-relative frame, using
       the existing `pose.normalization.normalize_frame` so the learned
       model and the kinematic branch see the same normalisation.
    2. Honest handling of missing data. MediaPipe drops joints; the
       normaliser returns None when the basis landmarks are absent. Both
       produce gaps, and a gap filled with 0.0 is a joint sitting exactly
       at the pelvis — an anatomically meaningful, wrong value. Gaps are
       interpolated where that is defensible, held where it isn't, and
       always reported through a validity mask.
    3. Fixed-length resampling, so clips of 5 to 54 frames become
       uniform windows the model can batch.
    4. Derived streams: bone vectors, joint motion, bone motion.

Nothing here imports torch — the model consumes plain float32 arrays.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pose.landmarks import PoseFrame
from pose.normalization import normalize_frame
from motion.graph import (
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    ROOT_INDEX,
    parent_array,
)

# Channel count of the base coordinate stream (x, y, z).
NUM_COORDS = 3

# MediaPipe's z is a relative-depth estimate from a single camera and is
# substantially noisier than x/y — `pose/normalization.py` already avoids
# it for the torso-length basis. It is kept as a channel because the
# model can learn to downweight it, but it is scaled down so it cannot
# dominate the input distribution before training starts.
Z_CHANNEL_SCALE = 0.3

STREAM_NAMES: Tuple[str, ...] = ("joint", "bone", "joint_motion", "bone_motion")


@dataclass
class SkeletonWindow:
    """
    One fixed-length window of movement, plus the provenance a training
    pipeline needs to decide whether to trust it.

    coords       (T, J, 3) float32, body-relative normalised coordinates
    valid        (T, J)    bool,  True where the coordinate came from a
                                  detected landmark rather than being
                                  interpolated, held, or fabricated
    frame_valid  (T,)      bool,  True where normalisation succeeded
    source_fps   float,            effective sampling rate after the
                                   extractor's frame_step, used to put
                                   velocity in per-second units
    """
    coords: np.ndarray
    valid: np.ndarray
    frame_valid: np.ndarray
    source_fps: float = 7.5  # 30 fps / frame_step 4, the extractor's default

    @property
    def num_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def valid_ratio(self) -> float:
        """Fraction of joint-frames that are real detections. A useful
        quality filter: windows below ~0.5 are mostly invention."""
        return float(self.valid.mean()) if self.valid.size else 0.0

    @property
    def frame_valid_ratio(self) -> float:
        return float(self.frame_valid.mean()) if self.frame_valid.size else 0.0


def _midpoint(a, b) -> Optional[np.ndarray]:
    if a is None or b is None:
        return None
    return np.array([(a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0], dtype=np.float64)


def frames_to_raw_array(
    pose_frames: Sequence[PoseFrame],
    min_visibility: float = 0.4,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PoseFrame sequence -> (coords, valid, frame_valid) with gaps left as NaN.

    coords      (T, J, 3) float64 with NaN where a joint is missing
    valid       (T, J)    bool
    frame_valid (T,)      bool — normalisation succeeded for this frame

    Frames whose normalisation basis is missing are kept in place (so the
    time axis stays uniform) but marked invalid and left entirely NaN;
    dropping them instead would silently compress time and corrupt every
    velocity computed downstream.
    """
    n = len(pose_frames)
    coords = np.full((n, NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float64)
    valid = np.zeros((n, NUM_JOINTS), dtype=bool)
    frame_valid = np.zeros((n,), dtype=bool)

    for t, raw in enumerate(pose_frames):
        frame = normalize_frame(raw, min_visibility=min_visibility) if normalize else raw
        if frame is None:
            continue
        frame_valid[t] = True

        for name, lm in frame.landmarks.items():
            idx = JOINT_INDEX.get(name)
            if idx is None:
                continue  # a landmark we don't model (MediaPipe has 33; we use 13)
            if lm.visibility < min_visibility:
                continue
            coords[t, idx] = (lm.x, lm.y, lm.z)
            valid[t, idx] = True

        # Synthetic midline nodes, from the normalised frame so they live
        # in the same coordinate system as everything else.
        hip = _midpoint(frame.get("left_hip"), frame.get("right_hip"))
        sho = _midpoint(frame.get("left_shoulder"), frame.get("right_shoulder"))
        if hip is not None:
            coords[t, JOINT_INDEX["hip_center"]] = hip
            valid[t, JOINT_INDEX["hip_center"]] = True
        if sho is not None:
            coords[t, JOINT_INDEX["shoulder_center"]] = sho
            valid[t, JOINT_INDEX["shoulder_center"]] = True

    return coords, valid, frame_valid


def fill_gaps(coords: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """
    Fill NaN gaps per joint per channel by linear interpolation over the
    time axis, but only across runs no longer than `max_gap` frames.

    Longer gaps are held at the nearest observed value instead of being
    interpolated. Interpolating a one-second occlusion invents a smooth
    trajectory the person never performed, and the model has no way to
    tell that apart from a real slow movement; holding at least produces
    a visibly static joint, and the validity mask records that neither is
    real. A joint never observed at all in the window is left NaN for the
    caller to resolve.
    """
    out = coords.copy()
    T = out.shape[0]
    if T == 0:
        return out

    flat = out.reshape(T, -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        observed = np.flatnonzero(~np.isnan(col))
        if observed.size == 0:
            continue
        if observed.size == 1:
            col[:] = col[observed[0]]
            continue

        # Interior gaps.
        for a, b in zip(observed[:-1], observed[1:]):
            gap = int(b - a - 1)
            if gap <= 0:
                continue
            if gap <= max_gap:
                col[a + 1:b] = np.linspace(col[a], col[b], gap + 2)[1:-1]
            else:
                mid = a + gap // 2 + 1
                col[a + 1:mid] = col[a]
                col[mid:b] = col[b]

        # Leading / trailing gaps are always held — there is nothing to
        # interpolate toward.
        col[:observed[0]] = col[observed[0]]
        col[observed[-1] + 1:] = col[observed[-1]]

    return flat.reshape(out.shape)


def resample_time(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample along axis 0 to exactly `target_len` frames by linear
    interpolation at uniformly spaced positions.

    Used to make variable-length clips batchable. Note the tradeoff: this
    normalises away tempo, so a fast squat and a slow squat become the
    same tensor. That is the right default for *recognising* the
    exercise, and it is why tempo is measured on the kinematic branch
    (which sees real timestamps) rather than being expected to survive
    here.
    """
    T = arr.shape[0]
    if T == target_len:
        return arr.astype(np.float32, copy=True)
    if T == 0:
        return np.zeros((target_len,) + arr.shape[1:], dtype=np.float32)
    if T == 1:
        return np.repeat(arr.astype(np.float32), target_len, axis=0)

    src = np.linspace(0.0, T - 1, num=T)
    dst = np.linspace(0.0, T - 1, num=target_len)
    flat = arr.reshape(T, -1).astype(np.float64)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape((target_len,) + arr.shape[1:]).astype(np.float32)


def resample_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbour resample for boolean masks — interpolating a
    validity flag would produce fractional truth."""
    T = mask.shape[0]
    if T == target_len:
        return mask.copy()
    if T == 0:
        return np.zeros((target_len,) + mask.shape[1:], dtype=bool)
    idx = np.clip(np.round(np.linspace(0, T - 1, target_len)).astype(int), 0, T - 1)
    return mask[idx]


def build_window(
    pose_frames: Sequence[PoseFrame],
    target_len: int = 64,
    min_visibility: float = 0.4,
    max_gap: int = 4,
    source_fps: float = 7.5,
) -> Optional[SkeletonWindow]:
    """
    Full path from a PoseFrame sequence to a model-ready window.

    Returns None when no frame in the sequence could be normalised at all
    — there is no honest tensor to build from that, and returning zeros
    would hand the model a "person standing perfectly still at the
    origin" example labelled as an exercise.
    """
    if not pose_frames:
        return None

    coords, valid, frame_valid = frames_to_raw_array(
        pose_frames, min_visibility=min_visibility, normalize=True
    )
    if not frame_valid.any():
        return None

    filled = fill_gaps(coords, max_gap=max_gap)

    # Joints never seen anywhere in the window: place at the root (the
    # pelvis, which is the origin after normalisation) and leave them
    # marked invalid. Downstream, `valid` is available as a model input
    # channel so the network can learn to discount them.
    filled = np.nan_to_num(filled, nan=0.0)

    filled[:, :, 2] *= Z_CHANNEL_SCALE

    coords_rs = resample_time(filled, target_len)
    valid_rs = resample_mask(valid, target_len)
    frame_valid_rs = resample_mask(frame_valid, target_len)

    return SkeletonWindow(
        coords=coords_rs.astype(np.float32),
        valid=valid_rs,
        frame_valid=frame_valid_rs,
        source_fps=float(source_fps),
    )


def bone_stream(coords: np.ndarray) -> np.ndarray:
    """
    Bone vectors: child position minus parent position, shape (T, J, 3).

    The root's bone is the zero vector (it has no parent). Bones are
    translation-invariant by construction and encode limb orientation and
    length directly, which is why the joint/bone two-stream split is
    standard for skeleton action recognition.
    """
    parent = parent_array()
    bones = coords - coords[:, parent, :]
    bones[:, ROOT_INDEX, :] = 0.0
    return bones.astype(np.float32)


def motion_stream(arr: np.ndarray) -> np.ndarray:
    """
    First temporal difference, shape unchanged. The first frame is zero
    (no predecessor) rather than wrapped — wrapping would fabricate a
    large spurious velocity at every window boundary.
    """
    out = np.zeros_like(arr)
    if arr.shape[0] > 1:
        out[1:] = arr[1:] - arr[:-1]
    return out.astype(np.float32)


def build_streams(
    window: SkeletonWindow,
    streams: Sequence[str] = STREAM_NAMES,
    include_validity_channel: bool = True,
) -> np.ndarray:
    """
    Assemble the model input, shape (C_total, T, J).

    Channel-first with joints last is the ST-GCN convention: the temporal
    convolution then runs over a contiguous axis and the graph
    multiplication is a plain matmul over the last axis.

    `include_validity_channel` appends the per-joint validity mask as one
    extra channel. Giving the network an explicit "this joint was
    guessed" signal is cheap and stops it from reading interpolated
    stillness as real stillness.
    """
    unknown = [s for s in streams if s not in STREAM_NAMES]
    if unknown:
        raise ValueError(f"unknown stream(s): {unknown}; expected any of {list(STREAM_NAMES)}")
    if not streams:
        raise ValueError("at least one stream is required")

    joints = window.coords
    bones = bone_stream(joints)
    table = {
        "joint": joints,
        "bone": bones,
        "joint_motion": motion_stream(joints),
        "bone_motion": motion_stream(bones),
    }

    parts = [table[s] for s in streams]                     # each (T, J, 3)
    stacked = np.concatenate(parts, axis=2)                 # (T, J, 3 * n_streams)
    out = np.transpose(stacked, (2, 0, 1))                  # (C, T, J)

    if include_validity_channel:
        mask = window.valid.astype(np.float32)[None, :, :]  # (1, T, J)
        out = np.concatenate([out, mask], axis=0)

    return np.ascontiguousarray(out, dtype=np.float32)


def stream_channel_count(streams: Sequence[str] = STREAM_NAMES,
                         include_validity_channel: bool = True) -> int:
    """Input channel count for a given stream configuration — used to
    construct the model without building a dummy sample first."""
    return len(streams) * NUM_COORDS + (1 if include_validity_channel else 0)


def describe_channels(streams: Sequence[str] = STREAM_NAMES,
                      include_validity_channel: bool = True) -> List[str]:
    """Human-readable channel names, for debugging and for saliency plots
    where an unlabelled channel index is useless."""
    names: List[str] = []
    for s in streams:
        names.extend(f"{s}_{axis}" for axis in ("x", "y", "z"))
    if include_validity_channel:
        names.append("valid")
    return names
