"""
Core geometry engine. Everything else (rep counting, ROM, form checks)
is built on top of these primitives.
"""
import numpy as np


def angle_3pt(a, b, c) -> float | None:
    """
    Angle ABC in degrees: the angle at vertex B formed by rays B->A and B->C.
    a, b, c are (x, y) or (x, y, z) points (tuples, lists, or arrays).

    Example: angle_3pt(shoulder, elbow, wrist) -> elbow flexion angle.

    Returns None if either segment has ~zero length (degenerate — usually
    means a landmark is missing/garbage; callers should treat None like a
    dropped reading, not a real 0-degree angle).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)  # guard against fp drift outside [-1, 1]
    return float(np.degrees(np.arccos(cosine)))


def angular_velocity(angle_prev: float | None, angle_curr: float | None,
                      dt: float) -> float | None:
    """Degrees per second between two angle readings. None if either
    reading is missing or dt is non-positive (avoid div-by-zero / bogus
    spikes from duplicate timestamps)."""
    if angle_prev is None or angle_curr is None or dt <= 0:
        return None
    return (angle_curr - angle_prev) / dt


def distance(a, b) -> float:
    """Euclidean distance between two points."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def normalized_displacement(a, b, scale: float | None) -> float | None:
    """
    Distance between two points, divided by a body-scale reference
    (e.g. torso length), so a threshold like "moved more than 0.15 units"
    means roughly the same thing for someone standing close to the camera
    as someone standing far away. Returns None if scale is missing/zero.
    """
    if not scale or scale < 1e-6:
        return None
    return distance(a, b) / scale
