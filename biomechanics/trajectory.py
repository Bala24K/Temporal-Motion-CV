"""
Trajectory comparison via Dynamic Time Warping (V2 section 14).

DTW compares two sequences by their SHAPE, tolerating different speeds —
a slow, careful rep and a fast rep of the same movement should compare as
similar; a rep that follows a genuinely different path should not, even
if it happens to take the same amount of time.

This is a plain O(N*M) DP implementation — no external DTW dependency.
For V1/V2's per-rep sequences (tens to low hundreds of samples) this is
fast enough; if trajectories grow much longer, a banded/windowed DTW
(restricting the warping path to near the diagonal) would be the next
step rather than reaching for a compiled library right away.
"""
from typing import List, Sequence, Tuple, Union
import numpy as np

Point = Union[float, Tuple[float, float], Tuple[float, float, float]]


def _as_points(seq: Sequence[Point]) -> np.ndarray:
    arr = np.asarray(seq, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def dtw_distance(seq_a: Sequence[Point], seq_b: Sequence[Point]) -> float:
    """
    Raw DTW distance between two sequences of scalars or points (e.g. a
    trajectory of (x, y) wrist positions, or a plain angle-over-time
    series). Lower = more similar shape. Not normalized — use
    `normalized_dtw_distance` to compare pairs of different lengths on a
    common scale.
    """
    a = _as_points(seq_a)
    b = _as_points(seq_b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("dtw_distance requires non-empty sequences")

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def normalized_dtw_distance(seq_a: Sequence[Point], seq_b: Sequence[Point]) -> float:
    """
    DTW distance divided by the length of the warping path (approximated
    as n + m), so sequences of different lengths are comparable on
    roughly the same scale. Still an absolute distance in the sequences'
    own units (e.g. degrees) — pair with `similarity_score` for a bounded
    0-100 number to show a user.
    """
    a = _as_points(seq_a)
    b = _as_points(seq_b)
    raw = dtw_distance(seq_a, seq_b)
    return raw / (len(a) + len(b))


def similarity_score(seq_a: Sequence[Point], seq_b: Sequence[Point],
                      scale: float = 30.0) -> float:
    """
    A bounded 0-100 similarity score derived from the normalized DTW
    distance, for display. `scale` sets how quickly the score falls off —
    it should be in the same units as the sequences (e.g. degrees for an
    angle trajectory) and roughly represents "a difference of this size
    should meaningfully lower the score." The default (30) is a reasonable
    starting point for joint-angle trajectories in degrees; tune per
    exercise once you have real reps to calibrate against.
    """
    dist = normalized_dtw_distance(seq_a, seq_b)
    score = 100.0 * np.exp(-dist / scale)
    return float(max(0.0, min(100.0, score)))


def best_matching_reference(seq: Sequence[Point], references: List[Sequence[Point]]) -> int:
    """Returns the index of the reference trajectory most similar to seq
    (lowest normalized DTW distance). Useful once you have a small library
    of labeled reference reps (e.g. from calibration or a "good rep"
    captured earlier in the set) to compare a new rep against."""
    if not references:
        raise ValueError("best_matching_reference requires at least one reference")
    distances = [normalized_dtw_distance(seq, ref) for ref in references]
    return int(np.argmin(distances))
