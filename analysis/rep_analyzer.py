"""
Rep-level analysis (V2 section 9).

Takes the raw per-rep signal (angle-over-time, optionally for both sides)
and produces a richer per-rep record than V1's bare RepResult — ROM/tempo
plus symmetry and trajectory-shape similarity against a reference. This
is deliberately a pure function of "measurements in, metrics out" (V2
section 8's "separate measurement from judgement" principle) — it does
NOT decide what counts as good or bad; that's SetAnalyzer's/baseline's job.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from biomechanics.symmetry import align_and_compare, SymmetryResult
from biomechanics.trajectory import similarity_score


@dataclass
class RepMetrics:
    rep_number: int
    rom: float
    tempo: float
    concentric_duration: float
    eccentric_duration: float
    symmetry: Optional[SymmetryResult] = None
    trajectory_similarity: Optional[float] = None  # 0-100 vs. a reference rep


def analyze_rep(rep_number: int, times: Sequence[float], angles: Sequence[float],
                 concentric_duration: float, eccentric_duration: float,
                 other_side_times: Optional[Sequence[float]] = None,
                 other_side_angles: Optional[Sequence[float]] = None,
                 reference_trajectory: Optional[Sequence[float]] = None) -> RepMetrics:
    """
    times/angles: the primary-side angle signal across this rep (matches
    what RepStateMachine tracked for min/max — see exercises/base.py).
    other_side_*: optional, for exercises where both limbs are tracked —
    enables symmetry scoring.
    reference_trajectory: an optional angle sequence from an earlier
    "good" rep (e.g. rep 1, or a calibration rep) to compare shape against
    via DTW.
    """
    rom = max(angles) - min(angles) if angles else 0.0
    tempo = concentric_duration + eccentric_duration

    symmetry = None
    if other_side_times is not None and other_side_angles is not None:
        symmetry = align_and_compare(list(times), list(angles),
                                      list(other_side_times), list(other_side_angles))

    traj_sim = None
    if reference_trajectory is not None and len(reference_trajectory) > 0 and len(angles) > 0:
        traj_sim = similarity_score(list(angles), list(reference_trajectory))

    return RepMetrics(
        rep_number=rep_number,
        rom=rom,
        tempo=tempo,
        concentric_duration=concentric_duration,
        eccentric_duration=eccentric_duration,
        symmetry=symmetry,
        trajectory_similarity=traj_sim,
    )
