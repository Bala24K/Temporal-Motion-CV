"""
Left/right symmetry analysis (V2 section 13).

Compares two time-aligned angle sequences (e.g. left elbow angle vs. right
elbow angle across a rep) and produces observational metrics — never a
diagnosis. Per the doc's own framing: "Left/right movement asymmetry
increased" is defensible; a claim about *why* is not.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class SymmetryResult:
    rom_difference: float          # |left_ROM - right_ROM|, in the angle's units
    peak_timing_offset: float      # seconds between each side's peak contraction
    mean_absolute_difference: float  # average |left(t) - right(t)| across the aligned window
    symmetry_score: float          # 0-100, 100 = perfectly symmetric, heuristic scale


def rom_symmetry(left_rom: float, right_rom: float) -> float:
    """Absolute difference in range of motion between sides."""
    return abs(left_rom - right_rom)


def peak_timing_offset(left_times: List[float], left_angles: List[float],
                        right_times: List[float], right_angles: List[float],
                        find_min: bool = True) -> Optional[float]:
    """
    Time gap between when each side reaches its extreme (e.g. most
    contracted point). find_min=True looks for the minimum angle (e.g.
    bottom of a curl); False looks for the maximum.
    Returns None if either series is empty.
    """
    if not left_times or not right_times:
        return None
    l_idx = int(np.argmin(left_angles) if find_min else np.argmax(left_angles))
    r_idx = int(np.argmin(right_angles) if find_min else np.argmax(right_angles))
    return float(right_times[r_idx] - left_times[l_idx])


def mean_absolute_difference(left_angles: List[float], right_angles: List[float]) -> Optional[float]:
    """
    Average |left(t) - right(t)| over a common, time-aligned window. The
    two lists must already be sampled at matching timestamps (same length,
    same cadence) — this function does not do time alignment itself; see
    `align_and_compare` for the version that resamples first.
    """
    if not left_angles or not right_angles or len(left_angles) != len(right_angles):
        return None
    diffs = np.abs(np.array(left_angles) - np.array(right_angles))
    return float(np.mean(diffs))


def align_and_compare(left_times, left_angles, right_times, right_angles,
                       n_samples: int = 50) -> Optional[SymmetryResult]:
    """
    Resamples both sides onto a common time grid (n_samples points spanning
    the overlap of the two time windows) via linear interpolation, then
    computes symmetry metrics. This is the entry point most callers should
    use — it tolerates the two sides being sampled at slightly different
    rates or having a few missing frames, as real pose data will.
    """
    if len(left_times) < 2 or len(right_times) < 2:
        return None

    t_start = max(min(left_times), min(right_times))
    t_end = min(max(left_times), max(right_times))
    if t_end <= t_start:
        return None

    grid = np.linspace(t_start, t_end, n_samples)
    l_interp = np.interp(grid, left_times, left_angles)
    r_interp = np.interp(grid, right_times, right_angles)

    left_rom = float(np.max(l_interp) - np.min(l_interp))
    right_rom = float(np.max(r_interp) - np.min(r_interp))
    rom_diff = rom_symmetry(left_rom, right_rom)

    timing_offset = peak_timing_offset(list(grid), list(l_interp), list(grid), list(r_interp))
    mad = mean_absolute_difference(list(l_interp), list(r_interp))

    # Heuristic 0-100 score: penalize ROM difference and mean angle
    # difference relative to the overall movement's own ROM, so the score
    # is scale-appropriate (a 5-degree difference matters more on a
    # 40-degree-ROM exercise than a 140-degree one).
    reference_rom = max(left_rom, right_rom, 1e-6)
    rom_penalty = min(1.0, rom_diff / reference_rom)
    mad_penalty = min(1.0, (mad or 0.0) / reference_rom)
    score = 100.0 * (1.0 - 0.5 * rom_penalty - 0.5 * mad_penalty)
    score = max(0.0, min(100.0, score))

    return SymmetryResult(
        rom_difference=rom_diff,
        peak_timing_offset=timing_offset if timing_offset is not None else 0.0,
        mean_absolute_difference=mad if mad is not None else 0.0,
        symmetry_score=score,
    )
