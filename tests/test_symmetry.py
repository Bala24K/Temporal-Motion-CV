import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from biomechanics.symmetry import (  # noqa: E402
    rom_symmetry, mean_absolute_difference, align_and_compare
)


def ramp(a, b, steps):
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def curl_sequence(steps=40, low=40, high=160):
    return ramp(high, low, steps // 2) + ramp(low, high, steps - steps // 2)


def test_rom_symmetry_zero_for_identical_rom():
    assert rom_symmetry(120.0, 120.0) == 0.0


def test_rom_symmetry_reports_the_gap():
    assert abs(rom_symmetry(120.0, 100.0) - 20.0) < 1e-9


def test_mean_absolute_difference_zero_for_identical_sequences():
    seq = curl_sequence()
    assert mean_absolute_difference(seq, seq) == 0.0


def test_mean_absolute_difference_none_on_length_mismatch():
    assert mean_absolute_difference([1, 2, 3], [1, 2]) is None


def test_perfectly_symmetric_movement_scores_high():
    angles = curl_sequence(steps=40)
    times = list(np.linspace(0, 2.0, len(angles)))
    result = align_and_compare(times, angles, times, angles)
    assert result is not None
    assert result.symmetry_score > 99.0
    assert result.rom_difference < 1e-6
    assert abs(result.peak_timing_offset) < 0.01


def test_asymmetric_rom_lowers_the_score():
    """Left arm does a full curl; right arm does a noticeably smaller one."""
    left = curl_sequence(steps=40, low=40, high=160)     # ROM 120
    right = curl_sequence(steps=40, low=90, high=160)    # ROM 70
    times = list(np.linspace(0, 2.0, 40))
    result = align_and_compare(times, left, times, right)
    assert result is not None
    assert result.rom_difference > 40
    assert result.symmetry_score < 70, f"expected a clearly lowered score, got {result.symmetry_score}"


def test_timing_offset_detects_a_lagging_side():
    """Right arm reaches the bottom of the curl half a second after the left."""
    angles = curl_sequence(steps=40)
    left_times = list(np.linspace(0, 2.0, 40))
    right_times = [t + 0.5 for t in left_times]
    result = align_and_compare(left_times, angles, right_times, angles)
    assert result is not None
    # a lagging right side should show up as a nonzero timing offset
    assert abs(result.peak_timing_offset) > 0.1


def test_too_short_series_returns_none():
    assert align_and_compare([0.0], [90.0], [0.0], [90.0]) is None


def test_non_overlapping_windows_return_none():
    left_times = [0.0, 1.0]
    right_times = [5.0, 6.0]
    assert align_and_compare(left_times, [90, 100], right_times, [90, 100]) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
