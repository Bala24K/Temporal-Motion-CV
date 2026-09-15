import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from biomechanics.trajectory import (  # noqa: E402
    dtw_distance, normalized_dtw_distance, similarity_score, best_matching_reference
)


def ramp(a, b, steps):
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def curl_shape(steps, low=40, high=160):
    half = steps // 2
    return ramp(high, low, half) + ramp(low, high, steps - half)


def test_identical_sequences_have_zero_distance():
    seq = curl_shape(30)
    assert dtw_distance(seq, seq) == 0.0


def test_same_shape_different_speed_scores_highly_similar():
    """The whole point of DTW: a slow rep and a fast rep of the SAME
    movement should compare as very similar, unlike a naive
    sample-by-sample comparison which would be thrown off by the length
    mismatch."""
    fast = curl_shape(15)   # same movement, fewer samples (faster / lower fps)
    slow = curl_shape(45)   # same movement, more samples (slower / higher fps)
    score = similarity_score(fast, slow)
    assert score > 85, f"expected high similarity for same shape at different speeds, got {score}"


def test_different_shape_scores_lower_than_same_shape():
    reference = curl_shape(30)
    same_shape_diff_speed = curl_shape(45)
    # a rep that goes only halfway down (partial ROM) — genuinely different shape
    partial = ramp(160, 100, 15) + ramp(100, 160, 15)

    score_same = similarity_score(reference, same_shape_diff_speed)
    score_partial = similarity_score(reference, partial)
    assert score_same > score_partial, \
        f"same-shape score ({score_same}) should exceed different-shape score ({score_partial})"


def test_normalized_distance_scales_reasonably_with_length():
    """Raw dtw_distance grows with sequence length even for identical
    shapes (more points along a longer diagonal path); the normalized
    version should not blow up the same way."""
    short_ref = curl_shape(20)
    long_ref = curl_shape(200)
    # compare each against a same-shape sequence of matching length
    short_cmp = curl_shape(22)
    long_cmp = curl_shape(210)

    norm_short = normalized_dtw_distance(short_ref, short_cmp)
    norm_long = normalized_dtw_distance(long_ref, long_cmp)
    # both should be small (same shape); neither should be wildly larger
    # than the other just because the sequences are longer
    assert norm_short < 5.0
    assert norm_long < 5.0


def test_2d_trajectory_points_supported():
    """DTW should also work on (x, y) wrist-position trajectories, not
    just scalar angle sequences."""
    traj_a = [(0.0, i * 0.1) for i in range(10)]
    traj_b = [(0.0, i * 0.1) for i in range(10)]  # identical path
    traj_c = [(i * 0.1, 0.0) for i in range(10)]  # perpendicular path
    assert dtw_distance(traj_a, traj_b) == 0.0
    assert dtw_distance(traj_a, traj_c) > 0.0


def test_best_matching_reference_picks_the_closer_one():
    good_rep = curl_shape(30)
    references = [
        ramp(160, 100, 15) + ramp(100, 160, 15),  # a partial-ROM reference
        curl_shape(28),                            # a full-ROM reference, matching shape
    ]
    idx = best_matching_reference(good_rep, references)
    assert idx == 1


def test_empty_sequence_raises():
    try:
        dtw_distance([], [1, 2, 3])
        assert False, "expected a ValueError for an empty sequence"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
