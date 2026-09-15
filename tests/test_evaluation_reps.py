import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.reps import rep_counting_error  # noqa: E402


def test_perfect_counting_gives_zero_error():
    m = rep_counting_error([10, 8, 12], [10, 8, 12])
    assert m.absolute_error == 0
    assert m.signed_error == 0
    assert m.mean_absolute_error == 0.0


def test_known_errors_computed_correctly():
    # actual: [10, 8], detected: [9, 8] -> per-set errors [-1, 0]
    m = rep_counting_error([10, 8], [9, 8])
    assert m.total_actual == 18
    assert m.total_detected == 17
    assert m.per_set_errors == [-1, 0]
    assert m.absolute_error == 1
    assert m.signed_error == -1
    assert abs(m.mean_absolute_error - 0.5) < 1e-9


def test_overcounting_shows_positive_signed_error():
    m = rep_counting_error([10, 10], [12, 11])
    assert m.signed_error == 3
    assert m.absolute_error == 3


def test_mixed_over_and_undercounting():
    m = rep_counting_error([10, 10], [12, 8])
    assert m.per_set_errors == [2, -2]
    assert m.absolute_error == 4
    assert m.signed_error == 0  # errors cancel out in aggregate, which is exactly why we also report absolute_error


def test_mismatched_lengths_raise():
    try:
        rep_counting_error([10, 10], [10])
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
