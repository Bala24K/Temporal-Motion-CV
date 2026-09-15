import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomechanics.moment_arms import moment_arm_at_angle, PEAK_MOMENT_ARMS_M  # noqa: E402


def test_peaks_at_ninety_degrees():
    for name, peak in PEAK_MOMENT_ARMS_M.items():
        assert abs(moment_arm_at_angle(name, 90.0) - peak) < 1e-9


def test_zero_at_full_extension():
    for name in PEAK_MOMENT_ARMS_M:
        assert abs(moment_arm_at_angle(name, 180.0)) < 1e-9


def test_symmetric_about_ninety_degrees():
    for name in PEAK_MOMENT_ARMS_M:
        a = moment_arm_at_angle(name, 60.0)
        b = moment_arm_at_angle(name, 120.0)
        assert abs(a - b) < 1e-9


def test_brachioradialis_has_the_longest_lever_arm():
    """Qualitatively well-established: brachioradialis's distal insertion
    at the radial styloid gives it the longest moment arm of the three
    prime elbow flexors."""
    br = moment_arm_at_angle("brachioradialis", 90.0)
    bic = moment_arm_at_angle("biceps_brachii", 90.0)
    bra = moment_arm_at_angle("brachialis", 90.0)
    assert br > bic > bra


def test_unknown_muscle_raises():
    try:
        moment_arm_at_angle("not_a_real_muscle", 90.0)
        assert False, "expected KeyError for an unmodeled muscle"
    except KeyError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
