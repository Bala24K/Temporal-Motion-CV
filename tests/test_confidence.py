import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.confidence import compute_confidence, involvement_label, SYSTEMATIC_CAP  # noqa: E402


def test_perfect_inputs_still_capped_below_full_confidence():
    """Even flawless visibility, a fully feasible solve, and known
    anthropometrics must never report near-100% confidence — the
    systematic scaling reflects the generic-parameters/monocular-camera
    limitation that no amount of clean tracking can fix. With every input
    at its best, the score should land exactly at the systematic cap."""
    result = compute_confidence(
        mean_joint_visibility=1.0, feasible=True, moment_deficit_nm=0.0,
        achieved_moment_nm=5.0, used_subject_specific_anthropometrics=True,
    )
    assert abs(result.score - SYSTEMATIC_CAP) < 1e-6


def test_poor_visibility_lowers_confidence():
    good = compute_confidence(0.95, True, 0.0, 5.0, True)
    poor = compute_confidence(0.4, True, 0.0, 5.0, True)
    assert poor.score < good.score


def test_infeasible_solve_lowers_confidence():
    feasible = compute_confidence(0.95, True, 0.0, 5.0, True)
    infeasible = compute_confidence(0.95, False, 4.0, 5.0, True)
    assert infeasible.score < feasible.score


def test_missing_anthropometrics_lowers_confidence():
    known = compute_confidence(0.95, True, 0.0, 5.0, used_subject_specific_anthropometrics=True)
    unknown = compute_confidence(0.95, True, 0.0, 5.0, used_subject_specific_anthropometrics=False)
    assert unknown.score < known.score


def test_generic_parameters_caveat_always_present():
    result = compute_confidence(1.0, True, 0.0, 5.0, True)
    assert any("generic" in c.lower() for c in result.caveats)


def test_score_never_negative_or_over_100():
    result = compute_confidence(0.0, False, 100.0, 0.001, False)
    assert 0.0 <= result.score <= 100.0


def test_involvement_labels_are_monotonic_and_cover_the_range():
    assert involvement_label(0.9) == "HIGH"
    assert involvement_label(0.6) == "HIGH"
    assert involvement_label(0.45) == "MODERATE"
    assert involvement_label(0.3) == "MODERATE"
    assert involvement_label(0.1) == "LOW"
    assert involvement_label(0.0) == "MINIMAL"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
