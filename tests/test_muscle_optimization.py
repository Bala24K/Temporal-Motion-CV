import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomechanics.muscle_optimization import solve_muscle_forces, MuscleCandidate  # noqa: E402


def test_single_muscle_exactly_meets_moment():
    m = MuscleCandidate("biceps", max_isometric_force_n=600.0, moment_arm_m=0.02)
    coeff = 600.0 * 0.02  # 12 N*m at 100% activation
    required = 6.0  # exactly half its capacity
    sol = solve_muscle_forces(required, [m])
    assert sol.feasible
    assert abs(sol.activations["biceps"] - 0.5) < 1e-6
    assert abs(sol.achieved_moment_nm - required) < 1e-6
    assert sol.moment_deficit_nm < 1e-6


def test_two_identical_muscles_split_the_load_equally():
    """Minimizing sum-of-squared activation with two IDENTICAL muscles
    (same force*moment_arm coefficient) should split the required moment
    equally — this is the textbook redundancy-resolution property: sharing
    is cheaper than loading one muscle alone (a^2 + a^2 < (2a)^2)."""
    m1 = MuscleCandidate("m1", max_isometric_force_n=500.0, moment_arm_m=0.02)
    m2 = MuscleCandidate("m2", max_isometric_force_n=500.0, moment_arm_m=0.02)
    sol = solve_muscle_forces(5.0, [m1, m2])
    assert sol.feasible
    assert abs(sol.activations["m1"] - sol.activations["m2"]) < 1e-4
    assert abs(sol.achieved_moment_nm - 5.0) < 1e-4


def test_stronger_muscle_takes_proportionally_more_activation():
    """For the sum-of-squares objective, at the unconstrained optimum
    activation_i is proportional to that muscle's force*moment_arm
    coefficient (from the Lagrange condition 2*a_i = lambda*c_i) — a
    muscle with double the coefficient should end up with roughly double
    the activation, not the same or less."""
    strong = MuscleCandidate("strong", max_isometric_force_n=1000.0, moment_arm_m=0.02)  # coeff=20
    weak = MuscleCandidate("weak", max_isometric_force_n=500.0, moment_arm_m=0.02)        # coeff=10
    sol = solve_muscle_forces(6.0, [strong, weak])  # well within both bounds
    assert sol.feasible
    ratio = sol.activations["strong"] / sol.activations["weak"]
    assert abs(ratio - 2.0) < 0.05, f"expected ~2x activation ratio, got {ratio}"


def test_moment_constraint_is_satisfied_for_a_three_muscle_case():
    """A biceps/brachialis/brachioradialis-like setup — verifies the
    optimizer actually satisfies the equality constraint, not just that
    it runs."""
    candidates = [
        MuscleCandidate("biceps_brachii", 620.0, 0.021),
        MuscleCandidate("brachialis", 990.0, 0.018),
        MuscleCandidate("brachioradialis", 260.0, 0.045),
    ]
    required = 8.5
    sol = solve_muscle_forces(required, candidates)
    assert sol.feasible
    assert abs(sol.achieved_moment_nm - required) < 1e-4
    for name, a in sol.activations.items():
        assert 0.0 <= a <= 1.0, f"{name} activation out of bounds: {a}"


def test_required_moment_exceeding_capacity_is_reported_infeasible():
    m = MuscleCandidate("weak_muscle", max_isometric_force_n=50.0, moment_arm_m=0.01)  # max moment = 0.5 N*m
    sol = solve_muscle_forces(required_moment_nm=10.0, candidates=[m])
    assert sol.feasible is False
    assert sol.moment_deficit_nm > 0
    assert sol.activations["weak_muscle"] == 1.0  # saturated, best effort


def test_zero_required_moment_gives_zero_activation():
    candidates = [
        MuscleCandidate("biceps_brachii", 620.0, 0.021),
        MuscleCandidate("brachialis", 990.0, 0.018),
    ]
    sol = solve_muscle_forces(0.0, candidates)
    assert sol.feasible
    assert all(a < 1e-6 for a in sol.activations.values())


def test_negative_required_moment_is_infeasible_for_flexor_only_candidates():
    m = MuscleCandidate("biceps", 600.0, 0.02)
    sol = solve_muscle_forces(-5.0, [m])
    assert sol.feasible is False


def test_empty_candidates_raises():
    try:
        solve_muscle_forces(5.0, [])
        assert False, "expected ValueError for empty candidate list"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
