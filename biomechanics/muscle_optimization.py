"""
Muscle redundancy / static optimization (V3 sections 8-9): more muscles
than degrees of freedom, so there's no unique way to produce a given joint
moment. This is the actual crux of "estimated muscle activation" — not a
lookup table.

Follows the standard static-optimization formulation (conceptually what
OpenSim's Static Optimization tool does, greatly simplified — no
force-length/force-velocity curves, no tendon compliance, no pennation):

    minimize   sum(activation_i^2)
    subject to sum(activation_i * max_force_i * moment_arm_i) == required_moment
               0 <= activation_i <= 1

Minimizing summed-squared activation is a common, defensible physiological
objective (roughly: don't work any one muscle harder than necessary) — see
the doc's section 9 for the range of objectives real biomechanics software
supports; this implementation uses the simplest one.

If the required moment exceeds what the available muscles can produce even
at 100% activation, the solve is infeasible — this is reported explicitly
(feasible=False) rather than silently returning saturated nonsense, because
a required-moment-exceeds-capacity result usually means the segment
parameters or the measured moment are off, and that's worth surfacing.
"""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


@dataclass
class MuscleCandidate:
    name: str
    max_isometric_force_n: float
    moment_arm_m: float  # at the CURRENT joint angle — moment arms are angle-dependent


@dataclass
class MuscleForceSolution:
    activations: Dict[str, float]      # 0-1 per muscle
    forces_n: Dict[str, float]         # activation * max_isometric_force
    feasible: bool
    moment_deficit_nm: float           # 0 if feasible; how much moment is unaccounted for otherwise
    required_moment_nm: float
    achieved_moment_nm: float


def solve_muscle_forces(required_moment_nm: float,
                         candidates: List[MuscleCandidate]) -> MuscleForceSolution:
    if not candidates:
        raise ValueError("solve_muscle_forces requires at least one candidate muscle")

    n = len(candidates)
    max_forces = np.array([c.max_isometric_force_n for c in candidates])
    moment_arms = np.array([c.moment_arm_m for c in candidates])
    coeffs = max_forces * moment_arms  # activation_i's contribution per unit activation

    max_possible_moment = float(np.sum(np.abs(coeffs)))  # if required_moment has the "wrong" sign this still bounds magnitude

    if required_moment_nm < 0:
        # This solver only handles producing a moment in the flexion
        # direction the candidates were given for (moment_arms already
        # encode direction). A negative required moment with all-positive
        # coefficients is infeasible by construction.
        feasible = False
    else:
        feasible = required_moment_nm <= max_possible_moment + 1e-9

    if not candidates or max_possible_moment < 1e-9:
        # no muscle can contribute anything
        activations = {c.name: 0.0 for c in candidates}
        forces = {c.name: 0.0 for c in candidates}
        return MuscleForceSolution(
            activations=activations, forces_n=forces, feasible=(required_moment_nm <= 1e-9),
            moment_deficit_nm=max(0.0, required_moment_nm),
            required_moment_nm=required_moment_nm, achieved_moment_nm=0.0,
        )

    if not feasible:
        # Saturate everything (best effort) and report the shortfall
        # explicitly rather than pretending a solution was found.
        a = np.ones(n)
        achieved = float(np.dot(coeffs, a))
        activations = {c.name: 1.0 for c in candidates}
        forces = {c.name: float(max_forces[i]) for i, c in enumerate(candidates)}
        return MuscleForceSolution(
            activations=activations, forces_n=forces, feasible=False,
            moment_deficit_nm=max(0.0, required_moment_nm - achieved),
            required_moment_nm=required_moment_nm, achieved_moment_nm=achieved,
        )

    # Equality-constrained QP: minimize sum(a_i^2) s.t. coeffs . a == required_moment, 0<=a<=1.
    # Solved numerically (SLSQP) rather than via closed-form Lagrange
    # multipliers so the box constraints [0, 1] are respected exactly even
    # when the unconstrained optimum would violate them.
    x0 = np.clip(required_moment_nm / max_possible_moment, 0, 1) * np.ones(n)

    constraint = LinearConstraint(coeffs.reshape(1, -1), lb=required_moment_nm, ub=required_moment_nm)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

    res = minimize(
        fun=lambda a: float(np.sum(a ** 2)),
        x0=x0,
        jac=lambda a: 2 * a,
        constraints=[constraint],
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-10},
    )

    a = np.clip(res.x, 0.0, 1.0)
    achieved = float(np.dot(coeffs, a))

    activations = {c.name: float(a[i]) for i, c in enumerate(candidates)}
    forces = {c.name: float(a[i] * max_forces[i]) for i, c in enumerate(candidates)}

    return MuscleForceSolution(
        activations=activations,
        forces_n=forces,
        feasible=True,
        moment_deficit_nm=max(0.0, abs(required_moment_nm - achieved)),
        required_moment_nm=required_moment_nm,
        achieved_moment_nm=achieved,
    )
