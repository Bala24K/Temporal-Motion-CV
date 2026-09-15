"""
Muscle activation estimation pipeline (V3 sections 1, 6-10 tied together).

This is the "deterministic biomechanical pipeline" the doc insists on
building BEFORE any ML (section 10) — the stand-in for what a real system
would hand off to OpenSim. Nothing here is a neural network; every step is
inspectable: kinematics -> required joint moment -> muscle-redundancy
optimization -> confidence-scored, categorically-labeled output.

Per section 1's three-way distinction, every result separates:
  - involvement (categorical: HIGH/MODERATE/LOW/MINIMAL)
  - activation (continuous 0-1 "estimated normalized activation")
  - confidence (0-100, with caveats explaining what limits it)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from anatomy.muscles import ELBOW_FLEXORS, elbow_flexor_names
from biomechanics.kinematics import compute_kinematics
from biomechanics.anthropometrics import (
    estimate_forearm_segment, DEFAULT_BODY_MASS_KG, DEFAULT_BODY_HEIGHT_M,
)
from biomechanics.inverse_dynamics import required_elbow_moment
from biomechanics.moment_arms import moment_arm_at_angle
from biomechanics.muscle_optimization import solve_muscle_forces, MuscleCandidate
from analysis.confidence import compute_confidence, involvement_label, ConfidenceResult


@dataclass
class MuscleActivationFrame:
    time: float
    elbow_angle_deg: float
    required_moment_nm: float
    activations: Dict[str, float]          # muscle_key -> 0-1
    involvement: Dict[str, str]            # muscle_key -> HIGH/MODERATE/LOW/MINIMAL
    confidence: ConfidenceResult
    feasible: bool


def estimate_activation_sequence(
    times: Sequence[float],
    elbow_angles_deg: Sequence[float],
    joint_visibility: Sequence[float],
    body_mass_kg: Optional[float] = None,
    body_height_m: Optional[float] = None,
    external_load_kg: float = 0.0,
    include_inertial_term: bool = True,
) -> List[MuscleActivationFrame]:
    """
    times/elbow_angles_deg: a rep's (or a whole set's) angle-over-time
        signal, same convention used throughout this codebase.
    joint_visibility: per-frame mean visibility of the shoulder/elbow/wrist
        landmarks that angle came from (feeds the confidence score, not the
        physics — a low-visibility angle reading is still used, just
        flagged as less trustworthy).
    body_mass_kg/body_height_m: the person's own values if known; None
        falls back to population-average defaults AND is reflected as a
        confidence penalty (see analysis/confidence.py).
    external_load_kg: weight held in hand (e.g. a dumbbell).
    """
    used_real_anthro = body_mass_kg is not None and body_height_m is not None
    mass = body_mass_kg or DEFAULT_BODY_MASS_KG
    height = body_height_m or DEFAULT_BODY_HEIGHT_M
    segment = estimate_forearm_segment(mass, height, external_load_kg=external_load_kg)

    kin = compute_kinematics(times, elbow_angles_deg)
    flexor_names = elbow_flexor_names()

    frames: List[MuscleActivationFrame] = []
    for i in range(len(times)):
        angle = kin.angle_deg[i]
        alpha = kin.angular_acceleration_deg_s2[i] if include_inertial_term else None

        moment_result = required_elbow_moment(angle, segment, angular_acceleration_deg_s2=alpha)
        required_moment = max(0.0, moment_result.net_flexion_moment_nm)

        candidates = [
            MuscleCandidate(
                name=name,
                max_isometric_force_n=ELBOW_FLEXORS[name].max_isometric_force_n,
                moment_arm_m=moment_arm_at_angle(name, angle),
            )
            for name in flexor_names
        ]
        solution = solve_muscle_forces(required_moment, candidates)

        activations = {name: solution.activations[name] for name in flexor_names}
        involvement = {name: involvement_label(activations[name]) for name in flexor_names}

        vis = joint_visibility[i] if i < len(joint_visibility) else 1.0
        confidence = compute_confidence(
            mean_joint_visibility=vis,
            feasible=solution.feasible,
            moment_deficit_nm=solution.moment_deficit_nm,
            achieved_moment_nm=solution.achieved_moment_nm,
            used_subject_specific_anthropometrics=used_real_anthro,
        )

        frames.append(MuscleActivationFrame(
            time=times[i],
            elbow_angle_deg=angle,
            required_moment_nm=required_moment,
            activations=activations,
            involvement=involvement,
            confidence=confidence,
            feasible=solution.feasible,
        ))

    return frames


def summarize_rep_activation(frames: List[MuscleActivationFrame]) -> Dict[str, dict]:
    """
    Per-muscle peak and mean activation across a rep (or set), plus the
    involvement label derived from the PEAK (a muscle that briefly hits
    high activation at the bottom of a curl is fairly described as
    "highly involved" even if its time-average is much lower).
    """
    if not frames:
        return {}
    names = list(frames[0].activations.keys())
    summary = {}
    for name in names:
        values = [f.activations[name] for f in frames]
        peak = max(values)
        mean = sum(values) / len(values)
        summary[name] = {
            "peak_activation": peak,
            "mean_activation": mean,
            "involvement": involvement_label(peak),
        }
    return summary
