"""
Confidence-aware output (V3 section 20). This is treated as mandatory
infrastructure, not a nice-to-have — the doc is explicit that a bare
"Biceps 82%" is irresponsible without it. Every muscle-activation estimate
this codebase produces MUST be paired with a ConfidenceResult; nothing in
analysis/muscle_estimation.py returns activation numbers without one.

The score is a deliberately blunt, explainable combination of factors —
not itself a learned or rigorously calibrated uncertainty estimate. Given
everything this V3 pass is built on (generic muscle parameters, a
single-segment model, a monocular camera, a vertical-upper-arm
assumption), no number above this module's SYSTEMATIC_CAP is honest to
report — see the cap applied in compute_confidence.
"""
from dataclasses import dataclass, field
from typing import List

# However clean the pose tracking and however well-satisfied the moment
# equilibrium constraint, this whole pipeline rests on generic (not
# subject-specific) muscle and segment parameters and a single-camera,
# single-plane assumption. Confidence is scaled down by this factor so a
# "perfect" input can never be reported as anything resembling a
# validated measurement, while still preserving relative differences
# between e.g. good and poor visibility below that ceiling.
SYSTEMATIC_CAP = 75.0


@dataclass
class ConfidenceResult:
    score: float  # 0-100
    caveats: List[str] = field(default_factory=list)


def compute_confidence(mean_joint_visibility: float,
                        feasible: bool,
                        moment_deficit_nm: float,
                        achieved_moment_nm: float,
                        used_subject_specific_anthropometrics: bool = False) -> ConfidenceResult:
    """
    mean_joint_visibility: average pose-landmark visibility (0-1) for the
        joints this estimate depends on (shoulder/elbow/wrist).
    feasible / moment_deficit_nm / achieved_moment_nm: from
        MuscleForceSolution — how well the required moment was actually
        accounted for by the modeled muscles.
    used_subject_specific_anthropometrics: whether real height/mass were
        supplied, vs. population-average defaults.
    """
    caveats: List[str] = []
    score = 100.0

    v = max(0.0, min(1.0, mean_joint_visibility))
    score *= v
    if v < 0.8:
        caveats.append("Reduced confidence: shoulder/elbow/wrist tracking was only partially reliable.")

    if not used_subject_specific_anthropometrics:
        score *= 0.9
        caveats.append("Body height/mass not provided — using population-average segment estimates.")

    if not feasible or moment_deficit_nm > 1e-6:
        denom = max(abs(achieved_moment_nm), 1e-6)
        rel_deficit = moment_deficit_nm / denom
        penalty = max(0.0, 1.0 - min(1.0, rel_deficit))
        score *= penalty
        caveats.append("The estimated required joint moment was not fully explained by the modeled muscles.")

    score = score * (SYSTEMATIC_CAP / 100.0)
    caveats.append(
        "Estimated from a single camera view using generic (non-subject-specific) "
        "muscle and segment parameters — treat as illustrative, not a validated measurement."
    )

    score = max(0.0, min(100.0, score))
    return ConfidenceResult(score=score, caveats=caveats)


def involvement_label(activation: float) -> str:
    """Maps a continuous activation estimate to the HIGH/MODERATE/LOW
    categorical label the doc's UI mockups use — coarser categories are
    more honest here than a false-precision percentage for casual display."""
    if activation >= 0.6:
        return "HIGH"
    if activation >= 0.3:
        return "MODERATE"
    if activation > 0.02:
        return "LOW"
    return "MINIMAL"
