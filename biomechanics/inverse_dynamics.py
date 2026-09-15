"""
Simplified single-segment inverse dynamics for elbow flexion (V3 section 7
— a deliberately reduced stand-in for a real OpenSim inverse-dynamics
solve, which would use a full multi-segment model, measured/scaled
segment parameters, and 3D kinematics).

MODEL AND ASSUMPTIONS (read before trusting any output):
  - Single rigid segment (forearm + hand + any external load, lumped)
    rotating about a frictionless elbow hinge.
  - The upper arm is assumed held vertically at the side (standard
    standing curl form). This is what lets a single joint angle imply the
    forearm's orientation relative to gravity — a real biomechanics model
    would track the shoulder angle too and not need this assumption. It
    means this specifically does NOT apply to preacher curls, incline
    curls, or anything where the upper arm isn't roughly vertical.
  - elbow_angle_deg uses the SAME convention as the rest of this codebase
    (angle_3pt(shoulder, elbow, wrist)): 180 degrees = fully extended,
    small angles = fully flexed.

Sign convention: a positive returned moment is a FLEXION moment — the net
moment the elbow flexors must produce (against gravity, load, and/or to
accelerate the segment) to sustain the observed motion. This is exact for
the static case (validated in tests against a known equilibrium
calculation) and directionally correct for the dynamic case; the inertial
term's precise sign depends on which direction you define theta
increasing, which is why the static case — not the dynamic one — is what
this module's correctness is actually pinned to.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from biomechanics.anthropometrics import ForearmSegmentParameters, G


@dataclass
class JointMomentResult:
    gravity_moment_nm: float
    inertial_moment_nm: float
    net_flexion_moment_nm: float


def required_elbow_moment(elbow_angle_deg: float, segment: ForearmSegmentParameters,
                           angular_acceleration_deg_s2: Optional[float] = None) -> JointMomentResult:
    """
    Net flexion moment (N*m) the elbow flexors must produce at this instant.

    gravity_moment: horizontal distance of each mass from the elbow,
    under the vertical-upper-arm assumption, is proportional to
    sin(elbow_angle) — zero at full extension (forearm hangs straight
    down, no horizontal offset), maximal at 90 degrees (forearm
    horizontal). See module docstring for the geometric assumption this
    relies on.

    inertial_moment: I * alpha, using the segment's moment of inertia
    about the elbow. Omitted (0) if angular_acceleration_deg_s2 is None —
    i.e. this reduces to a pure static/quasi-static calculation, which is
    the mode this module's correctness is validated against.
    """
    theta = np.radians(elbow_angle_deg)
    horizontal_fraction = np.sin(theta)

    gravity_moment = (
        segment.mass_kg * segment.com_distance_m
        + segment.hand_load_mass_kg * segment.length_m
    ) * G * horizontal_fraction

    inertial_moment = 0.0
    if angular_acceleration_deg_s2 is not None:
        alpha_rad = np.radians(angular_acceleration_deg_s2)
        # Note the sign: elbow_angle DECREASING corresponds to flexion in
        # this codebase's convention, so a flexion-direction acceleration
        # is a NEGATIVE angular acceleration in raw theta terms, and
        # requires a POSITIVE flexion moment to produce it — hence the
        # negative sign here.
        inertial_moment = -segment.moment_of_inertia_kg_m2 * alpha_rad

    return JointMomentResult(
        gravity_moment_nm=float(gravity_moment),
        inertial_moment_nm=float(inertial_moment),
        net_flexion_moment_nm=float(gravity_moment + inertial_moment),
    )
