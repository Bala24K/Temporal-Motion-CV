"""
Angle-dependent moment arms for the elbow flexors (supports V3 sections 6/9).

Real moment-arm-vs-angle curves are muscle-specific, subject-specific, and
nonlinear (see An et al. 1981, Murray et al. 1995, Holzbaur et al. 2005 for
the real measured/modeled curves). This is a deliberately simple stand-in:
a bell shape peaking at 90 degrees elbow flexion, which IS a well-supported
qualitative feature of elbow flexor moment arms in the literature (peak
torque near 90 degrees flexion is explicitly reported for this muscle
group — see the model docstring in inverse_dynamics.py for the same
90-degree convention). The specific PEAK_MOMENT_ARMS_M values are
illustrative, in a qualitatively correct order (brachioradialis has the
longest lever arm of the three, consistent with its more distal
insertion at the radial styloid; brachialis the shortest, consistent with
its insertion close to the elbow axis on the ulna) but not a fit to any
specific published curve.
"""
import numpy as np

PEAK_MOMENT_ARMS_M = {
    "biceps_brachii": 0.021,
    "brachialis": 0.018,
    "brachioradialis": 0.045,
}


def moment_arm_at_angle(muscle_name: str, elbow_angle_deg: float) -> float:
    """
    elbow_angle_deg uses this codebase's convention (180 = fully extended).
    Returns the moment arm in meters, peaking at PEAK_MOMENT_ARMS_M[muscle]
    when elbow_angle_deg == 90.
    """
    if muscle_name not in PEAK_MOMENT_ARMS_M:
        raise KeyError(f"No moment-arm model for muscle '{muscle_name}'")
    peak = PEAK_MOMENT_ARMS_M[muscle_name]
    theta = np.radians(elbow_angle_deg)
    return float(peak * np.sin(theta))
