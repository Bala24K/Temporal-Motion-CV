"""
Anthropometric segment parameters (supports V3 section 7, inverse dynamics).

Estimates forearm segment mass, length, center-of-mass location, and
moment of inertia from a person's height and body mass, using the kind of
regression fractions widely reproduced in biomechanics textbooks (e.g.
Winter, "Biomechanics and Motor Control of Human Movement").

HONESTY NOTE: these are generic population-average regression coefficients,
not measurements of any specific person. Real segment parameters vary by
individual (limb proportions, body composition) well beyond what a
height/mass regression captures. Treat this as "plausible ballpark," not
"measured" — same caveat as anatomy/muscles.py.
"""
from dataclasses import dataclass

# Fractions below are typical textbook approximations (proportions of total
# body mass / height), not a precise reproduction of any single source table.
FOREARM_MASS_FRACTION = 0.016     # forearm mass as a fraction of total body mass
HAND_MASS_FRACTION = 0.006        # hand mass as a fraction of total body mass
FOREARM_LENGTH_FRACTION = 0.146   # forearm (elbow-to-wrist) length as a fraction of height
FOREARM_COM_FRACTION = 0.43       # forearm center-of-mass location, fraction of forearm length from the elbow
FOREARM_RADIUS_OF_GYRATION_PROXIMAL = 0.53  # radius of gyration about the elbow, fraction of forearm length

G = 9.81  # m/s^2


@dataclass
class ForearmSegmentParameters:
    mass_kg: float                  # forearm mass only
    hand_load_mass_kg: float        # hand mass + any external load (dumbbell), lumped at the wrist
    length_m: float                 # elbow-to-wrist length
    com_distance_m: float           # elbow-to-forearm-COM distance
    moment_of_inertia_kg_m2: float  # about the elbow (pivot), forearm rod + point mass at wrist


def estimate_forearm_segment(body_mass_kg: float, body_height_m: float,
                              external_load_kg: float = 0.0) -> ForearmSegmentParameters:
    """
    body_mass_kg, body_height_m: the person's own mass/height. If unknown,
    reasonable adult defaults are provided by DEFAULT_BODY_MASS_KG /
    DEFAULT_BODY_HEIGHT_M below — using those instead of a real value
    should lower downstream confidence (see analysis/confidence.py).
    external_load_kg: weight held in the hand (e.g. a dumbbell), modeled
    as a point mass at the wrist, same simplification as the hand itself.
    """
    if body_mass_kg <= 0 or body_height_m <= 0:
        raise ValueError("body_mass_kg and body_height_m must be positive")

    forearm_mass = FOREARM_MASS_FRACTION * body_mass_kg
    hand_mass = HAND_MASS_FRACTION * body_mass_kg
    length = FOREARM_LENGTH_FRACTION * body_height_m
    com_distance = FOREARM_COM_FRACTION * length

    # Forearm modeled as a rod: I_proximal = m * (k * L)^2 (radius of
    # gyration about the elbow). Hand + external load modeled as a point
    # mass concentrated at the wrist (distance = length) — a simplification;
    # see module docstring.
    i_forearm = forearm_mass * (FOREARM_RADIUS_OF_GYRATION_PROXIMAL * length) ** 2
    i_hand_load = (hand_mass + external_load_kg) * length ** 2

    return ForearmSegmentParameters(
        mass_kg=forearm_mass,
        hand_load_mass_kg=hand_mass + external_load_kg,
        length_m=length,
        com_distance_m=com_distance,
        moment_of_inertia_kg_m2=i_forearm + i_hand_load,
    )


DEFAULT_BODY_MASS_KG = 75.0
DEFAULT_BODY_HEIGHT_M = 1.75
