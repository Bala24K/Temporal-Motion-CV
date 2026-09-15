"""
Anatomical knowledge base (V3 section 2): structured domain knowledge, not
learned. Deliberately scoped to the muscles relevant to elbow flexion
(bicep curl) rather than a whole-body model.

IMPORTANT — honesty about the numbers:
max_isometric_force_n and the moment-arm model below are ILLUSTRATIVE,
order-of-magnitude values in the range reported by generic upper-limb
musculoskeletal models in the literature (e.g. the Holzbaur et al. 2005
model family, which is also what OpenSim's included arm models are based
on) — they are NOT a verified reproduction of any specific published
table, and they are NOT subject-specific (a real analysis would use
subject MRI/ultrasound-derived PCSA, or at minimum OpenSim's scaled
generic model). Treat every downstream number in analysis/muscle_estimation.py
as "illustrative estimate," not measurement — see the confidence system
in analysis/confidence.py, which is deliberately capped for this reason.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MuscleParameters:
    name: str
    origin: str
    insertion: str
    joints_crossed: List[str]
    primary_actions: List[str]
    secondary_actions: List[str] = field(default_factory=list)
    # Illustrative order-of-magnitude max isometric force. See module
    # docstring — not a validated subject-specific or literature-table value.
    max_isometric_force_n: float = 0.0
    # Whether this muscle is a prime mover (agonist) for elbow flexion, vs.
    # a secondary contributor. Used to decide which muscles even enter the
    # optimization for a given movement.
    is_elbow_flexor: bool = False


ELBOW_FLEXORS = {
    "biceps_brachii": MuscleParameters(
        name="Biceps brachii",
        origin="Supraglenoid tubercle of scapula (long head); coracoid process (short head)",
        insertion="Radial tuberosity; bicipital aponeurosis",
        joints_crossed=["shoulder", "elbow", "radioulnar"],
        primary_actions=["elbow flexion", "forearm supination"],
        secondary_actions=["shoulder flexion"],
        max_isometric_force_n=620.0,  # illustrative — see module docstring
        is_elbow_flexor=True,
    ),
    "brachialis": MuscleParameters(
        name="Brachialis",
        origin="Distal anterior humerus",
        insertion="Ulnar tuberosity, coronoid process",
        joints_crossed=["elbow"],
        primary_actions=["elbow flexion"],
        secondary_actions=[],
        # Brachialis has the largest physiological cross-sectional area of
        # the elbow flexors in most cited models, hence the largest force
        # here — that qualitative ordering is well supported even though
        # the exact number is illustrative.
        max_isometric_force_n=990.0,
        is_elbow_flexor=True,
    ),
    "brachioradialis": MuscleParameters(
        name="Brachioradialis",
        origin="Proximal lateral supracondylar ridge of humerus",
        insertion="Styloid process of radius",
        joints_crossed=["elbow"],
        primary_actions=["elbow flexion"],
        secondary_actions=["forearm pronation/supination toward neutral"],
        max_isometric_force_n=260.0,
        is_elbow_flexor=True,
    ),
    "triceps_brachii": MuscleParameters(
        name="Triceps brachii",
        origin="Infraglenoid tubercle of scapula (long head); posterior humerus (lateral/medial heads)",
        insertion="Olecranon process of ulna",
        joints_crossed=["shoulder", "elbow"],
        primary_actions=["elbow extension"],
        secondary_actions=["shoulder extension (long head)"],
        max_isometric_force_n=850.0,
        is_elbow_flexor=False,  # antagonist, not part of the flexion optimization
    ),
}


def elbow_flexor_names() -> List[str]:
    return [k for k, v in ELBOW_FLEXORS.items() if v.is_elbow_flexor]
