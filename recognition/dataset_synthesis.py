"""
Synthetic training data for exercise classification.

READ THIS BEFORE TRUSTING ANY ACCURACY NUMBER DOWNSTREAM OF THIS MODULE.

These examples are NOT recorded human movement. Each exercise's label is
generated directly from the SAME joint-angle definitions the rule-based
analyzers in exercises/ use — e.g. "this is a squat" means "knee and hip
angles both swept through a squat-like range while elbow and
shoulder-elevation angles stayed near neutral," by construction, not by
observation of an actual squat. That is a legitimate way to teach a
classifier the geometric DEFINITION of each exercise (the defining
difference between a curl and a squat really is which joints move, and
that's a fact I can state correctly without any recorded data).

It is NOT a legitimate way to learn:
  - what real pose-tracking noise/jitter looks like (this uses simple
    Gaussian noise, not MediaPipe's actual error characteristics)
  - real anatomical variation beyond the crude per-subject randomization
    below (limb proportions, unusual body types)
  - exercise variants this generator doesn't model (sumo squats, incline
    presses, seated raises, single-arm variations, etc.)
  - what happens with a partially-occluded camera view, unusual clothing,
    multiple people in frame, etc.

A classifier trained ONLY on this data has UNKNOWN real-world accuracy.
The accuracy number in classifier.py's evaluation measures exactly one
thing: how well the classifier recovers the geometric definitions encoded
below, on more synthetic examples generated the same way. See STATUS.md.

Each synthetic "subject" gets randomized baseline stance angles, achieved
ROM, and noise level, and examples are split BY SUBJECT (not by
individual example) between train and test in classifier.py — the same
"split by person, not by rep" principle the V1 doc itself calls for with
real data, so the classifier can't just memorize one subject's exact
noise pattern and call it generalization.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

EXERCISES = ["bicep_curl", "squat", "shoulder_press", "lateral_raise"]

N_FEATURES = 8  # must match recognition.features.FEATURE_NAMES


@dataclass
class SubjectProfile:
    standing_knee_angle: float
    standing_hip_angle: float
    standing_elbow_angle: float
    rest_shoulder_elevation: float
    noise_std: float


def _make_subject(rng: np.random.Generator) -> SubjectProfile:
    return SubjectProfile(
        standing_knee_angle=rng.uniform(165, 178),
        standing_hip_angle=rng.uniform(168, 179),
        standing_elbow_angle=rng.uniform(150, 172),
        rest_shoulder_elevation=rng.uniform(0, 12),
        noise_std=rng.uniform(1.5, 5.0),
    )


def _phase_value(rng: np.random.Generator, low: float, high: float, phase: float, noise_std: float) -> float:
    """phase in [0,1] interpolates from low to high, with noise — models
    a random point in a rep's cycle rather than only ever sampling the
    two extremes, so the dataset isn't trivially bimodal."""
    return low + (high - low) * phase + rng.normal(0, noise_std)


def generate_example(rng: np.random.Generator, exercise: str, subject: SubjectProfile) -> np.ndarray:
    """
    Returns an 8-feature vector in the same order as
    recognition.features.FEATURE_NAMES:
    [left_elbow, right_elbow, left_knee, right_knee,
     left_hip, right_hip, left_shoulder_elev, right_shoulder_elev]
    """
    phase = rng.uniform(0, 1)
    n = subject.noise_std

    if exercise == "bicep_curl":
        # Defining pattern: one elbow sweeps through a large range; knee,
        # hip, and shoulder-elevation stay near standing/rest — the upper
        # arm doesn't leave the side during a controlled curl.
        active_elbow = _phase_value(rng, subject.standing_elbow_angle, 45, phase, n)
        rest_elbow = subject.standing_elbow_angle + rng.normal(0, n)
        knee = subject.standing_knee_angle + rng.normal(0, n)
        hip = subject.standing_hip_angle + rng.normal(0, n)
        shoulder_elev = subject.rest_shoulder_elevation + rng.normal(0, n * 0.5)
        return np.array([active_elbow, rest_elbow, knee, knee, hip, hip, shoulder_elev, shoulder_elev])

    if exercise == "squat":
        # Defining pattern: BOTH knees and BOTH hips sweep together;
        # elbow and shoulder-elevation stay near neutral.
        knee = _phase_value(rng, subject.standing_knee_angle, 90, phase, n)
        hip = _phase_value(rng, subject.standing_hip_angle, 100, phase, n)
        elbow = subject.standing_elbow_angle + rng.normal(0, n)
        shoulder_elev = subject.rest_shoulder_elevation + rng.normal(0, n * 0.5)
        return np.array([elbow, elbow, knee, knee, hip, hip, shoulder_elev, shoulder_elev])

    if exercise == "shoulder_press":
        # Defining pattern: elbow AND shoulder-elevation both sweep large
        # ranges together (upper arm is already raised at the rack
        # position, unlike lateral raise's rest-at-the-side start), knee
        # and hip stay near standing.
        elbow = _phase_value(rng, 85, 172, phase, n)
        shoulder_elev = _phase_value(rng, 75, 172, phase, n)
        knee = subject.standing_knee_angle + rng.normal(0, n)
        hip = subject.standing_hip_angle + rng.normal(0, n)
        return np.array([elbow, elbow, knee, knee, hip, hip, shoulder_elev, shoulder_elev])

    if exercise == "lateral_raise":
        # Defining pattern: shoulder-elevation sweeps from near-zero (arm
        # at the side) upward; elbow stays roughly constant (mostly
        # straight arm) — this is what separates it from shoulder_press,
        # where the elbow ALSO sweeps a large range.
        elbow = subject.standing_elbow_angle + rng.normal(0, n)
        shoulder_elev = _phase_value(rng, subject.rest_shoulder_elevation, 78, phase, n)
        knee = subject.standing_knee_angle + rng.normal(0, n)
        hip = subject.standing_hip_angle + rng.normal(0, n)
        return np.array([elbow, elbow, knee, knee, hip, hip, shoulder_elev, shoulder_elev])

    raise ValueError(f"unknown exercise: {exercise}")


def generate_dataset(n_subjects: int = 30, examples_per_subject_per_exercise: int = 40,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, subject_ids) — X is (N, 8), y is exercise labels,
    subject_ids identifies which synthetic subject each row came from
    (for the subject-level train/test split in classifier.py)."""
    rng = np.random.default_rng(seed)
    X, y, subject_ids = [], [], []
    for subj_id in range(n_subjects):
        subject = _make_subject(rng)
        for exercise in EXERCISES:
            for _ in range(examples_per_subject_per_exercise):
                X.append(generate_example(rng, exercise, subject))
                y.append(exercise)
                subject_ids.append(subj_id)
    return np.array(X), np.array(y), np.array(subject_ids)
