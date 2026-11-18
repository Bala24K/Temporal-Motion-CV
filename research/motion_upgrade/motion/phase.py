"""
Movement phases and repetition boundaries.

Two jobs, deliberately kept in one module because they are two views of
the same signal:

  1. DERIVING phase labels from kinematics, so the temporal model has a
     per-frame target to learn (`derive_phase_labels`). These are weak
     labels — computed from the movement signal, not annotated by a
     human — and every function that produces them says so in its return
     value. A phase model trained on them can only ever be as good as
     the rule, which is exactly the trap the exercise labels already fell
     into. The difference here is that it is *stated*, and the
     `PhaseLabelSource` marker travels with the data all the way into the
     checkpoint.

  2. SEGMENTING reps from predicted phase probabilities
     (`segment_reps_from_phases`), which is the learned replacement for
     the angle-threshold state machine in `exercises/base.py`. The
     existing state machine is not deleted: it stays as the baseline this
     is measured against, and as the fallback when the model is not
     confident.

PHASE CONVENTION
    Phases are named physiologically (concentric = muscle shortening),
    which means the mapping from "movement signal increasing" to
    "concentric" is exercise-dependent: flexing into a curl is
    concentric, but flexing into a squat is the eccentric descent. That
    flag lives in EXERCISE_SIGNALS rather than being assumed globally.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from motion.graph import JOINT_INDEX

PHASE_NAMES: List[str] = ["REST", "CONCENTRIC", "PEAK", "ECCENTRIC", "TRANSITION"]
PHASE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(PHASE_NAMES)}
NUM_PHASES = len(PHASE_NAMES)

# Marker recorded alongside derived labels so a later human-annotated
# dataset can be mixed in without the two being confused.
PHASE_LABEL_SOURCE_DERIVED = "derived_from_kinematics"
PHASE_LABEL_SOURCE_HUMAN = "human_annotated"


def _angle_at(coords: np.ndarray, a: str, b: str, c: str) -> np.ndarray:
    """Angle ABC in degrees per frame, shape (T,), using x/y only.

    z is excluded for the same reason `pose/normalization.py` excludes it
    from the torso basis: single-camera depth is too noisy to put inside
    an arccos.
    """
    pa = coords[:, JOINT_INDEX[a], :2]
    pb = coords[:, JOINT_INDEX[b], :2]
    pc = coords[:, JOINT_INDEX[c], :2]
    ba, bc = pa - pb, pc - pb
    na = np.linalg.norm(ba, axis=1)
    nc = np.linalg.norm(bc, axis=1)
    denom = np.maximum(na * nc, 1e-8)
    cos = np.clip((ba * bc).sum(axis=1) / denom, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang[(na < 1e-6) | (nc < 1e-6)] = np.nan
    return ang


def _bilateral(coords: np.ndarray, fn: Callable[[np.ndarray, str], np.ndarray]) -> np.ndarray:
    """
    Evaluate a signal for both sides and keep the side with the larger
    excursion.

    Single-arm and single-leg variants are common, and averaging the two
    sides halves the amplitude of a one-sided movement — enough to push
    a real rep under the detection threshold.
    """
    left = fn(coords, "left")
    right = fn(coords, "right")

    def excursion(s: np.ndarray) -> float:
        finite = s[np.isfinite(s)]
        return float(finite.max() - finite.min()) if finite.size > 1 else -np.inf

    return left if excursion(left) >= excursion(right) else right


def _elbow_flexion(coords: np.ndarray) -> np.ndarray:
    """Higher = more flexed (curled)."""
    return _bilateral(coords, lambda c, s: 180.0 - _angle_at(c, f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"))


def _knee_flexion(coords: np.ndarray) -> np.ndarray:
    """Higher = deeper squat."""
    return _bilateral(coords, lambda c, s: 180.0 - _angle_at(c, f"{s}_hip", f"{s}_knee", f"{s}_ankle"))


def _wrist_elevation(coords: np.ndarray) -> np.ndarray:
    """
    Wrist height above the shoulder line, in torso-length units.

    Image y grows downward and normalisation preserves that, so height is
    the negated y difference.
    """
    def side(c: np.ndarray, s: str) -> np.ndarray:
        wrist_y = c[:, JOINT_INDEX[f"{s}_wrist"], 1]
        sho_y = c[:, JOINT_INDEX[f"{s}_shoulder"], 1]
        return sho_y - wrist_y

    return _bilateral(coords, side)


def _shoulder_abduction(coords: np.ndarray) -> np.ndarray:
    """Torso-to-upper-arm angle; higher = arm raised further from the body."""
    return _bilateral(coords, lambda c, s: _angle_at(c, "hip_center", f"{s}_shoulder", f"{s}_elbow"))


@dataclass(frozen=True)
class ExerciseSignal:
    """
    How to turn a skeleton window into the one scalar per frame that
    drives phase and rep detection for a given exercise.

    increasing_is_concentric — whether the signal rising corresponds to
    the concentric (shortening) phase. False for the squat, where rising
    knee flexion is the eccentric descent.
    min_amplitude — excursion below which the window is treated as no
    movement at all, in the signal's own units (degrees, or torso lengths
    for elevation).
    """
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    increasing_is_concentric: bool
    min_amplitude: float


EXERCISE_SIGNALS: Dict[str, ExerciseSignal] = {
    "bicep_curl": ExerciseSignal("elbow_flexion", _elbow_flexion, True, 25.0),
    "squat": ExerciseSignal("knee_flexion", _knee_flexion, False, 20.0),
    "shoulder_press": ExerciseSignal("wrist_elevation", _wrist_elevation, True, 0.35),
    "lateral_raise": ExerciseSignal("shoulder_abduction", _shoulder_abduction, True, 25.0),
}


def movement_signal(coords: np.ndarray, exercise: str) -> Optional[np.ndarray]:
    """Per-frame scalar signal for an exercise, shape (T,), or None if the
    exercise has no defined signal (e.g. the `other` class)."""
    spec = EXERCISE_SIGNALS.get(exercise)
    if spec is None:
        return None
    return spec.fn(coords)


def smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Centred moving average with edge padding, NaN-aware.

    Pose jitter at 7.5 effective fps produces velocity sign flips on
    nearly every frame; without smoothing the phase labels alternate
    CONCENTRIC/ECCENTRIC frame to frame and are useless as targets.
    """
    if signal.size == 0:
        return signal.copy()
    window = max(1, int(window))
    if window == 1:
        return signal.astype(np.float64, copy=True)

    s = signal.astype(np.float64)
    finite = np.isfinite(s)
    if not finite.any():
        return s
    # Fill NaNs by nearest-observed before averaging so a single dropout
    # doesn't erase a whole window's worth of signal.
    idx = np.flatnonzero(finite)
    filled = np.interp(np.arange(s.size), idx, s[idx])

    pad = window // 2
    padded = np.pad(filled, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    out = np.convolve(padded, kernel, mode="valid")
    return out[: s.size]


def _normalize_signal(signal: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Scale to [0, 1] using robust percentiles, returning (normalised,
    amplitude_in_original_units).

    Percentiles rather than min/max: one blown-out frame from a pose
    glitch would otherwise set the range and squash the real movement
    into a narrow band in the middle.
    """
    finite = signal[np.isfinite(signal)]
    if finite.size < 2:
        return np.zeros_like(signal), 0.0
    lo, hi = np.percentile(finite, [5.0, 95.0])
    amplitude = float(hi - lo)
    if amplitude < 1e-6:
        return np.zeros_like(signal), 0.0
    return np.clip((signal - lo) / amplitude, 0.0, 1.0), amplitude


def _enforce_min_duration(labels: np.ndarray, min_len: int) -> np.ndarray:
    """
    Absorb runs shorter than `min_len` into the preceding run.

    Without this, a single noisy frame inside a smooth descent becomes a
    one-frame PEAK, which later reads as a completed rep.
    """
    if labels.size == 0 or min_len <= 1:
        return labels
    out = labels.copy()
    start = 0
    for i in range(1, out.size + 1):
        if i == out.size or out[i] != out[start]:
            if (i - start) < min_len and start > 0:
                out[start:i] = out[start - 1]
            start = i
    return out


def derive_phase_labels(
    coords: np.ndarray,
    exercise: str,
    velocity_threshold: float = 0.045,
    rest_level: float = 0.18,
    peak_level: float = 0.82,
    smooth_window: int = 5,
    min_phase_frames: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Per-frame phase labels, shape (T,) int64, plus a diagnostics dict.

    These are WEAK labels derived from the movement signal. They are good
    enough to teach a temporal model the shape of a repetition — which is
    genuinely useful, because the model then generalises the phase
    boundary beyond the single joint angle the rule looks at — but a
    phase-accuracy number measured against them is measuring agreement
    with the rule, not with a human. Report it that way.

    Windows whose excursion is below the exercise's `min_amplitude` are
    labelled entirely REST and flagged via diagnostics["degenerate"].
    """
    T = int(coords.shape[0])
    diagnostics: Dict[str, float] = {"amplitude": 0.0, "degenerate": 1.0, "label_source": 0.0}
    if T == 0:
        return np.zeros((0,), dtype=np.int64), diagnostics

    spec = EXERCISE_SIGNALS.get(exercise)
    raw = movement_signal(coords, exercise)
    if spec is None or raw is None:
        return np.full((T,), PHASE_INDEX["TRANSITION"], dtype=np.int64), diagnostics

    sm = smooth(raw, window=smooth_window)
    norm, amplitude = _normalize_signal(sm)
    diagnostics["amplitude"] = amplitude

    if amplitude < spec.min_amplitude:
        return np.full((T,), PHASE_INDEX["REST"], dtype=np.int64), diagnostics

    diagnostics["degenerate"] = 0.0

    velocity = np.zeros_like(norm)
    if T > 1:
        velocity[1:] = np.diff(norm)
        velocity[0] = velocity[1]

    rising_phase = PHASE_INDEX["CONCENTRIC"] if spec.increasing_is_concentric else PHASE_INDEX["ECCENTRIC"]
    falling_phase = PHASE_INDEX["ECCENTRIC"] if spec.increasing_is_concentric else PHASE_INDEX["CONCENTRIC"]

    labels = np.full((T,), PHASE_INDEX["TRANSITION"], dtype=np.int64)
    slow = np.abs(velocity) < velocity_threshold

    labels[velocity >= velocity_threshold] = rising_phase
    labels[velocity <= -velocity_threshold] = falling_phase
    labels[slow & (norm <= rest_level)] = PHASE_INDEX["REST"]
    labels[slow & (norm >= peak_level)] = PHASE_INDEX["PEAK"]

    labels = _enforce_min_duration(labels, min_phase_frames)
    return labels, diagnostics


@dataclass
class RepSegment:
    """One detected repetition, as frame indices into the window."""
    start: int
    peak: int
    end: int
    confidence: float

    @property
    def duration(self) -> int:
        return self.end - self.start


def segment_reps_from_phases(
    phase_probs: np.ndarray,
    min_rep_frames: int = 4,
    min_phase_confidence: float = 0.35,
) -> List[RepSegment]:
    """
    Turn per-frame phase probabilities, shape (T, NUM_PHASES), into
    repetition boundaries.

    A rep is one out-and-back excursion: a run in one movement direction,
    a turnaround (usually through PEAK), then a run in the opposite
    direction. Deliberately direction-AGNOSTIC — which direction comes
    first is exercise-dependent. A curl begins with the concentric
    flexion; a squat begins with the eccentric descent. Hard-coding
    concentric-first counts zero reps for every squat, which is a
    plausible-looking failure the phase labels themselves do not reveal.

    The rep closes when the return run ends, so it is credited when the
    lifter arrives back at the start position — the same instant
    `exercises/base.py` credits it, which keeps the learned and
    deterministic counters directly comparable.

    Confidence is the mean probability assigned to the phases actually
    traversed, so a rep stitched together from frames the model was
    unsure about is reported as a low-confidence rep rather than being
    silently equal to a clean one.
    """
    if phase_probs.ndim != 2 or phase_probs.shape[1] != NUM_PHASES:
        raise ValueError(f"expected phase_probs of shape (T, {NUM_PHASES}), got {phase_probs.shape}")

    T = phase_probs.shape[0]
    if T == 0:
        return []

    argmax = phase_probs.argmax(axis=1)
    top = phase_probs.max(axis=1)
    conc, ecc, rest = PHASE_INDEX["CONCENTRIC"], PHASE_INDEX["ECCENTRIC"], PHASE_INDEX["REST"]
    directions = (conc, ecc)

    reps: List[RepSegment] = []
    state = "idle"
    outward = conc
    start = peak = 0
    traversed: List[float] = []

    def opposite(direction: int) -> int:
        return ecc if direction == conc else conc

    for t in range(T):
        label = argmax[t]
        confident = top[t] >= min_phase_confidence

        if state == "idle":
            if label in directions and confident:
                state, outward, start, peak = "outward", label, t, t
                traversed = [float(top[t])]

        elif state == "outward":
            traversed.append(float(top[t]))
            if label == opposite(outward) and confident:
                state, peak = "returning", t
            elif label == rest and confident:
                state = "idle"  # sank back to rest without turning around

        elif state == "returning":
            traversed.append(float(top[t]))
            restarted = label == outward and confident
            if (label == rest and confident) or restarted or t == T - 1:
                if (t - start) >= min_rep_frames:
                    reps.append(RepSegment(start, peak, t, float(np.mean(traversed))))
                if restarted:
                    state, start, peak = "outward", t, t
                    traversed = [float(top[t])]
                else:
                    state = "idle"

    return reps


def phase_labels_to_onehot(labels: np.ndarray) -> np.ndarray:
    """(T,) int labels -> (T, NUM_PHASES) float32, for use as soft targets
    or for feeding derived labels through the same code path as model
    predictions in tests."""
    out = np.zeros((labels.shape[0], NUM_PHASES), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def count_reps_from_signal(coords: np.ndarray, exercise: str) -> int:
    """
    Deterministic rep count straight from the kinematic signal, used as
    the baseline the learned segmentation is compared against in
    `motion/evaluate.py`. Intentionally simple: peak counting on the
    smoothed, normalised signal with a prominence floor.
    """
    labels, diag = derive_phase_labels(coords, exercise)
    if diag["degenerate"] >= 1.0:
        return 0
    return len(segment_reps_from_phases(phase_labels_to_onehot(labels)))
