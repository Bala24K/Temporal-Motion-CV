"""
Phase and repetition tests.

Run against analytically generated movement where the correct rep count
is known exactly, so these are assertions rather than judgement calls.
The squat cases matter most: it is the one supported exercise whose
repetition begins with the eccentric phase, and a concentric-first
segmenter counts zero reps for it while looking perfectly healthy on
every other exercise.
"""
import numpy as np
import pytest

from motion.phase import (
    EXERCISE_SIGNALS,
    NUM_PHASES,
    PHASE_INDEX,
    PHASE_NAMES,
    count_reps_from_signal,
    derive_phase_labels,
    movement_signal,
    phase_labels_to_onehot,
    segment_reps_from_phases,
    smooth,
)
from motion.tensors import build_window
from tests.motion_fixtures import curl_clip, neutral_clip, press_clip, raise_clip, squat_clip

CLIP_BUILDERS = {
    "bicep_curl": curl_clip,
    "squat": squat_clip,
    "shoulder_press": press_clip,
    "lateral_raise": raise_clip,
}


def window_coords(exercise: str, cycles: float, n_frames: int = 60) -> np.ndarray:
    clip = CLIP_BUILDERS[exercise](n_frames, cycles)
    window = build_window(clip, target_len=n_frames)
    assert window is not None
    return window.coords


@pytest.mark.parametrize("exercise", sorted(CLIP_BUILDERS))
@pytest.mark.parametrize("cycles", [1, 2, 3])
def test_rep_count_matches_the_number_of_generated_cycles(exercise, cycles):
    coords = window_coords(exercise, cycles)
    labels, diagnostics = derive_phase_labels(coords, exercise)
    reps = segment_reps_from_phases(phase_labels_to_onehot(labels))
    assert len(reps) == cycles, (
        f"{exercise}: expected {cycles} reps, got {len(reps)} "
        f"(phases seen: {sorted({PHASE_NAMES[i] for i in labels})})"
    )
    assert diagnostics["degenerate"] == 0.0


@pytest.mark.parametrize("exercise", sorted(CLIP_BUILDERS))
def test_every_exercise_produces_both_movement_directions(exercise):
    labels, _ = derive_phase_labels(window_coords(exercise, 2), exercise)
    seen = set(labels.tolist())
    assert PHASE_INDEX["CONCENTRIC"] in seen
    assert PHASE_INDEX["ECCENTRIC"] in seen
    assert PHASE_INDEX["PEAK"] in seen


def test_squat_descent_is_eccentric_and_curl_flexion_is_concentric():
    # The physiological convention: rising knee flexion is the eccentric
    # descent, rising elbow flexion is the concentric curl. These must not
    # collapse into a single "signal increasing" phase.
    assert EXERCISE_SIGNALS["squat"].increasing_is_concentric is False
    assert EXERCISE_SIGNALS["bicep_curl"].increasing_is_concentric is True

    squat = window_coords("squat", 1)
    signal = movement_signal(squat, "squat")
    labels, _ = derive_phase_labels(squat, "squat")

    # Find a frame in the first half where the signal is clearly rising.
    smoothed = smooth(signal)
    rising = np.flatnonzero(np.diff(smoothed) > 0)
    rising = rising[rising < len(smoothed) // 2]
    assert rising.size > 0
    mid_descent = int(rising[len(rising) // 2]) + 1
    assert labels[mid_descent] == PHASE_INDEX["ECCENTRIC"]


def test_stationary_pose_is_all_rest_and_flagged_degenerate():
    coords = build_window(neutral_clip(40), target_len=40).coords
    labels, diagnostics = derive_phase_labels(coords, "bicep_curl")
    assert set(labels.tolist()) == {PHASE_INDEX["REST"]}
    assert diagnostics["degenerate"] == 1.0
    assert count_reps_from_signal(coords, "bicep_curl") == 0


def test_tiny_movement_below_the_amplitude_floor_counts_no_reps():
    # Pose jitter must not become reps. Two degrees of elbow travel is
    # noise, not a curl.
    coords = window_coords("bicep_curl", 3)
    flattened = coords.copy()
    centre = flattened.mean(axis=0, keepdims=True)
    flattened = centre + (flattened - centre) * 0.02
    assert count_reps_from_signal(flattened, "bicep_curl") == 0


def test_unknown_exercise_yields_transition_labels_not_a_crash():
    coords = window_coords("bicep_curl", 2)
    labels, diagnostics = derive_phase_labels(coords, "other")
    assert labels.shape == (coords.shape[0],)
    assert set(labels.tolist()) == {PHASE_INDEX["TRANSITION"]}
    assert movement_signal(coords, "other") is None


def test_empty_sequence_is_handled():
    labels, _ = derive_phase_labels(np.zeros((0, 15, 3), dtype=np.float32), "squat")
    assert labels.shape == (0,)
    assert segment_reps_from_phases(np.zeros((0, NUM_PHASES))) == []


def test_segmenter_rejects_wrong_shaped_input():
    with pytest.raises(ValueError):
        segment_reps_from_phases(np.zeros((10, NUM_PHASES + 1)))
    with pytest.raises(ValueError):
        segment_reps_from_phases(np.zeros((10,)))


def test_partial_movement_without_a_turnaround_is_not_a_rep():
    # Rising and sinking back without reaching a turnaround is a false
    # start, not a repetition.
    probs = np.zeros((20, NUM_PHASES))
    probs[:5, PHASE_INDEX["REST"]] = 1.0
    probs[5:12, PHASE_INDEX["CONCENTRIC"]] = 1.0
    probs[12:, PHASE_INDEX["REST"]] = 1.0
    assert segment_reps_from_phases(probs) == []


def test_low_confidence_frames_do_not_start_a_rep():
    probs = np.full((24, NUM_PHASES), 1.0 / NUM_PHASES)  # maximally uncertain
    assert segment_reps_from_phases(probs, min_phase_confidence=0.5) == []


def test_rep_confidence_reflects_model_certainty():
    def cycle(peak_confidence: float) -> np.ndarray:
        probs = np.full((24, NUM_PHASES), (1.0 - peak_confidence) / (NUM_PHASES - 1))
        probs[:8, PHASE_INDEX["CONCENTRIC"]] = peak_confidence
        probs[8:16, PHASE_INDEX["ECCENTRIC"]] = peak_confidence
        probs[16:, PHASE_INDEX["REST"]] = peak_confidence
        return probs

    confident = segment_reps_from_phases(cycle(0.95))
    unsure = segment_reps_from_phases(cycle(0.4))
    assert len(confident) == len(unsure) == 1
    assert confident[0].confidence > unsure[0].confidence


def test_short_bursts_are_rejected_by_min_rep_frames():
    probs = np.zeros((10, NUM_PHASES))
    probs[0:2, PHASE_INDEX["CONCENTRIC"]] = 1.0
    probs[2:4, PHASE_INDEX["ECCENTRIC"]] = 1.0
    probs[4:, PHASE_INDEX["REST"]] = 1.0
    assert segment_reps_from_phases(probs, min_rep_frames=6) == []
    assert len(segment_reps_from_phases(probs, min_rep_frames=2)) == 1


def test_rep_segment_bounds_are_ordered():
    coords = window_coords("bicep_curl", 3)
    labels, _ = derive_phase_labels(coords, "bicep_curl")
    for rep in segment_reps_from_phases(phase_labels_to_onehot(labels)):
        assert rep.start <= rep.peak <= rep.end
        assert rep.duration > 0


def test_smooth_is_nan_aware_and_length_preserving():
    signal = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    out = smooth(signal, window=3)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


def test_phase_labels_to_onehot_round_trips():
    labels = np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)
    onehot = phase_labels_to_onehot(labels)
    assert onehot.shape == (6, NUM_PHASES)
    np.testing.assert_array_equal(onehot.argmax(axis=1), labels)
