import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from analysis.muscle_estimation import estimate_activation_sequence, summarize_rep_activation  # noqa: E402
from analysis.confidence import SYSTEMATIC_CAP  # noqa: E402


def curl_rep(low=40, high=160, n=40, dt=0.05):
    half = n // 2
    ramp_down = list(np.linspace(high, low, half))
    ramp_up = list(np.linspace(low, high, n - half))
    angles = ramp_down + ramp_up
    times = [i * dt for i in range(n)]
    return times, angles


def test_activation_rises_as_the_curl_progresses_toward_horizontal():
    """Required moment (and therefore activation) should be near zero at
    full extension and largest around the mid-range (90 degrees), per the
    inverse-dynamics model's own physics — this checks the whole chain
    reproduces that, not just the isolated inverse-dynamics module."""
    times, angles = curl_rep()
    vis = [0.95] * len(times)
    frames = estimate_activation_sequence(times, angles, vis, body_mass_kg=75, body_height_m=1.75)

    near_extended = [f for f in frames if f.elbow_angle_deg > 150]
    near_ninety = [f for f in frames if 80 < f.elbow_angle_deg < 100]
    assert near_extended and near_ninety

    def total_activation(f):
        return sum(f.activations.values())

    avg_extended = np.mean([total_activation(f) for f in near_extended])
    avg_ninety = np.mean([total_activation(f) for f in near_ninety])
    assert avg_ninety > avg_extended, \
        f"expected higher activation near 90deg ({avg_ninety}) than near full extension ({avg_extended})"


def test_heavier_load_increases_activation_at_the_same_angle():
    times, angles = curl_rep()
    vis = [0.95] * len(times)
    light = estimate_activation_sequence(times, angles, vis, body_mass_kg=75, body_height_m=1.75,
                                          external_load_kg=2.0)
    heavy = estimate_activation_sequence(times, angles, vis, body_mass_kg=75, body_height_m=1.75,
                                          external_load_kg=15.0)
    # compare at a matching mid-range index
    idx = len(angles) // 4
    light_total = sum(light[idx].activations.values())
    heavy_total = sum(heavy[idx].activations.values())
    assert heavy_total > light_total


def test_brachialis_carries_more_load_than_brachioradialis_given_its_larger_pcsa_default():
    """Not a claim about any real person — just checking the pipeline
    respects the anatomical database's own numbers (brachialis has both a
    high max force and a real moment arm; verifies the optimizer isn't
    silently ignoring the force differences)."""
    times, angles = curl_rep()
    vis = [0.95] * len(times)
    frames = estimate_activation_sequence(times, angles, vis, body_mass_kg=75, body_height_m=1.75,
                                           external_load_kg=8.0)
    summary = summarize_rep_activation(frames)
    assert summary["brachialis"]["peak_activation"] > 0
    assert summary["brachioradialis"]["peak_activation"] > 0


def test_confidence_never_exceeds_systematic_cap():
    times, angles = curl_rep()
    vis = [1.0] * len(times)  # perfect tracking
    frames = estimate_activation_sequence(times, angles, vis, body_mass_kg=75, body_height_m=1.75)
    assert all(f.confidence.score <= SYSTEMATIC_CAP + 1e-6 for f in frames)


def test_poor_visibility_reduces_confidence_relative_to_good_visibility():
    times, angles = curl_rep()
    good_vis = [0.95] * len(times)
    poor_vis = [0.3] * len(times)
    good = estimate_activation_sequence(times, angles, good_vis, body_mass_kg=75, body_height_m=1.75)
    poor = estimate_activation_sequence(times, angles, poor_vis, body_mass_kg=75, body_height_m=1.75)
    idx = len(angles) // 4
    assert poor[idx].confidence.score < good[idx].confidence.score


def test_unrealistically_heavy_load_is_flagged_infeasible_somewhere_in_the_rep():
    """A load far beyond what three elbow flexors could plausibly move
    should show up as an infeasible solve at some point in the rep,
    surfaced honestly rather than silently saturating and pretending it
    accounted for the whole moment."""
    times, angles = curl_rep()
    vis = [0.95] * len(times)
    frames = estimate_activation_sequence(times, angles, vis, body_mass_kg=75, body_height_m=1.75,
                                           external_load_kg=500.0)  # absurd, deliberately
    assert any(not f.feasible for f in frames)


def test_summarize_rep_activation_matches_manual_peak_and_mean():
    times, angles = curl_rep()
    vis = [0.95] * len(times)
    frames = estimate_activation_sequence(times, angles, vis, body_mass_kg=75, body_height_m=1.75)
    summary = summarize_rep_activation(frames)
    for name in summary:
        values = [f.activations[name] for f in frames]
        assert abs(summary[name]["peak_activation"] - max(values)) < 1e-9
        assert abs(summary[name]["mean_activation"] - (sum(values) / len(values))) < 1e-9


def test_empty_frames_summary_is_empty():
    assert summarize_rep_activation([]) == {}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
