import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.rep_analyzer import analyze_rep  # noqa: E402
from analysis.set_analyzer import summarize_set, _consistency_score  # noqa: E402
from analysis.baseline import PersonalBaseline, RunningStat  # noqa: E402


def ramp(a, b, steps):
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def curl_rep(low, high, n=20, dt=0.05, t0=0.0):
    half = n // 2
    angles = ramp(high, low, half) + ramp(low, high, n - half)
    times = [t0 + i * dt for i in range(len(angles))]
    concentric = (half - 1) * dt
    eccentric = (len(angles) - half) * dt
    return times, angles, concentric, eccentric


# ---------- rep_analyzer ----------

def test_analyze_rep_basic_metrics():
    times, angles, c, e = curl_rep(40, 160)
    m = analyze_rep(1, times, angles, c, e)
    assert m.rep_number == 1
    assert abs(m.rom - 120) < 1.0
    assert abs(m.tempo - (c + e)) < 1e-9
    assert m.symmetry is None
    assert m.trajectory_similarity is None


def test_analyze_rep_with_symmetry():
    times, angles, c, e = curl_rep(40, 160)
    m = analyze_rep(1, times, angles, c, e,
                     other_side_times=times, other_side_angles=angles)
    assert m.symmetry is not None
    assert m.symmetry.symmetry_score > 99


def test_analyze_rep_with_trajectory_reference():
    times, angles, c, e = curl_rep(40, 160, n=20)
    reference = curl_rep(40, 160, n=30)[1]  # same shape, different sample count
    m = analyze_rep(1, times, angles, c, e, reference_trajectory=reference)
    assert m.trajectory_similarity is not None
    assert m.trajectory_similarity > 85


# ---------- set_analyzer ----------

def test_consistency_score_perfect_for_identical_values():
    assert _consistency_score([120.0, 120.0, 120.0]) == 100.0


def test_consistency_score_drops_with_variance():
    tight = _consistency_score([120, 119, 121, 120])
    loose = _consistency_score([120, 90, 140, 80])
    assert tight > loose


def test_summarize_set_empty_returns_none():
    assert summarize_set([]) is None


def test_summarize_set_basic_aggregation():
    reps = []
    for i in range(6):
        times, angles, c, e = curl_rep(40, 160, t0=i * 3.0)
        reps.append(analyze_rep(i + 1, times, angles, c, e))
    summary = summarize_set(reps)
    assert summary.rep_count == 6
    assert abs(summary.avg_rom - 120) < 2.0
    assert summary.rom_consistency > 95  # all reps identical shape here


def test_progressive_rom_decline_is_detected_and_matches_docs_worked_example():
    """
    Reproduces the doc's own worked example: a set where ROM declines over
    the final reps should surface exactly the kind of observation shown
    in the V2 writeup ('ROM decreased during final reps'), and NOT invent
    a claim about fatigue, muscles, or cause.
    """
    roms = [124, 122, 121, 119, 113, 108, 104, 102, 100, 98]  # matches doc's shape
    reps = []
    t = 0.0
    for i, rom in enumerate(roms):
        times, angles, c, e = curl_rep(160 - rom, 160, n=20, t0=t)
        reps.append(analyze_rep(i + 1, times, angles, c, e))
        t += 3.0

    summary = summarize_set(reps)
    assert summary.rep_count == 10
    assert summary.lowest_rom < summary.best_rom
    assert any("ROM decreased" in obs for obs in summary.observations), \
        f"expected a ROM-decline observation, got: {summary.observations}"
    # must not claim WHY — only mention exercise-adjacent movement facts
    for obs in summary.observations:
        assert "fatigue" not in obs.lower()
        assert "muscle" not in obs.lower()
        assert "injury" not in obs.lower()


def test_stable_set_produces_no_false_trend_observations():
    reps = []
    t = 0.0
    for i in range(10):
        times, angles, c, e = curl_rep(40, 160, t0=t)  # identical every rep
        reps.append(analyze_rep(i + 1, times, angles, c, e))
        t += 3.0
    summary = summarize_set(reps)
    assert summary.observations == [], f"expected no trend claims on a stable set, got {summary.observations}"


def test_short_set_does_not_claim_a_trend():
    """Trend claims from 3 data points aren't defensible — min_reps guard."""
    reps = []
    t = 0.0
    for i, rom in enumerate([124, 110, 95]):
        times, angles, c, e = curl_rep(160 - rom, 160, t0=t)
        reps.append(analyze_rep(i + 1, times, angles, c, e))
        t += 3.0
    summary = summarize_set(reps)
    assert summary.observations == []


# ---------- baseline ----------

def test_running_stat_matches_known_mean_and_std():
    stat = RunningStat()
    values = [10, 12, 23, 23, 16, 23, 21, 16]
    for v in values:
        stat.update(v)
    assert abs(stat.mean - (sum(values) / len(values))) < 1e-9
    # sample std of this dataset is a known value (~5.237)
    assert abs(stat.std - 5.2372293) < 1e-4


def test_baseline_not_established_before_min_samples():
    b = PersonalBaseline(min_samples=5)
    for v in [120, 121, 119]:
        b.update("rom", v)
    assert not b.is_established("rom")
    assert b.z_score("rom", 200) is None
    assert b.is_deviation("rom", 200) is None


def test_baseline_flags_deviation_once_established():
    b = PersonalBaseline(min_samples=5)
    for v in [120, 121, 119, 120, 122, 118, 121]:
        b.update("rom", v)
    assert b.is_established("rom")
    assert b.is_deviation("rom", 121) is False   # well within normal range
    assert b.is_deviation("rom", 60) is True     # way outside normal range


def test_baseline_serialization_round_trips():
    b = PersonalBaseline(min_samples=3)
    for v in [120, 118, 122, 121]:
        b.update("rom", v)
    d = b.to_dict()
    b2 = PersonalBaseline.from_dict(d)
    assert b2.is_established("rom")
    assert abs(b2.summary("rom")["mean"] - b.summary("rom")["mean"]) < 1e-9
    assert abs(b2.summary("rom")["std"] - b.summary("rom")["std"]) < 1e-9


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
