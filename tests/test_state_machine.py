"""
M0: synthetic-data test harness for the rep state machine.

These tests feed hand-crafted angle sequences straight into
RepStateMachine — no camera, no pose model, no MediaPipe. The point is to
pin down the counting/hysteresis/timeout logic in isolation, with exact
control over timing, before M1 (camera) and M2 (pose) exist. Reuse this
harness as a regression test forever after.
"""
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercises.base import RepStateMachine, Zone  # noqa: E402


EXTENDED_THRESHOLD = 150.0
CONTRACTED_THRESHOLD = 55.0


def make_sm(**kwargs):
    return RepStateMachine(
        extended_threshold=EXTENDED_THRESHOLD,
        contracted_threshold=CONTRACTED_THRESHOLD,
        min_dwell_seconds=0.08,
        **kwargs,
    )


def feed(sm, angles_and_dts, start_time=0.0):
    """angles_and_dts: list of (angle_or_None, dt_since_previous_reading)."""
    t = start_time
    results = []
    for angle, dt in angles_and_dts:
        t += dt
        r = sm.update(angle, now=t)
        if r is not None:
            results.append(r)
    return results, t


def ramp(a, b, steps):
    """Linearly interpolate from a to b over `steps` samples."""
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def hold(value, steps):
    return [value] * steps


def one_clean_rep_angles():
    """160 -> 40 -> 160, like the doc's worked example, each leg held long
    enough to clear the dwell debounce at a plausible frame rate."""
    seq = []
    seq += ramp(160, 40, 20)   # concentric: ~ (20 samples * 0.02s = 0.4s)
    seq += hold(40, 5)         # brief pause at the bottom
    seq += ramp(40, 160, 25)   # eccentric, a bit slower
    seq += hold(160, 5)        # brief pause at the top
    return seq


def test_clean_single_rep_counts_exactly_once():
    sm = make_sm()
    angles = one_clean_rep_angles()
    data = [(a, 0.02) for a in angles]
    results, _ = feed(sm, data)

    assert sm.rep_count == 1, f"expected 1 rep, got {sm.rep_count}"
    assert len(results) == 1
    rep = results[0]
    # ROM should be close to 160-40=120, allowing for the exact sample points
    assert 110 <= rep.rom <= 121, f"unexpected ROM: {rep.rom}"
    assert rep.concentric_duration > 0, f"concentric_duration should be > 0, got {rep.concentric_duration}"
    assert rep.eccentric_duration > 0, f"eccentric_duration should be > 0, got {rep.eccentric_duration}"


def test_five_clean_reps_count_exactly_five():
    sm = make_sm()
    data = []
    for _ in range(5):
        for a in one_clean_rep_angles():
            data.append((a, 0.02))
    results, _ = feed(sm, data)

    assert sm.rep_count == 5, f"expected 5 reps, got {sm.rep_count}"
    assert [r.rep_number for r in results] == [1, 2, 3, 4, 5]


def test_jitter_near_extended_threshold_does_not_count_reps():
    """Person stands at roughly full extension; sensor noise jitters the
    angle a few degrees around the threshold. The arm never actually
    curls. This must NOT count as reps, even over a long window — this is
    the exact failure mode a naive `if angle < X: rep += 1` counter has."""
    sm = make_sm()
    random.seed(0)
    data = []
    for _ in range(400):
        angle = 152 + random.uniform(-6, 6)  # bounces across the 150 threshold
        data.append((angle, 0.02))
    results, _ = feed(sm, data)

    assert sm.rep_count == 0, f"jitter falsely counted {sm.rep_count} reps"
    assert results == []


def test_partial_rom_never_reaching_contracted_zone_is_not_counted():
    """Arm goes from extended down to 100 degrees (never reaching the
    contracted zone below 55) and back up. Per the state model, a rep
    requires an actual confirmed contraction — a partial movement that
    never gets there isn't a completed rep."""
    sm = make_sm()
    data = []
    for _ in range(3):
        for a in ramp(160, 100, 15) + hold(100, 5) + ramp(100, 160, 15) + hold(160, 5):
            data.append((a, 0.02))
    results, _ = feed(sm, data)

    assert sm.rep_count == 0, f"expected 0 reps for partial ROM, got {sm.rep_count}"


def test_no_double_counting_from_oscillation_at_top_after_a_rep():
    """After completing one rep, the arm oscillates near full extension
    (e.g. tiny tracking jitter while resting at the top) without going
    back down. This must not inflate the count."""
    sm = make_sm()
    data = [(a, 0.02) for a in one_clean_rep_angles()]  # 1 clean rep
    random.seed(1)
    for _ in range(200):
        data.append((158 + random.uniform(-5, 5), 0.02))  # jitter at the top
    results, _ = feed(sm, data)

    assert sm.rep_count == 1, f"oscillation at top caused double-count: {sm.rep_count}"


def test_stuck_mid_rep_times_out_instead_of_hanging_forever():
    """Person curls down, then stops and walks away mid-rep (angle just
    holds steady at a contracted-zone value forever). The state machine
    should give up on completing that rep rather than waiting eternally,
    so a later, unrelated extension doesn't get misread as finishing an
    ancient rep with a bogus multi-hour duration."""
    sm = make_sm()
    data = []
    for a in ramp(160, 40, 20):
        data.append((a, 0.02))
    # hold at the bottom well past STUCK_TIMEOUT_SECONDS
    for _ in range(int((RepStateMachine.STUCK_TIMEOUT_SECONDS + 2) / 0.02)):
        data.append((40, 0.02))
    results, t_after_stuck = feed(sm, data)
    assert results == []
    assert sm.rep_count == 0
    assert sm._armed is False, "armed flag should have been cleared by the stuck-state timeout"

    # now a fresh, clean rep afterward should still count normally
    fresh_data = [(a, 0.02) for a in one_clean_rep_angles()]
    fresh_results, _ = feed(sm, fresh_data, start_time=t_after_stuck)
    assert sm.rep_count == 1
    assert fresh_results[0].duration < 5.0, "post-timeout rep should have a sane duration"


def test_tracking_loss_mid_rep_does_not_produce_a_bogus_rep():
    """Occlusion mid-curl: angle readings go missing (None) for longer than
    DEGRADED_TIMEOUT_SECONDS, then tracking recovers back at full
    extension. This must not be scored as a completed rep off stale
    extremes, and the machine should flag `degraded` while lost."""
    sm = make_sm()
    data = []
    for a in ramp(160, 90, 15):  # partway into the curl
        data.append((a, 0.02))
    lost_samples = int((RepStateMachine.DEGRADED_TIMEOUT_SECONDS + 1) / 0.02)
    for _ in range(lost_samples):
        data.append((None, 0.02))

    t = 0.0
    saw_degraded = False
    for angle, dt in data:
        t += dt
        sm.update(angle, now=t)
        if sm.degraded:
            saw_degraded = True
    assert saw_degraded, "expected the machine to flag degraded tracking"

    # tracking recovers back at extension
    recovery = [(a, 0.02) for a in hold(160, 5)]
    results, t = feed(sm, recovery, start_time=t)
    assert sm.rep_count == 0
    assert results == []
    assert sm.degraded is False, "should clear degraded flag once tracking recovers"

    # and a subsequent clean rep should still work correctly
    fresh_results, _ = feed(sm, [(a, 0.02) for a in one_clean_rep_angles()], start_time=t)
    assert sm.rep_count == 1


def test_variable_frame_rate_does_not_break_counting():
    """Same movement, but sampled at an irregular, sometimes-slow frame
    rate (simulating a laptop that briefly stutters). Dwell time is
    computed from wall-clock time, not frame count, so this should still
    count exactly one rep."""
    sm = make_sm()
    angles = one_clean_rep_angles()
    dts = [random.Random(2).choice([0.015, 0.02, 0.05, 0.09]) for _ in angles]
    data = list(zip(angles, dts))
    results, _ = feed(sm, data)
    assert sm.rep_count == 1, f"variable frame rate broke counting: {sm.rep_count}"


def _run_all():
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
