import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from exercises.bicep_curl import BicepCurlAnalyzer  # noqa: E402

SHOULDER = (0.5, 0.3)
ELBOW = (0.5, 0.5)
HIP = (0.5, 0.8)


def wrist_for_angle(angle_deg, length=0.2):
    """Places a wrist point such that angle_3pt(shoulder, elbow, wrist) == angle_deg."""
    shoulder = np.array(SHOULDER)
    elbow = np.array(ELBOW)
    ba = shoulder - elbow
    ba_unit = ba / np.linalg.norm(ba)
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    bc_unit = rot @ ba_unit
    wrist = elbow + bc_unit * length
    return tuple(wrist)


def make_frame(angle_deg, side="left", visible=True, shoulder_offset=(0.0, 0.0), t=0.0):
    frame = PoseFrame(timestamp=t)
    vis = 0.9 if visible else 0.05
    sx, sy = SHOULDER[0] + shoulder_offset[0], SHOULDER[1] + shoulder_offset[1]
    wx, wy = wrist_for_angle(angle_deg)
    frame.landmarks[f"{side}_shoulder"] = Landmark(sx, sy, visibility=vis)
    frame.landmarks[f"{side}_elbow"] = Landmark(*ELBOW, visibility=vis)
    frame.landmarks[f"{side}_wrist"] = Landmark(wx, wy, visibility=vis)
    frame.landmarks[f"{side}_hip"] = Landmark(*HIP, visibility=vis)
    return frame


def ramp(a, b, steps):
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def one_rep_frames(side="left"):
    angles = ramp(160, 40, 15) + [40] * 5 + ramp(40, 160, 18) + [160] * 5
    return angles


def has_flag(form_flags, level, substring):
    """form_flags are {"level": "ok"|"warn", "message": str} dicts (see
    exercises/base.py ok()/warn()) — not glyph-prefixed strings, since
    cv2.putText can't render checkmark/warning symbols."""
    return any(f["level"] == level and substring in f["message"] for f in form_flags)


def test_auto_side_resolves_to_visible_arm():
    analyzer = BicepCurlAnalyzer(side="auto")
    frame = make_frame(160, side="left", t=0.0)
    out = analyzer.process(frame, now=0.0)
    assert out["side"] == "left"


def test_phase_label_shows_curling_and_lowering_directionally():
    """Regression test for a real bug: an earlier version captured
    self._prev_angle for the direction check AFTER it had already been
    overwritten with the current frame's angle, so the comparison was
    always angle-vs-itself and direction detection silently always
    returned None — CURLING/LOWERING never actually displayed, only the
    generic MOVING fallback. This checks the actual displayed phase
    string, not just rep-counting behavior, which wouldn't have caught it."""
    analyzer = BicepCurlAnalyzer(side="left")
    t = 0.0
    dt = 0.05
    seen_curling = False
    seen_lowering = False
    for a in ramp(160, 40, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "CURLING":
            seen_curling = True
    for a in ramp(40, 160, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "LOWERING":
            seen_lowering = True
    assert seen_curling, "expected to see CURLING while the angle was decreasing"
    assert seen_lowering, "expected to see LOWERING while the angle was increasing"


def test_three_clean_reps_counted_through_full_analyzer():
    analyzer = BicepCurlAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    completed = 0
    last_out = None
    for _ in range(3):
        for a in one_rep_frames():
            t += dt
            frame = make_frame(a, t=t)
            out = analyzer.process(frame, now=t)
            last_out = out
            if out["just_completed_rep"] is not None:
                completed += 1

    assert completed == 3, f"expected 3 completed reps, got {completed}"
    assert last_out["rep_count"] == 3
    assert last_out["rom"] is not None and last_out["rom"] > 90


def test_partial_rom_flagged_on_completed_rep():
    """A rep that does complete (reaches contracted zone) but with a small
    ROM should be flagged, not silently accepted as a good rep."""
    thresholds_analyzer = BicepCurlAnalyzer(side="left")
    thresholds_analyzer.t.partial_rom_threshold = 200  # force everything to look "partial" for this test
    t = 0.0
    dt = 0.03
    flagged = False
    for a in one_rep_frames():
        t += dt
        frame = make_frame(a, t=t)
        out = thresholds_analyzer.process(frame, now=t)
        if out["just_completed_rep"] is not None:
            flagged = has_flag(out["form_flags"], "warn", "Partial ROM")
    assert flagged, "expected the completed rep to be flagged as partial ROM"


def test_shoulder_compensation_flagged_when_shoulder_drifts():
    analyzer = BicepCurlAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    # calibrate with shoulder at baseline, extended
    frame = make_frame(160, t=t)
    analyzer.process(frame, now=t)

    saw_flag = False
    for a in ramp(160, 40, 15):
        t += dt
        # shoulder drifts substantially while curling — compensation pattern
        frame = make_frame(a, shoulder_offset=(0.08, 0.05), t=t)
        out = analyzer.process(frame, now=t)
        if has_flag(out["form_flags"], "warn", "Possible shoulder compensation"):
            saw_flag = True
    assert saw_flag, "expected shoulder drift to trigger a compensation flag"


def test_low_visibility_does_not_crash_and_reports_low_confidence():
    analyzer = BicepCurlAnalyzer(side="left")
    frame = make_frame(120, visible=False, t=0.0)
    out = analyzer.process(frame, now=0.0)
    assert out["confidence"] == 0.0
    assert has_flag(out["form_flags"], "warn", "Can't see")


def test_occlusion_mid_rep_recovers_without_corrupting_count():
    analyzer = BicepCurlAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    # partway into a curl
    for a in ramp(160, 100, 10):
        t += dt
        analyzer.process(make_frame(a, t=t), now=t)

    # lose tracking for a while
    for _ in range(100):
        t += dt
        analyzer.process(make_frame(100, visible=False, t=t), now=t)

    # recover at extended
    for _ in range(5):
        t += dt
        out = analyzer.process(make_frame(160, t=t), now=t)

    assert out["rep_count"] == 0

    # a subsequent clean rep should still work
    completed = 0
    for a in one_rep_frames():
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["just_completed_rep"] is not None:
            completed += 1
    assert completed == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
