import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from exercises.lateral_raise import LateralRaiseAnalyzer  # noqa: E402

HIP = (0.5, 0.8)
SHOULDER = (0.5, 0.3)


def wrist_for_angle(angle_deg, length=0.35):
    """Places a wrist point such that angle_3pt(hip, shoulder, wrist) ==
    angle_deg. At angle=0 the wrist is in the same direction as
    shoulder->hip (arm hanging at the side); at angle=90 it's rotated a
    quarter turn away (arm raised to shoulder height)."""
    shoulder = np.array(SHOULDER)
    hip = np.array(HIP)
    v = hip - shoulder
    v_unit = v / np.linalg.norm(v)
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    new_unit = rot @ v_unit
    return tuple(shoulder + new_unit * length)


def make_frame(angle_deg, side="left", visible=True, shoulder_rise=0.0, t=0.0):
    """shoulder_rise > 0 shifts the shoulder landmark UP on screen (image y
    decreases upward) independent of the angle geometry — same decoupling
    trick used in the bicep-curl tests' shoulder_offset, to isolate the
    shrug check from the raise-angle geometry."""
    frame = PoseFrame(timestamp=t)
    vis = 0.9 if visible else 0.05
    wrist = wrist_for_angle(angle_deg)
    shoulder_pos = (SHOULDER[0], SHOULDER[1] - shoulder_rise)
    frame.landmarks[f"{side}_hip"] = Landmark(*HIP, visibility=vis)
    frame.landmarks[f"{side}_shoulder"] = Landmark(*shoulder_pos, visibility=vis)
    frame.landmarks[f"{side}_wrist"] = Landmark(*wrist, visibility=vis)
    return frame


def ramp(a, b, steps):
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def one_raise_frames():
    # Starts at rest (arm at side, contracted zone) — same "needs a
    # leading hold" requirement as shoulder_press.py's test, for the same
    # reason (see that file's comment).
    return [10] * 5 + ramp(10, 80, 15) + [80] * 5 + ramp(80, 10, 18) + [10] * 5


def test_auto_side_resolves_to_visible_arm():
    analyzer = LateralRaiseAnalyzer(side="auto")
    out = analyzer.process(make_frame(10, side="left", t=0.0), now=0.0)
    assert out["side"] == "left"


def test_three_clean_raises_counted():
    analyzer = LateralRaiseAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    completed = 0
    last_out = None
    for _ in range(3):
        for a in one_raise_frames():
            t += dt
            out = analyzer.process(make_frame(a, t=t), now=t)
            last_out = out
            if out["just_completed_rep"] is not None:
                completed += 1
    assert completed == 3, f"expected 3 completed raises, got {completed}"
    assert last_out["rep_count"] == 3


def test_phase_shows_raising_and_lowering_directionally():
    analyzer = LateralRaiseAnalyzer(side="left")
    t = 0.0
    dt = 0.05
    seen_raising = False
    seen_lowering = False
    for a in ramp(10, 80, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "RAISING":
            seen_raising = True
    for a in ramp(80, 10, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "LOWERING":
            seen_lowering = True
    assert seen_raising
    assert seen_lowering


def test_partial_range_flagged_on_completed_rep():
    analyzer = LateralRaiseAnalyzer(side="left")
    analyzer.t.partial_rom_threshold = 200  # force everything to look partial
    t = 0.0
    dt = 0.03
    flagged = False
    for a in one_raise_frames():
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["just_completed_rep"] is not None:
            flagged = any(f["level"] == "warn" and "partial" in f["message"].lower() for f in out["form_flags"])
    assert flagged


def test_good_range_not_flagged_partial():
    analyzer = LateralRaiseAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    completed_flags = None
    for a in one_raise_frames():
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["just_completed_rep"] is not None:
            completed_flags = out["form_flags"]
    assert completed_flags is not None
    assert any(f["level"] == "ok" and "range" in f["message"].lower() for f in completed_flags)


def test_shoulder_shrug_flagged():
    analyzer = LateralRaiseAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    analyzer.process(make_frame(10, shoulder_rise=0.0, t=t), now=t)  # calibrate baseline

    saw_flag = False
    for a in ramp(10, 80, 15):
        t += dt
        out = analyzer.process(make_frame(a, shoulder_rise=0.15, t=t), now=t)
        if any(f["level"] == "warn" and "shrug" in f["message"].lower() for f in out["form_flags"]):
            saw_flag = True
    assert saw_flag


def test_no_shrug_does_not_flag():
    analyzer = LateralRaiseAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    saw_flag = False
    for a in ramp(10, 80, 15):
        t += dt
        out = analyzer.process(make_frame(a, shoulder_rise=0.0, t=t), now=t)
        if any(f["level"] == "warn" and "shrug" in f["message"].lower() for f in out["form_flags"]):
            saw_flag = True
    assert not saw_flag


def test_low_visibility_reports_zero_confidence():
    analyzer = LateralRaiseAnalyzer(side="left")
    out = analyzer.process(make_frame(40, visible=False, t=0.0), now=0.0)
    assert out["confidence"] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
