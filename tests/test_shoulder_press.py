import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from exercises.shoulder_press import ShoulderPressAnalyzer  # noqa: E402

SHOULDER = (0.5, 0.3)
ELBOW = (0.5, 0.5)
HIP = (0.5, 0.8)


def wrist_for_angle(angle_deg, length=0.2):
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


def one_press_frames():
    # Starts AT the rack (contracted zone) — presses start here, unlike
    # curl/squat which start at extended. Needs a LEADING hold (unlike
    # curl's one_rep_frames) because the first zone that needs confirming
    # here is CONTRACTED, right at the start, before any ramp movement —
    # curl's equivalent first-confirmation (also CONTRACTED) happens via a
    # hold in the MIDDLE of its sequence instead, so it didn't need one at
    # the very start. Found by an actual failing test, not by inspection.
    return [85] * 5 + ramp(85, 170, 15) + [170] * 5 + ramp(170, 85, 18) + [85] * 5


def test_auto_side_resolves_to_visible_arm():
    analyzer = ShoulderPressAnalyzer(side="auto")
    out = analyzer.process(make_frame(85, side="left", t=0.0), now=0.0)
    assert out["side"] == "left"


def test_three_clean_presses_counted():
    analyzer = ShoulderPressAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    completed = 0
    last_out = None
    for _ in range(3):
        for a in one_press_frames():
            t += dt
            out = analyzer.process(make_frame(a, t=t), now=t)
            last_out = out
            if out["just_completed_rep"] is not None:
                completed += 1
    assert completed == 3, f"expected 3 completed presses, got {completed}"
    assert last_out["rep_count"] == 3


def test_phase_shows_pressing_and_lowering_directionally():
    analyzer = ShoulderPressAnalyzer(side="left")
    t = 0.0
    dt = 0.05
    seen_pressing = False
    seen_lowering = False
    for a in ramp(85, 170, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "PRESSING":
            seen_pressing = True
    for a in ramp(170, 85, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "LOWERING":
            seen_lowering = True
    assert seen_pressing
    assert seen_lowering


def test_partial_lockout_flagged_on_completed_rep():
    analyzer = ShoulderPressAnalyzer(side="left")
    analyzer.t.partial_lockout_threshold = 200  # force everything to look partial
    t = 0.0
    dt = 0.03
    flagged = False
    for a in one_press_frames():
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["just_completed_rep"] is not None:
            flagged = any(f["level"] == "warn" and "lockout" in f["message"].lower() for f in out["form_flags"])
    assert flagged


def test_full_lockout_not_flagged_as_partial():
    analyzer = ShoulderPressAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    completed_flags = None
    for a in one_press_frames():
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["just_completed_rep"] is not None:
            completed_flags = out["form_flags"]
    assert completed_flags is not None
    assert any(f["level"] == "ok" and "lockout" in f["message"].lower() for f in completed_flags)


def test_shoulder_drift_flagged():
    analyzer = ShoulderPressAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    frame = make_frame(85, t=t)
    analyzer.process(frame, now=t)  # calibrate baseline at rack

    saw_flag = False
    for a in ramp(85, 170, 15):
        t += dt
        out = analyzer.process(make_frame(a, shoulder_offset=(0.1, 0.08), t=t), now=t)
        if any(f["level"] == "warn" and "shoulder" in f["message"].lower() for f in out["form_flags"]):
            saw_flag = True
    assert saw_flag


def test_low_visibility_reports_zero_confidence():
    analyzer = ShoulderPressAnalyzer(side="left")
    out = analyzer.process(make_frame(120, visible=False, t=0.0), now=0.0)
    assert out["confidence"] == 0.0
    assert any("Can't see" in f["message"] for f in out["form_flags"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
