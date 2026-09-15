import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pose.landmarks import PoseFrame, Landmark  # noqa: E402
from exercises.squat import SquatAnalyzer  # noqa: E402

HIP = (0.5, 0.5)
HIP_HALF_WIDTH = 0.08


def rotate_for_angle(vertex, reference_point, angle_deg, length=0.35):
    """Places a point P such that angle_3pt(reference_point, vertex, P) == angle_deg."""
    vertex = np.array(vertex)
    reference_point = np.array(reference_point)
    v = reference_point - vertex
    v_unit = v / np.linalg.norm(v)
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    new_unit = rot @ v_unit
    return tuple(vertex + new_unit * length)


def make_frame(knee_angle_deg, side="left", visible=True, lean_deg=0.0, knee_width_ratio=1.0, t=0.0):
    """
    Builds a two-legged synthetic frame. The LEFT leg's hip/knee/ankle
    drive the tracked angle (via rotate_for_angle, same trick used in the
    other exercise tests) — the RIGHT knee/hip exist so the bilateral
    valgus ratio check (comparing inter-knee to inter-hip distance) has
    something to compare against. knee_width_ratio=1.0 means both knees
    are exactly as far apart as both hips (normal); < 1 means the knees
    have moved closer together than the hips (valgus pattern), independent
    of how deep the tracked leg's bend is — see squat.py's
    _knee_valgus_ratio docstring for why this bilateral comparison is used
    instead of a single-leg one.
    """
    frame = PoseFrame(timestamp=t)
    vis = 0.9 if visible else 0.05
    hip_cx = HIP[0]
    left_hip = (hip_cx - HIP_HALF_WIDTH, HIP[1])
    right_hip = (hip_cx + HIP_HALF_WIDTH, HIP[1])

    knee_half_width = HIP_HALF_WIDTH * knee_width_ratio
    knee_y = HIP[1] + 0.25
    left_knee = (hip_cx - knee_half_width, knee_y)
    right_knee = (hip_cx + knee_half_width, knee_y)

    ankle = rotate_for_angle(left_knee, left_hip, knee_angle_deg)

    lean_rad = np.radians(lean_deg)
    shoulder = (left_hip[0] + 0.3 * np.sin(lean_rad), left_hip[1] - 0.3 * np.cos(lean_rad))

    frame.landmarks["left_hip"] = Landmark(*left_hip, visibility=vis)
    frame.landmarks["right_hip"] = Landmark(*right_hip, visibility=vis)
    frame.landmarks["left_knee"] = Landmark(*left_knee, visibility=vis)
    frame.landmarks["right_knee"] = Landmark(*right_knee, visibility=vis)
    frame.landmarks["left_ankle"] = Landmark(*ankle, visibility=vis)
    frame.landmarks["left_shoulder"] = Landmark(*shoulder, visibility=vis)
    return frame


def ramp(a, b, steps):
    return [a + (b - a) * i / (steps - 1) for i in range(steps)]


def one_squat_frames():
    return ramp(170, 90, 15) + [90] * 5 + ramp(90, 170, 18) + [170] * 5


def test_auto_side_resolves_to_visible_leg():
    analyzer = SquatAnalyzer(side="auto")
    out = analyzer.process(make_frame(170, side="left", t=0.0), now=0.0)
    assert out["side"] == "left"


def test_three_clean_squats_counted():
    analyzer = SquatAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    completed = 0
    last_out = None
    for _ in range(3):
        for a in one_squat_frames():
            t += dt
            out = analyzer.process(make_frame(a, t=t), now=t)
            last_out = out
            if out["just_completed_rep"] is not None:
                completed += 1
    assert completed == 3
    assert last_out["rep_count"] == 3
    assert last_out["rom"] > 60


def test_phase_shows_descending_and_rising_directionally():
    analyzer = SquatAnalyzer(side="left")
    t = 0.0
    dt = 0.05
    seen_descending = False
    seen_rising = False
    for a in ramp(170, 90, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "DESCENDING":
            seen_descending = True
    for a in ramp(90, 170, 15):
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["phase"] == "RISING":
            seen_rising = True
    assert seen_descending
    assert seen_rising


def test_excessive_lean_is_flagged():
    analyzer = SquatAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    saw_flag = False
    for a in ramp(170, 90, 15) + [90] * 5:
        t += dt
        out = analyzer.process(make_frame(a, lean_deg=45.0, t=t), now=t)
        if any(f["level"] == "warn" and "lean" in f["message"].lower() for f in out["form_flags"]):
            saw_flag = True
    assert saw_flag


def test_upright_squat_does_not_flag_lean():
    analyzer = SquatAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    saw_flag = False
    for a in ramp(170, 90, 15) + [90] * 5:
        t += dt
        out = analyzer.process(make_frame(a, lean_deg=2.0, t=t), now=t)
        if any(f["level"] == "warn" and "lean" in f["message"].lower() for f in out["form_flags"]):
            saw_flag = True
    assert not saw_flag


def test_knee_valgus_ratio_direct_aligned():
    """Unit-level check of the metric itself: knees exactly as far apart
    as hips -> ratio of 1.0, well above the flag threshold."""
    analyzer = SquatAnalyzer(side="left")
    frame = make_frame(120, knee_width_ratio=1.0)
    ratio = analyzer._knee_valgus_ratio(frame)
    assert ratio is not None
    assert abs(ratio - 1.0) < 1e-6


def test_knee_valgus_ratio_direct_caved_in():
    analyzer = SquatAnalyzer(side="left")
    frame = make_frame(120, knee_width_ratio=0.4)
    ratio = analyzer._knee_valgus_ratio(frame)
    assert ratio is not None
    assert abs(ratio - 0.4) < 1e-6


def test_knee_valgus_flagged_for_large_inward_deviation():
    analyzer = SquatAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    saw_flag = False
    for a in ramp(170, 90, 20):
        t += dt
        out = analyzer.process(make_frame(a, knee_width_ratio=0.4, t=t), now=t)
        if any(f["level"] == "warn" and "valgus" in f["message"].lower() for f in out["form_flags"]):
            saw_flag = True
    assert saw_flag


def test_normal_knee_width_does_not_flag_valgus():
    """Normal squat: knees track roughly at or outside hip width
    throughout the descent — even though the LEFT leg's own 2D projection
    inherently shifts as it bends (see squat.py's docstring on why the
    metric compares both knees to each other instead of one knee to its
    own hip-ankle line), the bilateral ratio should stay close to 1 and
    never falsely flag."""
    analyzer = SquatAnalyzer(side="left")
    t = 0.0
    dt = 0.03
    saw_flag = False
    for a in ramp(170, 90, 20):
        t += dt
        out = analyzer.process(make_frame(a, knee_width_ratio=1.0, t=t), now=t)
        if any(f["level"] == "warn" and "valgus" in f["message"].lower() for f in out["form_flags"]):
            saw_flag = True
    assert not saw_flag


def test_partial_depth_flagged_on_completed_rep():
    analyzer = SquatAnalyzer(side="left")
    analyzer.t.partial_rom_threshold = 200  # force everything to look partial
    t = 0.0
    dt = 0.03
    flagged = False
    for a in one_squat_frames():
        t += dt
        out = analyzer.process(make_frame(a, t=t), now=t)
        if out["just_completed_rep"] is not None:
            flagged = any(f["level"] == "warn" and "depth" in f["message"].lower() for f in out["form_flags"])
    assert flagged


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
