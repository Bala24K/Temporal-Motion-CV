import numpy as np
import pytest

from exercise_system.registry import ExerciseRegistry, validate_definition
from exercise_system.models import ExerciseDefinition
from motion_engine.form import evaluate_rep, form_score
from motion_engine.personalization import Baseline
from motion_engine.primitives import compute_signal, validate_spec
from motion_engine.quality import check_frame
from motion_engine.recognition import recognize
from motion_engine.repetition import RepCounter
from motion_engine.session import LiveSession, analyze_sequence
from motion_engine.skeleton import frames_to_array, measure_body, normalize, fill_gaps
from motion_engine.synthetic import EX, synth_sequence, synth_standing

REG = ExerciseRegistry.load()
ALL = sorted(EX)


def fails(summary):
    return {e["rule_id"] for r in summary["reps_detail"] for e in r["evidence"] if not e["passed"]}


@pytest.mark.parametrize("ex", ALL)
def test_clean_reps_counted_and_pass(ex):
    f, t = synth_sequence(ex, reps=6)
    s = analyze_sequence(REG.get(ex), f, t)
    assert s["reps"] == 6 and s["partial_reps"] == 0
    assert s["avg_form_score"] == 100.0, fails(s)


@pytest.mark.parametrize("ex", ALL)
def test_shallow_reps_are_not_counted(ex):
    f, t = synth_sequence(ex, reps=4, fault="shallow")
    s = analyze_sequence(REG.get(ex), f, t)
    assert s["reps"] == 0 and s["partial_reps"] == 4


@pytest.mark.parametrize("ex", ALL)
def test_fast_reps_counted_but_tempo_flagged(ex):
    f, t = synth_sequence(ex, reps=4, fault="fast")
    s = analyze_sequence(REG.get(ex), f, t)
    assert s["reps"] == 4
    assert "tempo" in fails(s)


@pytest.mark.parametrize("ex,fault,rule", [
    ("bicep_curl", "drift", "elbow_pinned"), ("bicep_curl", "lean", "torso_steady"),
    ("squat", "lean", "chest_up"), ("lateral_raise", "lean", "no_swing"),
    ("shoulder_press", "lean", "no_lean_back"), ("overhead_triceps_extension", "lean", "no_lean"),
])
def test_each_fault_trips_its_own_rule_only(ex, fault, rule):
    f, t = synth_sequence(ex, reps=4, fault=fault)
    s = analyze_sequence(REG.get(ex), f, t)
    assert s["reps"] == 4
    assert fails(s) == {rule}


def test_squat_and_curl_are_direction_agnostic():
    # squat counts eccentric-first; the same counter handles both (regression for the
    # 'counts zero squats' failure mode documented in research/motion_upgrade/MOTION.md)
    for ex in ("squat", "bicep_curl"):
        f, t = synth_sequence(ex, reps=3)
        assert analyze_sequence(REG.get(ex), f, t)["reps"] == 3


def test_phase_semantics_squat_descent_is_eccentric():
    d = REG.get("squat")
    sess = LiveSession(d)
    f, t = synth_sequence("squat", reps=1)
    phases = []
    for fr, ti in zip(f, t):
        phases.append(sess.push(fr, ti)["phase"])
    first_active = next(p for p in phases if p != "rest")
    assert first_active == "eccentric"
    assert "concentric" in phases


def test_asymmetry_detected_via_symmetry_rule():
    f, t = synth_sequence("bicep_curl", reps=4)
    for fr in f:  # left arm only curls about halfway: blend its forearm toward straight
        sx, sy, _ = fr["left_shoulder"]
        ex_, ey, _ = fr["left_elbow"]
        wx, wy, _ = fr["left_wrist"]
        straight = np.array([ex_ - sx, ey - sy]); straight = straight / np.linalg.norm(straight)
        actual = np.array([wx - ex_, wy - ey]); actual = actual / np.linalg.norm(actual)
        d = 0.5 * straight + 0.5 * actual; d = d / np.linalg.norm(d) * 0.12
        fr["left_wrist"] = (ex_ + d[0], ey + d[1], 1.0)
    s = analyze_sequence(REG.get("bicep_curl"), f, t)
    assert "arm_sync" in fails(s)


@pytest.mark.parametrize("ex", ALL)
def test_recognition_finds_right_group(ex):
    f, _ = synth_sequence(ex, reps=4)
    r = recognize(f, REG)
    assert r.top is not None and ex in r.top.members
    assert r.status in ("recognized", "ambiguous")
    if ex in ("squat", "goblet_squat", "shoulder_press", "overhead_triceps_extension", "romanian_deadlift"):
        assert r.status == "recognized" and r.top.exercise == ex


def test_indistinguishable_exercises_are_not_faked():
    for ex, twin in (("bicep_curl", "hammer_curl"), ("lateral_raise", "front_raise")):
        r = recognize(synth_sequence(ex, reps=4)[0], REG)
        assert r.action == "confirm" and set(r.top.members) == {ex, twin}


def test_standing_still_is_no_movement():
    f, _ = synth_standing(90)
    r = recognize(f, REG)
    assert r.status == "no_movement" and r.action == "wait"


def test_unknown_movement_is_not_forced_into_a_label():
    # wave one arm sideways at the elbow only - nothing in the library describes this
    f, _ = synth_sequence("bicep_curl", reps=4)
    g = []
    for i, fr in enumerate(f):
        fr = dict(fr)
        a = 0.5 + 0.5 * np.sin(i / 6)
        x, y, v = fr["right_wrist"]
        fr["right_wrist"] = (x + 0.15 * a, y - 0.1 * a, v)
        fr["left_wrist"] = fr["left_elbow"][:2] + (1.0,)
        fr["left_wrist"] = (fr["left_shoulder"][0], fr["left_shoulder"][1] + 0.25, 1.0)
        fr["left_elbow"] = (fr["left_shoulder"][0], fr["left_shoulder"][1] + 0.12, 1.0)
        fr["right_elbow"] = (fr["right_shoulder"][0] + 0.02, fr["right_shoulder"][1] + 0.13, 1.0)
        g.append(fr)
    r = recognize(g, REG)
    assert r.action != "analyze"


def test_body_calibration_is_scale_invariant():
    f1, _ = synth_standing(45, scale=1.0)
    f2, _ = synth_standing(45, scale=0.6)
    p1, p2 = measure_body(f1, 175), measure_body(f2, 175)
    assert p1.calibrated and p2.calibrated
    assert p1.ratios() == pytest.approx(p2.ratios(), rel=0.03)
    assert p2.torso_len < p1.torso_len


def test_angles_unchanged_by_person_size_or_position():
    a = analyze_sequence(REG.get("squat"), *synth_sequence("squat", reps=3, scale=1.0))
    b = analyze_sequence(REG.get("squat"), *synth_sequence("squat", reps=3, scale=0.7))
    assert a["reps"] == b["reps"] == 3
    assert a["avg_rom"] == pytest.approx(b["avg_rom"], abs=0.03)


def test_quality_asks_to_reposition():
    f, _ = synth_sequence("squat", reps=1)
    fr = {k: v for k, v in f[0].items() if "ankle" not in k and "knee" not in k}
    q = check_frame(fr, REG.get("squat").required_joints)
    assert not q["ok"] and "whole body" in q["message"]
    s = LiveSession(REG.get("squat"))
    out = None
    for _ in range(20):
        out = s.push(fr, 0.0)
    assert out["quality"]["ok"] is False and s.counter.count == 0


def test_dropped_frames_tolerated():
    f, t = synth_sequence("bicep_curl", reps=4)
    rng = np.random.default_rng(3)
    arr = frames_to_array(f)
    assert fill_gaps(arr[:, :, :2]).shape == arr[:, :, :2].shape
    g = [fr for i, fr in enumerate(f) if rng.random() > 0.15]
    tt = [ti for i, ti in enumerate(t) if True][: len(g)]
    tt = sorted(rng.choice(t, len(g), replace=False))
    assert analyze_sequence(REG.get("bicep_curl"), g, tt)["reps"] == 4


def test_baseline_drives_personal_insight_not_pass_fail():
    d = REG.get("squat")
    b = Baseline()
    for _ in range(10):
        b.update({"counted": True, "peak_progress": 1.2, "duration": 2.4, "form_score": 100})
    f, t = synth_sequence("squat", reps=3, depth=0.88)
    s = LiveSession(d, baseline=b)
    for fr, ti in zip(f, t):
        s.push(fr, ti)
    assert any("below your usual" in m for r in s.reps for m in r["insights"])
    assert all(r["form_score"] == 100.0 for r in s.reps)        # objective result unchanged


def test_definitions_are_validated():
    bad = REG.get("squat").model_dump()
    bad["rep"]["signal"] = {"primitive": "nope"}
    with pytest.raises(ValueError):
        ExerciseRegistry().register(ExerciseDefinition.model_validate(bad))
    bad = REG.get("squat").model_dump()
    bad["form_rules"].append({"id": "x", "type": "limit", "signal": {"primitive": "joint_angle", "joints": ["a"]}})
    assert validate_definition(ExerciseDefinition.model_validate(bad))


def test_new_exercise_is_config_only():
    """Adding an exercise = a data file. Here: an 'incline-style' curl variant with no new code."""
    d = REG.get("bicep_curl").model_dump()
    d.update(id="zottman_curl", name="Zottman Curl", aliases=["zottman"])
    reg = ExerciseRegistry(list(REG.all()))
    reg.register(ExerciseDefinition.model_validate(d))
    f, t = synth_sequence("bicep_curl", reps=3)
    assert analyze_sequence(reg.get("zottman_curl"), f, t)["reps"] == 3
    assert reg.resolve("zottman")[0].id == "zottman_curl"


def test_registry_resolves_names_and_flags_unknowns():
    assert REG.resolve("Overhead Press")[0].id == "shoulder_press"
    assert REG.resolve("rdl")[0].id == "romanian_deadlift"
    assert REG.resolve("cable y raise")[0] is None
    assert REG.suggest_pattern("Cable Y Raise") == "raise"
    assert REG.suggest_pattern("Turkish Get-Up") is None
