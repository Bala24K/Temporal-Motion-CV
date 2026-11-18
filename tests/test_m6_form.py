"""Milestone 6 tests: shoulder press + lateral raise v2, explainable warnings, rule validation status and the
form-validation harness (docs/audit/REBUILD_ROADMAP.md "M6", TEST_AND_EVALUATION_PLAN.md section 6).

Everything here is synthetic. The real targets (warning precision >= 0.8, <= 5 false warnings per 100 clean reps)
need expert-labelled reps; `python -m ml.form_eval` computes them, and these tests check that it computes them right.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from exercise_system.registry import ExerciseRegistry
from ml import form_eval as FE
from ml.replay import replay
from motion_engine import synthetic as S
from motion_engine.form import evaluate_rep, form_score
from motion_engine.session import LiveSession

REG = ExerciseRegistry.load()
ASPECTS = {"4:3": (640, 480), "16:9": (1280, 720), "3:4": (480, 640), "9:16": (720, 1280)}
NEW = ["shoulder_press", "lateral_raise"]


def camera(frame, w, h):
    k = h / w
    out = {n: (0.5 + (v[0] - 0.5) * k, v[1], v[2]) for n, v in frame.items() if not n.startswith("_")}
    out["_image"] = [w, h]
    return out


def run(ex, frames, ts, defn=None):
    s = LiveSession(defn or REG.get(ex))
    for f, t in zip(frames, ts):
        s.push(f if "_image" in f else {**f, "_image": [1, 1]}, t)
    s.finish()
    return s


def warnings(s):
    return Counter(e["rule_id"] for r in s.reps for e in r["evidence"] if e.get("evaluated", True) and not e["passed"])


# ---------------------------------------------------------------- definitions v2
@pytest.mark.parametrize("ex", NEW)
def test_new_definitions_are_v2(ex):
    d = REG.get(ex)
    assert d.version == 2 and d.rep.population_range and d.limitations
    assert all(r.validation.status == "provisional" for r in d.form_rules)   # nothing validated without expert data


@pytest.mark.parametrize("ex", NEW)
@pytest.mark.parametrize("label", list(ASPECTS))
def test_clean_set_gives_no_warnings_at_any_aspect(ex, label):
    w, h = ASPECTS[label]
    f, t = S.synth_sequence(ex, reps=5, seed=4, scale=0.75 if h > w else 1.0)   # portrait users stand further back
    s = run(ex, [camera(x, w, h) for x in f], t)
    assert s.counter.count == 5 and not warnings(s), (s.counter.count, warnings(s))


@pytest.mark.parametrize("ex", NEW)
@pytest.mark.parametrize("yaw", [-30, 30])
def test_clean_set_gives_no_warnings_when_turned_slightly(ex, yaw):
    f, t = S.synth_sequence(ex, reps=5, seed=5)
    s = run(ex, [S.yaw_project(x, yaw) for x in f], t)
    assert s.counter.count == 5 and not warnings(s), warnings(s)


FAULTS = [("shoulder_press", "lean", "torso_upright"), ("shoulder_press", "fast", "tempo"),
          ("shoulder_press", "no_lockout", "lockout"), ("shoulder_press", "asym", "arm_sync"),
          ("lateral_raise", "lean", "no_swing"), ("lateral_raise", "fast", "tempo"),
          ("lateral_raise", "bent_arms", "arms_long"), ("lateral_raise", "asym", "arm_sync")]


@pytest.mark.parametrize("ex,fault,rule", FAULTS)
def test_each_fault_triggers_its_rule(ex, fault, rule):
    s = run(ex, *S.synth_sequence(ex, reps=5, fault=fault, seed=6))
    assert s.counter.count == 5 and warnings(s)[rule] == 5, warnings(s)


def test_lockout_is_not_triggered_by_other_faults():
    for fault in ("lean", "fast", None):
        assert warnings(run("shoulder_press", *S.synth_sequence("shoulder_press", fault=fault, seed=7)))["lockout"] == 0


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_lowering_the_dumbbells_after_a_press_set_is_not_a_rep(fps):
    """Regression: v1 counted presses with the elbow angle, which reads the same for hanging and locked-out arms,
    so racking the dumbbells down at the end of a set counted one extra rep."""
    s = run("shoulder_press", *S.press_with_rack(reps=5, fps=fps))
    assert s.counter.count == 5 and s.counter.partials == 0


def test_press_without_lockout_is_still_counted_but_warned():
    s = run("shoulder_press", *S.press_with_rack(reps=4, fault="no_lockout"))
    assert s.counter.count == 4 and warnings(s)["lockout"] == 4


# ---------------------------------------------------------------- explainability
EXPLAIN_CASES = [(ex, fault) for ex in ["squat", "bicep_curl", "hammer_curl", *NEW]
                 for fault in (None, "shallow", "fast", "lean", "drift", "asym")]


@pytest.mark.parametrize("ex,fault", EXPLAIN_CASES)
def test_every_warning_says_what_was_measured_from_which_view_and_when(ex, fault):
    s = run(ex, *S.synth_sequence(ex, reps=3, fault=fault, seed=8))
    for r in s.reps:
        assert r["t_start"] < r["t_end"]
        for e in r["evidence"]:
            if not e.get("evaluated", True):
                continue
            need = ("signal", "window", "status") + (() if e["passed"] else ("measured", "threshold"))
            for k in need:
                assert e.get(k) is not None, (ex, fault, e["rule_id"], k)
            assert "view" in e
            w0, w1 = e["window"]
            assert r["t_start"] - 1e-6 <= w0 < w1 <= r["t_end"] + 1e-6
            if e["type"] != "tempo":
                assert w0 - 1e-6 <= e["t_worst"] <= w1 + 1e-6, (ex, e)


def test_warning_view_is_the_detected_camera_view():
    s = run("shoulder_press", *S.synth_sequence("shoulder_press", fault="lean", seed=9))
    assert {e["view"] for r in s.reps for e in r["evidence"]} == {"front"}


def test_lean_warning_points_at_the_most_leaned_frame():
    s = run("lateral_raise", *S.synth_sequence("lateral_raise", fault="lean", seed=10))
    r = s.reps[2]
    e = next(e for e in r["evidence"] if e["rule_id"] == "no_swing")
    peak_t = r["t_start"] + r["toward_peak_seconds"]           # lean grows with progress, so it is worst at the peak
    assert abs(e["t_worst"] - peak_t) < 0.25


# ---------------------------------------------------------------- validation status
def _with_status(ex, rule_id, status):
    d = REG.get(ex).model_copy(deep=True)
    for r in d.form_rules:
        if r.id == rule_id:
            r.validation.status = status
    return d


def test_disabled_rule_is_not_evaluated():
    d = _with_status("lateral_raise", "no_swing", "disabled")
    s = run("lateral_raise", *S.synth_sequence("lateral_raise", fault="lean", seed=11), defn=d)
    assert all(e["rule_id"] != "no_swing" for r in s.reps for e in r["evidence"])


def test_info_only_rule_is_a_tip_and_does_not_lower_the_score():
    frames, ts = S.synth_sequence("lateral_raise", fault="lean", seed=12)
    base = run("lateral_raise", frames, ts)
    d = _with_status("lateral_raise", "no_swing", "info_only")
    s = run("lateral_raise", frames, ts, defn=d)
    e = next(e for e in s.reps[0]["evidence"] if e["rule_id"] == "no_swing")
    assert not e["passed"] and e["severity"] == "info" and e["status"] == "info_only"
    assert s.reps[0]["form_score"] > base.reps[0]["form_score"]
    assert {"status": "info", "message": e["message"], "rule": "no_swing"} in s.feedback


def test_api_exposes_validation_status_and_limitations(client):
    d = client.get("/api/exercises/shoulder_press").json()
    assert {c["id"] for c in d["check_validation"]} == {"rom", "lockout", "tempo", "torso_upright", "arm_sync"}
    assert all(c["status"] == "provisional" for c in d["check_validation"])
    assert any("from the front" in x for x in d["limitations"])


# ---------------------------------------------------------------- the harness
def test_score_computes_precision_recall_and_false_warnings_per_100_clean():
    obs = ([{"labels": {"lean": "fail"}, "engine": {"lean": True}}] * 8          # TP
           + [{"labels": {"lean": "fail"}, "engine": {"lean": False}}] * 2       # FN
           + [{"labels": {"lean": "pass"}, "engine": {"lean": True}}] * 2        # FP on clean reps
           + [{"labels": {"lean": "pass"}, "engine": {"lean": False}}] * 38      # TN
           + [{"labels": {"lean": "na"}, "engine": {"lean": True}}] * 5          # not judgeable: ignored
           + [{"labels": {"lean": "pass"}, "engine": {"lean": None}}] * 3)       # engine couldn't see it
    m = FE.score(obs, ["lean"], min_reps=30, min_faulty=9, min_clean=30)["lean"]
    assert (m["tp"], m["fn"], m["fp"], m["tn"], m["not_evaluated"]) == (8, 2, 2, 38, 3)
    assert m["precision"] == 0.8 and m["recall"] == 0.8
    assert m["false_per_100_clean"] == 5.0            # 2 of 40 judged clean reps
    assert m["recommendation"] == "validated"


@pytest.mark.parametrize("fp,expected", [(2, "validated"), (5, "info_only"), (12, "disabled")])
def test_recommendation_follows_the_targets(fp, expected):
    obs = ([{"labels": {"r": "fail"}, "engine": {"r": True}}] * 12
           + [{"labels": {"r": "pass"}, "engine": {"r": True}}] * fp
           + [{"labels": {"r": "pass"}, "engine": {"r": False}}] * (60 - fp))
    assert FE.score(obs, ["r"], 30, 9, 30)["r"]["recommendation"] == expected


def test_too_little_data_changes_nothing():
    obs = [{"labels": {"r": "fail"}, "engine": {"r": True}}] * 5
    assert FE.score(obs, ["r"], 30, 9, 30)["r"]["recommendation"] == "insufficient"


def test_reviewer_disagreement_is_excluded_and_kappa_reported():
    a = {"reviewer": "a", "reps": [{"t0": i, "t1": i + 0.8, "rules": {"r": "fail" if i % 2 else "pass"}} for i in range(10)]}
    b = {"reviewer": "b", "reps": [{"t0": i + 0.05, "t1": i + 0.85, "rules": {"r": "fail" if i % 2 or i == 4 else "pass"}}
                                   for i in range(10)]}
    reps, pairs = FE.consensus([a, b])
    assert reps[4]["rules"]["r"] == "na" and reps[3]["rules"]["r"] == "fail" and reps[2]["rules"]["r"] == "pass"
    assert FE.cohen_kappa(pairs["r"]) == pytest.approx(0.8, abs=0.01)


def _write_set(dirpath: Path, name, ex, fault, labels_for, seed, reviewers=("a",)):
    frames, ts = S.synth_sequence(ex, reps=5, fault=fault, seed=seed)
    rec = dirpath / f"{name}.jsonl"
    rows = [{"type": "meta", "exercise": ex}] + [{"type": "frame", "t": t, "landmarks": {**{k: list(v) for k, v in f.items()},
                                                                                          "_image": [1, 1]}}
                                                 for f, t in zip(frames, ts)]
    rec.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    reps = replay([(t, r["landmarks"]) for t, r in zip(ts, rows[1:])], REG.get(ex))["summary"]["reps_detail"]
    for rv in reviewers:
        lab = {"exercise": ex, "reviewer": rv,
               "reps": [{"t0": r["t_start"] - 0.1, "t1": r["t_end"] + 0.1, "rules": labels_for(i)} for i, r in enumerate(reps)]}
        (dirpath / f"{name}.labels.{rv}.json").write_text(json.dumps(lab), encoding="utf-8")


def test_harness_end_to_end_and_apply(tmp_path):
    data, defs = tmp_path / "data", tmp_path / "defs"
    data.mkdir()
    shutil.copytree(FE.DEFS_DIR, defs)
    ok = {"rom": "pass", "lockout": "pass", "tempo": "pass", "torso_upright": "pass", "arm_sync": "pass"}
    for k in range(8):           # 40 clean reps
        _write_set(data, f"clean{k}", "shoulder_press", None, lambda i: dict(ok), seed=20 + k, reviewers=("a", "b"))
    for k in range(3):           # 15 leaning reps, expert agrees
        _write_set(data, f"lean{k}", "shoulder_press", "lean", lambda i: {**ok, "torso_upright": "fail"}, seed=40 + k)
    # 10 reps the expert calls clean but that have no lockout: the engine warns -> false warnings for `lockout`
    for k in range(2):
        _write_set(data, f"nolock{k}", "shoulder_press", "no_lockout", lambda i: dict(ok), seed=60 + k)
    rep = FE.evaluate(data, "shoulder_press")["shoulder_press"]
    assert rep["recordings"] == 13 and rep["missed_reps"] == 0 and rep["extra_reps"] == 0
    up = rep["rules"]["torso_upright"]
    assert (up["tp"], up["fp"], up["fn"]) == (15, 0, 0) and up["recommendation"] == "validated"
    lk = rep["rules"]["lockout"]
    assert lk["fp"] == 10 and lk["false_per_100_clean"] == pytest.approx(100 * 10 / 50, abs=0.01)
    assert lk["recommendation"] == "insufficient"            # no labelled lockout faults -> can't judge precision
    assert rep["rules"]["arm_sync"]["kappa"] is None or rep["rules"]["arm_sync"]["kappa"] >= 0.99

    changed = FE.apply({"shoulder_press": rep}, "synthetic-test", defs)
    assert changed == ["shoulder_press.torso_upright"]
    d = ExerciseRegistry.load(defs).get("shoulder_press")
    v = {r.id: r.validation for r in d.form_rules}
    assert v["torso_upright"].status == "validated" and v["torso_upright"].precision == 1.0
    assert v["lockout"].status == "provisional"


def test_replay_closes_the_last_rep_like_a_live_set():
    frames = [S._pose(elbow_int=S._lerp(125, 40, p), arm_elev=8, bend=-1, lean=3)
              for p in [0.0] * 18 + list(S.smoothstep(1 - abs(2 * (__import__("numpy").arange(72) / 72) - 1)))]
    res = replay([(i / 30, {**f, "_image": [1, 1]}) for i, f in enumerate(frames)], REG.get("bicep_curl"))
    assert res["summary"]["reps"] == 1
