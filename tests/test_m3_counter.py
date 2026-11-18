"""Milestone 3 acceptance tests: rep counter v2 + personal range (docs/REBUILD_ROADMAP.md "M3").

Synthetic scenarios only. The real-world targets (count MAE <= 0.3 reps/set on a human-counted golden set) need
recorded, double-annotated sets and are NOT claimed here; `python -m ml.replay` is the harness for them.
"""
import numpy as np
import pytest

from exercise_system.registry import ExerciseRegistry
from motion_engine import synthetic as S
from motion_engine.form import evaluate_rep
from motion_engine.personal_range import RangeCaptureError, capture_range, signal_key
from motion_engine.personalization import Baseline
from motion_engine.repetition import RepCounter, RepEvent
from motion_engine.session import LiveSession, analyze_sequence, rep_signal

REG = ExerciseRegistry.load()
ISO = {"_image": [1, 1]}


def run(ex, frames, ts, baseline=None):
    s = LiveSession(REG.get(ex), baseline=baseline)
    pays = [s.push(f, t) for f, t in zip(frames, ts)]
    s.finish()
    return s, pays


def curl_clip(rest=170, top=40, reps=5, pause=0.3, fps=30.0, rep_seconds=2.4, lead=0.6):
    """Curl that re-extends only to `rest` and pauses `pause` s between reps."""
    prog = [0.0] * int(lead * fps)
    n = int(rep_seconds * fps)
    for _ in range(reps):
        x = np.arange(n) / n
        prog += list(S.smoothstep(1 - np.abs(2 * x - 1)))
        prog += [0.0] * int(pause * fps)
    frames = [S._pose(elbow_int=S._lerp(rest, top, p), arm_elev=8, bend=-1, lean=3) for p in prog]
    return frames, [i / fps for i in range(len(frames))]


# ---------------------------------------------------------------- completion rules
@pytest.mark.parametrize("rest,how", [(150, "return"), (140, "return"), (125, "reversal")])
def test_incomplete_reextension_is_still_counted(rest, how):
    f, t = curl_clip(rest=rest, pause=0.0)
    s, _ = run("bicep_curl", f, t)
    assert s.counter.count == 5, (rest, s.counter.count, s.counter.partials)
    assert how in {r["completion"] for r in s.reps}


def test_pausing_part_way_down_between_reps_completes_by_settling():
    f, t = curl_clip(rest=120, pause=1.2)          # back to 120 deg only, hold 1.2 s, next rep
    s, _ = run("bicep_curl", f, t)
    assert s.counter.count == 5 and "settle" in {r["completion"] for r in s.reps}


def test_last_rep_that_never_fully_returns_is_closed_at_end_of_set():
    f, t = curl_clip(rest=125, pause=0.0)
    s = LiveSession(REG.get("bicep_curl"))
    for a, b in zip(f, t):
        s.push(a, b)
    before = s.counter.count
    s.finish()
    assert s.counter.count == 5 and before == 4 and s.reps[-1]["completion"] in ("flush", "settle")


@pytest.mark.parametrize("ex", ["squat", "bicep_curl", "shoulder_press", "lateral_raise"])
def test_frame_rate_does_not_change_the_count(ex):
    counts = []
    for fps in (15, 24, 30, 60):
        f, t = S.synth_sequence(ex, reps=5, fps=fps)
        counts.append(run(ex, f, t)[0].counter.count)
    assert counts == [5, 5, 5, 5], counts


def test_shallow_and_noise_are_not_counted():
    s, _ = run("squat", *S.synth_sequence("squat", reps=5, fault="shallow"))
    assert s.counter.count == 0 and s.counter.partials == 5
    f, t = S.synth_standing(300, noise=0.004)
    assert run("squat", f, t)[0].counter.count == 0


def test_long_hold_at_the_bottom_counts_once_with_its_full_duration():
    f, t = S.synth_sequence("squat", reps=1, rep_seconds=2.4)
    mid = len(f) // 2
    frames = f[:mid] + [f[mid]] * int(13 * 30) + f[mid:]
    ts = [i / 30 for i in range(len(frames))]
    s, _ = run("squat", frames, ts)
    assert s.counter.count == 1 and s.reps[0]["duration"] > 13.5        # descent included (v1 truncated it)


def test_a_rep_open_beyond_max_rep_seconds_is_abandoned_and_reported():
    c = RepCounter(170, 45, max_rep_seconds=5)
    for i in range(300):
        c.update(i / 30, {"rep": 100.0})                                  # stuck mid-range for 10 s
    assert c.count == 0 and c.abandoned >= 1


# ---------------------------------------------------------------- tracking states
def _with_gap(frames, ts, start, seconds, joints=("left_knee", "right_knee")):
    out = []
    for f, t in zip(frames, ts):
        if start <= t < start + seconds:
            f = {k: ((v[0], v[1], 0.0) if k in joints else v) for k, v in f.items()}
        out.append(f)
    return out


def test_short_tracking_gap_mid_rep_keeps_the_rep():
    f, t = S.synth_sequence("squat", reps=3)
    s, pays = run("squat", _with_gap(f, t, 1.4, 1.0), t)
    assert s.counter.count == 3 and "degraded" in {p["tracking"] for p in pays}


def test_long_tracking_gap_abandons_the_rep_and_says_so():
    f, t = S.synth_sequence("squat", reps=3)
    s, pays = run("squat", _with_gap(f, t, 1.4, 4.0), t)
    assert s.counter.abandoned >= 1 and s.counter.count < 3
    lost = [p for p in pays if p["tracking"] == "lost"]
    assert lost and lost[0]["quality"]["message"]


# ---------------------------------------------------------------- side-on: visible side
def test_side_on_squat_with_far_leg_hidden_is_counted():
    f, t = S.synth_sequence("squat", reps=5)
    hidden = [{k: ((v[0], v[1], 0.1) if k.startswith("left_") and any(j in k for j in ("hip", "knee", "ankle")) else v)
               for k, v in fr.items()} for fr in f]
    assert run("squat", hidden, t)[0].counter.count == 5


def test_far_leg_guesses_are_ignored_when_clearly_less_visible():
    f, t = S.synth_sequence("squat", reps=5)
    wrong = [{**fr, "left_knee": (fr["left_hip"][0], (fr["left_hip"][1] + fr["left_ankle"][1]) / 2, 0.55)} for fr in f]
    assert run("squat", wrong, t)[0].counter.count == 5       # straight "guessed" far leg would hide every rep in a mean


# ---------------------------------------------------------------- form rules
def _event(signals, peak=1.0):
    return RepEvent(1, True, 0, 1, 2, peak, 2.0, 1.0, 1.0, signals=signals)


def test_symmetry_is_not_judged_from_a_side_view():
    ev = evaluate_rep(REG.get("bicep_curl"), _event({"r:arm_sync:L": np.full(30, 90.0), "r:arm_sync:R": np.full(30, 120.0),
                                                     "r:elbow_pinned": np.full(30, 5.0), "r:torso_steady": np.full(30, 2.0)}), view="side")
    sym = next(e for e in ev if e.rule_id == "arm_sync")
    assert not sym.evaluated and sym.passed and "side" in sym.message


def test_rule_without_visible_data_reports_not_evaluated():
    ev = evaluate_rep(REG.get("bicep_curl"), _event({"r:elbow_pinned": np.full(30, np.nan), "r:torso_steady": np.full(30, 2.0)}))
    pin = next(e for e in ev if e.rule_id == "elbow_pinned")
    assert not pin.evaluated and pin.measured is None


def test_one_bad_frame_does_not_fail_a_limit_rule():
    lean = np.full(60, 4.0)
    lean[30] = 60.0                                               # single mis-tracked frame
    ev = evaluate_rep(REG.get("bicep_curl"), _event({"r:torso_steady": lean, "r:elbow_pinned": np.full(60, 5.0)}))
    assert next(e for e in ev if e.rule_id == "torso_steady").passed


def test_injected_faults_are_still_caught():
    for fault, rule in (("drift", "elbow_pinned"), ("lean", "torso_steady")):
        s, _ = run("bicep_curl", *S.synth_sequence("bicep_curl", reps=4, fault=fault))
        assert any(not e["passed"] and e["rule_id"] == rule for r in s.reps for e in r["evidence"]), fault


# ---------------------------------------------------------------- personal range
def test_front_view_curl_needs_personal_range_and_works_with_it():
    """Filmed straight on, the forearm comes toward the camera: the 2D elbow angle only reaches ~110 deg."""
    f, t = curl_clip(rest=168, top=110, pause=0.3)
    s0, _ = run("bicep_curl", f, t)
    assert s0.counter.count == 0                                  # population range: every rep looks partial
    rng = capture_range(REG.get("bicep_curl"), rep_signal(REG.get("bicep_curl"), f[:300]))
    assert rng["peak"] == pytest.approx(110, abs=6) and rng["cycles"] >= 2
    s1, pays = run("bicep_curl", f, t, baseline=Baseline({"range": rng}))
    assert s1.counter.count == 5 and pays[-1]["personal_range"]["peak"] == rng["peak"]


def test_range_capture_is_clamped_and_refuses_lazy_captures():
    d = REG.get("bicep_curl")
    f, _ = curl_clip(rest=170, top=140, reps=3)
    with pytest.raises(RangeCaptureError, match="full comfortable range"):
        capture_range(d, rep_signal(d, f))
    f, _ = curl_clip(rest=170, top=40, reps=1, lead=1.0)
    with pytest.raises(RangeCaptureError, match="at least 2"):
        capture_range(d, rep_signal(d, f))
    f, _ = curl_clip(rest=179, top=5, reps=3)
    r = capture_range(d, rep_signal(d, f))
    assert r["clamped"] and 25 <= r["peak"] <= 105


def test_range_for_a_changed_definition_is_ignored():
    s = LiveSession(REG.get("squat"), baseline=Baseline({"range": {"rest": 160, "peak": 100, "signal_key": "stale"}}))
    assert s.range_info is None
    s = LiveSession(REG.get("squat"), baseline=Baseline({"range": {"rest": 160, "peak": 100, "signal_key": signal_key(REG.get("squat"))}}))
    assert s.range_info == {"rest": 160, "peak": 100}


# ---------------------------------------------------------------- view guidance
def test_front_view_squat_is_told_to_turn_side_on():
    pays = run("squat", *S.synth_sequence("squat", reps=2))[1]
    assert pays[-1]["view"] == "front" and not pays[-1]["view_ok"] and "side-on" in pays[-1]["view_hint"]
    side = [S.with_world(fr, 80) for fr in S.synth_sequence("squat", reps=2)[0]]
    pays = run("squat", side, S.synth_sequence("squat", reps=2)[1])[1]
    assert pays[-1]["view"] == "side" and pays[-1]["view_ok"]


# ---------------------------------------------------------------- API / WebSocket
def _lm(f):
    return {**{k: list(v) for k, v in f.items()}, **ISO}


def test_live_range_capture_flow_applies_and_persists(client, auth):
    tok = auth["Authorization"][7:]
    f, t = curl_clip(rest=168, top=110, reps=3, pause=0.3)
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&token={tok}") as ws:
        ws.send_json({"type": "range_capture", "action": "start"})
        assert ws.receive_json()["status"] == "started"
        for fr, ti in zip(f, t):
            ws.send_json({"type": "frame", "t": ti, "landmarks": _lm(fr)})
            m = ws.receive_json()
            while m["type"] != "state":
                m = ws.receive_json()
        assert m["range_capture"] and m["phase"] == "capturing_range" and m["reps"] == 0
        ws.send_json({"type": "range_capture", "action": "finish"})
        r = ws.receive_json()
        assert r["type"] == "range_captured" and r["ok"] and r["peak"] == pytest.approx(110, abs=6)
        f2, t2 = curl_clip(rest=168, top=110, reps=4, pause=0.3)
        for fr, ti in zip(f2, t2):
            ws.send_json({"type": "frame", "t": 100 + ti, "landmarks": _lm(fr)})
            m = ws.receive_json()
            while m["type"] != "state":
                m = ws.receive_json()
        assert m["reps"] >= 3 and m["personal_range"]
    base = client.get("/api/profile", headers=auth).json()["baselines"]["bicep_curl"]
    assert base["range"]["peak"] == r["peak"]
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&token={tok}") as ws:   # next set: range applied
        ws.send_json({"type": "frame", "t": 0, "landmarks": _lm(f[0])})
        m = ws.receive_json()
        while m["type"] != "state":
            m = ws.receive_json()
        assert m["personal_range"]["peak"] == r["peak"]


def test_live_range_capture_reports_a_bad_capture(client, auth):
    with client.websocket_connect(f"/ws/live?exercise=squat&token={auth['Authorization'][7:]}") as ws:
        ws.send_json({"type": "range_capture", "action": "start"}); ws.receive_json()
        ws.send_json({"type": "range_capture", "action": "finish"})
        r = ws.receive_json()
        assert r["type"] == "range_captured" and not r["ok"] and r["message"]


def test_rest_range_endpoints(client, auth):
    f, _ = curl_clip(rest=168, top=110, reps=3)
    r = client.post("/api/profile/range", json={"exercise": "bicep_curl", "frames": [_lm(x) for x in f]}, headers=auth)
    assert r.status_code == 200 and r.json()["peak"] == pytest.approx(110, abs=6)
    assert client.get("/api/profile", headers=auth).json()["baselines"]["bicep_curl"]["range"]
    assert client.delete("/api/profile/range/bicep_curl", headers=auth).status_code == 204
    assert "range" not in client.get("/api/profile", headers=auth).json()["baselines"]["bicep_curl"]
    small, _ = curl_clip(rest=170, top=150, reps=3)
    assert client.post("/api/profile/range", json={"exercise": "bicep_curl", "frames": [_lm(x) for x in small]}, headers=auth).status_code == 422


def test_end_set_closes_the_last_partially_returned_rep(client, auth):
    sid = client.post("/api/sessions", json={"name": "t"}, headers=auth).json()["id"]
    f, t = curl_clip(rest=125, pause=0.0, reps=3)
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&session_id={sid}&token={auth['Authorization'][7:]}") as ws:
        for fr, ti in zip(f, t):
            ws.send_json({"type": "frame", "t": ti, "landmarks": _lm(fr)})
            while ws.receive_json()["type"] != "state":
                pass
        ws.send_json({"type": "end_set", "weight_kg": 10})
        m = ws.receive_json()
        while m["type"] != "set_saved":
            m = ws.receive_json()
    assert m["summary"]["reps"] == 3


def test_analyze_sequence_counts_incomplete_reextension():
    f, t = curl_clip(rest=140, pause=0.3)
    assert analyze_sequence(REG.get("bicep_curl"), f, t)["reps"] == 5
