"""Milestone 1 acceptance tests (docs/REBUILD_ROADMAP.md, "M1 - Trustworthy input").

Each test names the acceptance criterion it locks in. The defects they guard against were reproduced in
docs/audit-evidence/verify_rt.py and pose_study.py.
"""
import asyncio
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from exercise_system.registry import ExerciseRegistry
from ml.replay import replay
from motion_engine import synthetic as S
from motion_engine.calibration import calibrate, check_stage_frame
from motion_engine.contract import frame_error
from motion_engine.form import signal_table
from motion_engine.primitives import compute_signal
from motion_engine.runtime import LatestSlot
from motion_engine.session import LiveSession
from motion_engine.skeleton import frame_to_array, frames_to_array, normalize

REG = ExerciseRegistry.load()
FIXTURE = Path(__file__).parent / "fixtures" / "qevd_front_frames.json"
ASPECTS = {"4:3": (640, 480), "16:9": (1280, 720), "3:4": (480, 640), "9:16": (720, 1280)}
ISO = {"_image": [1, 1]}


def camera(frame, w, h):
    """What a pose model returns for an isotropic pose filmed by a w x h camera: x as a fraction of WIDTH."""
    k = h / w
    out = {n: (0.5 + (v[0] - 0.5) * k, v[1], v[2]) for n, v in frame.items() if not n.startswith("_")}
    out["_image"] = [w, h]
    return out


def elbow_frame(true_deg):
    a = math.radians(180 - true_deg)
    sh, el = (0.5, 0.30), (0.5, 0.45)
    wr = (el[0] + 0.12 * math.sin(a), el[1] + 0.12 * math.cos(a))
    return {"left_shoulder": (*sh, 1.0), "left_elbow": (*el, 1.0), "left_wrist": (*wr, 1.0)}


# ---------------------------------------------------------------- A1, A4: geometry is aspect-invariant
@pytest.mark.parametrize("ex", sorted(S.EX))
def test_A1_every_definition_signal_is_identical_at_every_camera_aspect(ex):
    frames, _ = S.synth_sequence(ex, reps=1, noise=0.0)
    defn = REG.get(ex)
    for name, spec in signal_table(defn).items():
        ref = compute_signal(normalize(frames_to_array(frames)), spec)
        for label, (w, h) in ASPECTS.items():
            got = compute_signal(normalize(frames_to_array([camera(f, w, h) for f in frames])), spec)
            assert np.allclose(got, ref, atol=0.5, equal_nan=True), (ex, name, label)


@pytest.mark.parametrize("label", list(ASPECTS))
def test_A4_a_45_degree_elbow_reads_45_degrees(label):
    w, h = ASPECTS[label]
    arr = frame_to_array(camera(elbow_frame(45), w, h))[None]
    ang = compute_signal(arr, {"primitive": "joint_angle", "joints": ["shoulder", "elbow", "wrist"], "side": "left"})[0]
    assert ang == pytest.approx(45.0, abs=0.5)        # was 29.4 deg at 16:9 before the contract fix


def test_A4_without_image_metadata_the_distortion_is_what_the_audit_measured():
    """Documents the bug the contract prevents: per-axis normalised coords read as isotropic."""
    f = camera(elbow_frame(45), 1280, 720)
    del f["_image"]
    ang = compute_signal(frame_to_array(f)[None], {"primitive": "joint_angle", "joints": ["shoulder", "elbow", "wrist"], "side": "left"})[0]
    assert ang == pytest.approx(29.4, abs=0.5)


# ---------------------------------------------------------------- A2: calibration works at every aspect
@pytest.mark.parametrize("label", list(ASPECTS))
def test_A2_synthetic_body_passes_front_calibration_at_every_aspect(label):
    w, h = ASPECTS[label]
    front = [camera(f, w, h) for f in S.synth_standing(30)[0]]
    assert check_stage_frame("front", front[0])["ok"], check_stage_frame("front", front[0])
    prof, _ = calibrate({"front": front})
    assert prof.calibrated                            # front-only used to end up "not calibrated" (quality 0.50)
    assert prof.shoulder_ratio == pytest.approx(0.59, abs=0.02)


# ---------------------------------------------------------------- A3: real footage
def _fixture():
    if not FIXTURE.exists():
        pytest.skip("QEVD landmark fixture missing")
    return json.loads(FIXTURE.read_text())["frames"]


@pytest.mark.parametrize("res", ["640x480", "640x360"])
def test_A3_real_people_facing_the_camera_pass_the_front_gate(res):
    frames = [r["landmarks"] for r in _fixture()[res]]
    assert len(frames) >= 50
    rate = np.mean([check_stage_frame("front", f)["ok"] for f in frames])
    assert rate >= 0.90, rate                         # audit: 0% (4:3) and 25% (16:9) before the fix


def test_A3_real_frames_without_image_metadata_mostly_fail():
    """At 16:9 the per-axis distortion (x 0.5625) pushes real facing people far below the gate."""
    frames = [{k: v for k, v in r["landmarks"].items() if k != "_image"} for r in _fixture()["640x360"]]
    assert np.mean([check_stage_frame("front", f)["ok"] for f in frames]) < 0.5


# ---------------------------------------------------------------- A5: control messages are never dropped
@pytest.mark.parametrize("seed", range(10))
def test_A5_control_messages_survive_any_interleaving_with_frames(seed):
    rng = random.Random(seed)

    async def run():
        slot, got = LatestSlot(), []

        async def consumer():
            while (m := await slot.get()) is not None:
                got.append(m)
                if rng.random() < 0.5:
                    await asyncio.sleep(0)

        task = asyncio.create_task(consumer())
        sent = []
        for i in range(100):                          # 10 seeds x 100 messages = 1,000 interleavings
            if rng.random() < 0.15:
                slot.put_control({"type": "select_exercise", "n": i}); sent.append(i)
            else:
                slot.put({"type": "frame", "n": i})
            if rng.random() < 0.3:
                await asyncio.sleep(0)
        slot.close()
        await task
        return sent, [m["n"] for m in got if m["type"] != "frame"]

    sent, received = asyncio.run(run())
    assert received == sent                           # every control message, exactly once, in order


def test_A5_frames_remain_latest_wins():
    async def run():
        s = LatestSlot()
        for i in range(5):
            s.put({"type": "frame", "n": i})
        s.close()
        return await s.get(), s.dropped
    item, dropped = asyncio.run(run())
    assert item["n"] == 4 and dropped == 4


# ---------------------------------------------------------------- A10: contract enforced at the boundary
def test_A10_contract_validator():
    good = {**{k: list(v) for k, v in S.synth_standing(1)[0][0].items()}, "_image": [1280, 720]}
    assert frame_error(good) is None
    assert "_image" in frame_error({k: v for k, v in good.items() if k != "_image"})
    assert frame_error({**good, "_image": [0, 720]})
    assert frame_error({**good, "left_knee": [float("nan"), 0.5, 1.0]})


def test_A10_api_rejects_frames_without_image_size(client, auth):
    raw = {k: list(v) for k, v in S.synth_standing(1)[0][0].items()}
    assert client.post("/api/profile/calibrate/stage-check", json={"stage": "front", "frame": raw}, headers=auth).status_code == 422
    assert client.post("/api/profile/calibrate/staged", json={"stages": {"front": [raw] * 10}}, headers=auth).status_code == 422
    frames, ts = S.synth_sequence("squat", reps=1)
    body = {"exercise": "squat", "frames": [{k: list(v) for k, v in f.items()} for f in frames], "timestamps": ts}
    assert client.post("/api/analyze", json=body, headers=auth).status_code == 422
    with client.websocket_connect(f"/ws/live?exercise=squat&token={auth['Authorization'][7:]}") as ws:
        ws.send_json({"type": "frame", "t": 0.0, "landmarks": raw})
        m = ws.receive_json()
        assert m["type"] == "error" and m["code"] == "frame_contract"


# ---------------------------------------------------------------- A8: one truth about "calibrated"
def _stages():
    front = S.synth_standing(30)[0]
    return {"front": front, "side": [S.yaw_project(f, 85) for f in front],
            "arms_up": [dict(S._pose(arm_elev=170, elbow_int=175)) for _ in range(30)]}


@pytest.mark.parametrize("which", [("front",), ("front", "side"), ("front", "side", "arms_up")])
def test_A8_calibration_response_matches_stored_profile(client, auth, which):
    st = _stages()
    body = {"stages": {k: [{**{j: list(v) for j, v in f.items()}, **ISO} for f in st[k]] for k in which}}
    r = client.post("/api/profile/calibrate/staged", json=body, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["calibrated"] == client.get("/api/profile", headers=auth).json()["calibrated"] is True


# ---------------------------------------------------------------- A9: replay reproduces the live session
def test_A9_replay_reproduces_live_payloads_exactly(client, auth):
    frames, ts = S.synth_sequence("squat", reps=3)
    rec = [(t, {**{k: list(v) for k, v in f.items()}, **ISO}) for f, t in zip(frames, ts)]
    live = []
    with client.websocket_connect(f"/ws/live?exercise=squat&token={auth['Authorization'][7:]}") as ws:
        for t, lm in rec:                             # lock-step, so the server never drops a frame
            ws.send_json({"type": "frame", "t": t, "landmarks": lm})
            m = ws.receive_json()
            while m["type"] != "state":
                m = ws.receive_json()
            live.append({k: v for k, v in m.items() if k not in ("type", "dropped_frames")})
    off = replay(rec, REG.get("squat"), target_reps=10)["payloads"]
    assert json.loads(json.dumps(off)) == live and live[-1]["reps"] == 3


# ---------------------------------------------------------------- switching never loses counted work
def test_switching_exercise_saves_the_set_in_progress(client, auth):
    sid = client.post("/api/sessions", json={"name": "t"}, headers=auth).json()["id"]
    frames, ts = S.synth_sequence("bicep_curl", reps=3)
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&session_id={sid}&token={auth['Authorization'][7:]}") as ws:
        for f, t in zip(frames, ts):
            ws.send_json({"type": "frame", "t": t, "landmarks": {**{k: list(v) for k, v in f.items()}, **ISO}})
            while ws.receive_json()["type"] != "state":
                pass
        ws.send_json({"type": "select_exercise", "exercise": "squat", "weight_kg": 8})
        msgs = [ws.receive_json(), ws.receive_json()]
    saved = next(m for m in msgs if m["type"] == "set_saved")
    assert saved["saved"] and saved["reason"] == "switch" and saved["summary"]["reps"] == 3
    assert any(m["type"] == "exercise_selected" and m["exercise"] == "squat" for m in msgs)
    sets = client.get(f"/api/sessions/{sid}", headers=auth).json()["sets"]
    assert len(sets) == 1 and sets[0]["reps"] == 3 and sets[0]["weight_kg"] == 8


def test_demo_frames_satisfy_the_contract(client):
    d = client.get("/api/demo/sequence?exercise=squat&reps=1").json()
    assert all(frame_error(f) is None for f in d["frames"])
    assert all(frame_error(f) is None for f in client.get("/api/demo/standing?stage=side").json()["frames"])


def test_health_reports_ui_build_status(client):
    ui = client.get("/api/health").json()["ui"]
    assert "served" in ui


# ---------------------------------------------------------------- known counting defects (fixed in M3)
def _reps(ex, frames, ts):
    s = LiveSession(REG.get(ex))
    for f, t in zip(frames, ts):
        s.push(f, t)
    return s.counter.count


@pytest.mark.xfail(strict=True, reason="M3: strict return-to-rest rule misses incomplete re-extension (REP_COUNTING_AUDIT.md)")
@pytest.mark.parametrize("rest", [150, 140])
def test_M3_curl_with_incomplete_reextension_counts_every_rep(monkeypatch, rest):
    monkeypatch.setitem(S.EX, "bicep_curl", lambda p, f: dict(elbow_int=S._lerp(rest, 40, p), arm_elev=8, bend=-1, lean=3))
    f, t = S.synth_sequence("bicep_curl", reps=5)
    assert _reps("bicep_curl", f, t) == 5


@pytest.mark.xfail(strict=True, reason="M3: press bottoming out above the textbook 90 deg never completes a rep")
def test_M3_press_from_110_degrees_counts_every_rep(monkeypatch):
    monkeypatch.setitem(S.EX, "shoulder_press", lambda p, f: dict(elbow_int=S._lerp(110, 172, p), arm_elev=S._lerp(85, 168, p), bend=+1, lean=2))
    f, t = S.synth_sequence("shoulder_press", reps=5)
    assert _reps("shoulder_press", f, t) == 5
