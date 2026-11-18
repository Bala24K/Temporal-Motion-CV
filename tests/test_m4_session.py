"""Milestone 4 acceptance tests: reliable live sessions + product wiring (docs/REBUILD_ROADMAP.md "M4")."""
import uuid

import pytest

from backend.protocol import server_adapter
from backend.routers import live as live_mod
from backend.state import state
from motion_engine import synthetic as S

ISO = {"_image": [1, 1]}


def lm(f):
    return {**{k: list(v) for k, v in f.items()}, **ISO}


def tok(auth):
    return auth["Authorization"][7:]


def session(client, auth):
    return client.post("/api/sessions", json={"name": "t"}, headers=auth).json()["id"]


def stream(ws, frames, ts, t0=0.0, sink=None):
    """Lock-step: one frame, wait for its state (collect every message into `sink`)."""
    last = None
    for f, t in zip(frames, ts):
        ws.send_json({"type": "frame", "t": t0 + t, "landmarks": lm(f)})
        while True:
            m = ws.receive_json()
            if sink is not None:
                sink.append(m)
            if m["type"] == "state":
                last = m
                break
    return last


def connect(client, auth, **q):
    """Protocol v2: no token in the URL; auth in the first message; expect `ready`."""
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    ws = client.websocket_connect(f"/ws/live?{qs}")
    return ws


def curl(reps):
    return S.synth_sequence("bicep_curl", reps=reps)


# ---------------------------------------------------------------- auth in the first message
def test_auth_in_first_message_then_ready(client, auth):
    client.cookies.clear()                       # prove the first message alone authenticates
    with connect(client, auth, exercise="squat") as ws:
        ws.send_json({"type": "auth", "token": tok(auth)})
        r = ws.receive_json()
        assert r["type"] == "ready" and r["protocol"] == 2 and r["exercise"] == "squat" and not r["resumed"]
        f, t = S.synth_sequence("squat", reps=1)
        assert stream(ws, f[:5], t[:5])["type"] == "state"


def test_bad_or_missing_first_message_auth_is_refused(client, auth):
    client.cookies.clear()
    with connect(client, auth, exercise="squat") as ws:
        ws.send_json({"type": "auth", "token": "nope"})
        assert ws.receive_json() == {"type": "error", "code": "auth", "message": "Not authenticated"}
    with connect(client, auth, exercise="squat") as ws:
        ws.send_json({"type": "frame", "t": 0, "landmarks": {}})       # not an auth message
        assert ws.receive_json()["code"] == "auth"


def test_malformed_messages_get_an_error_and_the_set_continues(client, auth):
    with client.websocket_connect(f"/ws/live?exercise=squat&token={tok(auth)}") as ws:
        ws.send_json({"type": "range_capture", "action": "explode"})
        m = ws.receive_json()
        assert m["type"] == "error" and m["code"] == "bad_message"
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_message"
        f, t = S.synth_sequence("squat", reps=1)
        assert stream(ws, f[:3], t[:3])["type"] == "state"


# ---------------------------------------------------------------- resume after a dropped connection
def test_dropped_connection_resumes_with_every_counted_rep(client, auth):
    sid, key = session(client, auth), uuid.uuid4().hex
    f, t = curl(2)
    with connect(client, auth, exercise="bicep_curl", session_id=sid, set_key=key) as ws:
        ws.send_json({"type": "auth", "token": tok(auth)})
        assert ws.receive_json()["type"] == "ready"
        assert stream(ws, f, t)["reps"] == 2
    # socket gone without end_set -> parked; reconnect with the same key
    f2, t2 = curl(1)
    with connect(client, auth, exercise="bicep_curl", session_id=sid, set_key=key) as ws:
        ws.send_json({"type": "auth", "token": tok(auth)})
        r = ws.receive_json()
        assert r["type"] == "ready" and r["resumed"] and r["reps"] == 2
        assert stream(ws, f2, t2, t0=100)["reps"] == 3
        ws.send_json({"type": "end_set", "weight_kg": 12})
        m = ws.receive_json()
        while m["type"] != "set_saved":
            m = ws.receive_json()
    assert m["saved"] and m["summary"]["reps"] == 3
    sets = client.get(f"/api/sessions/{sid}", headers=auth).json()["sets"]
    assert len(sets) == 1 and sets[0]["reps"] == 3 and sets[0]["weight_kg"] == 12


def test_reconnect_takes_over_a_half_open_connection(client, auth):
    key = uuid.uuid4().hex
    f, t = curl(2)
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&set_key={key}&token={tok(auth)}") as a:
        assert a.receive_json()["type"] == "ready"
        stream(a, f, t)
        with client.websocket_connect(f"/ws/live?exercise=bicep_curl&set_key={key}&token={tok(auth)}") as b:
            r = b.receive_json()
            assert r["resumed"] and r["reps"] == 2                      # B owns the set now
            b.send_json({"type": "end_set"})
            m = b.receive_json()
            while m["type"] != "set_saved":
                m = b.receive_json()
            assert m["summary"]["reps"] == 2


@pytest.mark.parametrize("n", range(20))
def test_a_dropped_set_is_never_lost(client, auth, n):
    """Acceptance: kill the socket mid-set 20 times -> resumed or saved with every counted rep, every time."""
    key = uuid.uuid4().hex
    f, t = curl(1 + n % 3)
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&set_key={key}&token={tok(auth)}") as ws:
        ws.receive_json()
        reps = stream(ws, f, t)["reps"]
    r = client.post(f"/api/live/sets/{key}/end", json={"weight_kg": 5}, headers=auth)
    assert r.status_code == 200 and r.json()["summary"]["reps"] == reps == 1 + n % 3


def test_rest_end_of_parked_set(client, auth):
    sid, key = session(client, auth), uuid.uuid4().hex
    f, t = curl(2)
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&session_id={sid}&set_key={key}&token={tok(auth)}") as ws:
        ws.receive_json()
        stream(ws, f, t)
    r = client.post(f"/api/live/sets/{key}/end", json={"weight_kg": 8}, headers=auth).json()
    assert r["type"] == "set_saved" and r["saved"] and r["summary"]["reps"] == 2
    assert client.post(f"/api/live/sets/{key}/end", json={}, headers=auth).status_code == 404     # only once
    other = client.post("/api/auth/register", json={"email": "b@example.com", "password": "password123"}).json()["token"]
    key2 = uuid.uuid4().hex
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&set_key={key2}&token={tok(auth)}") as ws:
        ws.receive_json()
        stream(ws, f, t)
    assert client.post(f"/api/live/sets/{key2}/end", json={}, headers={"Authorization": f"Bearer {other}"}).status_code == 404


def test_parked_sets_expire(client, auth, monkeypatch):
    monkeypatch.setattr(live_mod, "PARK_SECONDS", -1.0)
    key = uuid.uuid4().hex
    with client.websocket_connect(f"/ws/live?exercise=squat&set_key={key}&token={tok(auth)}") as ws:
        ws.receive_json()
    assert client.post(f"/api/live/sets/{key}/end", json={}, headers=auth).status_code == 404


# ---------------------------------------------------------------- throttling, threads, empty sets
def test_state_messages_are_throttled_but_events_are_not(client, auth):
    f, t = curl(2)
    msgs = []
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&state_hz=2&token={tok(auth)}") as ws:
        for fr, ti in zip(f, t):
            ws.send_json({"type": "frame", "t": ti, "landmarks": lm(fr)})
        ws.send_json({"type": "end_set"})
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] == "set_saved":
                break
    states = [m for m in msgs if m["type"] == "state"]
    assert len(states) < len(f) / 2
    assert msgs[-1]["summary"]["reps"] == 2
    assert any(s["reps"] == 2 for s in states)                        # the rep event itself was sent


def test_recognition_runs_off_the_event_loop(client, auth, monkeypatch):
    seen = []
    real = state.recognizers.recognize

    def spy(frames):
        import asyncio
        try:
            asyncio.get_running_loop()
            seen.append("on-event-loop")          # would block every connection on this worker
        except RuntimeError:
            seen.append("worker-thread")
        return real(frames)

    monkeypatch.setattr(state.recognizers, "recognize", spy)
    f, t = S.synth_sequence("squat", reps=3)
    loop_thread = []
    with client.websocket_connect(f"/ws/live?exercise=auto&token={tok(auth)}") as ws:
        for fr, ti in zip(f[:90], t):
            ws.send_json({"type": "frame", "t": ti, "landmarks": lm(fr)})
        ws.send_json({"type": "end_set"})
        while ws.receive_json()["type"] != "set_saved":
            pass
    assert seen and set(seen) == {"worker-thread"}


def test_empty_ai_set_is_reported_not_silently_dropped(client, auth):
    sid = session(client, auth)
    f, t = S.synth_standing(60)
    with client.websocket_connect(f"/ws/live?exercise=squat&session_id={sid}&token={tok(auth)}") as ws:
        stream(ws, f, t)
        ws.send_json({"type": "end_set"})
        m = ws.receive_json()
        while m["type"] != "set_saved":
            m = ws.receive_json()
    assert m["empty"] and not m["saved"] and m["message"]


# ---------------------------------------------------------------- the protocol contract
def test_every_server_message_matches_the_protocol(client, auth):
    sid = session(client, auth)
    sink = []
    with connect(client, auth, exercise="auto", session_id=sid, set_key=uuid.uuid4().hex) as ws:
        ws.send_json({"type": "auth", "token": tok(auth)})
        sink.append(ws.receive_json())
        f, t = S.synth_sequence("squat", reps=3)
        for fr, ti in zip(f, t):                                      # auto: recognition messages, then states
            ws.send_json({"type": "frame", "t": ti, "landmarks": {**lm(fr), "_image": [640, 480]}})
        ws.send_json({"type": "select_exercise", "exercise": "bicep_curl"})
        ws.send_json({"type": "range_capture", "action": "start"})
        ws.send_json({"type": "range_capture", "action": "finish"})
        ws.send_json({"type": "select_exercise", "exercise": "nope"})
        ws.send_json({"type": "end_set"})
        while True:
            m = ws.receive_json()
            sink.append(m)
            if m["type"] == "set_saved" and m["reason"] == "end_set":
                break
    kinds = {m["type"] for m in sink}
    assert {"ready", "recognition", "state", "exercise_selected", "range_capture", "range_captured", "error", "set_saved", "setup"} <= kinds
    for m in sink:
        server_adapter.validate_python(m)                             # raises on any drift from backend/protocol.py


def test_generated_typescript_protocol_is_up_to_date():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("gen", root / "scripts" / "gen_protocol_ts.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (root / "web" / "lib" / "protocol.ts").read_text(encoding="utf-8") == gen.generate(), \
        "run: python scripts/gen_protocol_ts.py"


# ---------------------------------------------------------------- history and workout editing
def test_session_detail_has_per_rep_evidence(client, auth):
    sid = session(client, auth)
    f, t = curl(3)
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&session_id={sid}&token={tok(auth)}") as ws:
        stream(ws, f, t)
        ws.send_json({"type": "end_set", "weight_kg": 10})
        while ws.receive_json()["type"] != "set_saved":
            pass
    d = client.get(f"/api/sessions/{sid}", headers=auth).json()
    reps = d["sets"][0]["reps_detail"]
    assert len(reps) == 3 and all({"peak_progress", "duration", "evidence", "counted"} <= set(r) for r in reps)
    assert "reps_detail" not in client.get("/api/sessions", headers=auth).json()[0]["sets"][0]   # list stays light


def test_workout_edit_round_trip_with_rest_seconds(client, auth):
    w = client.post("/api/workouts", json={"name": "A", "items": [{"exercise": "squat", "sets": 3, "reps": 8}]}, headers=auth).json()
    r = client.put(f"/api/workouts/{w['id']}", json={"name": "A2", "items": [
        {"exercise": "bicep curl", "sets": 2, "reps": 12, "weight_kg": 10, "rest_seconds": 90},
        {"exercise": "squat", "sets": 4, "reps": 6, "rest_seconds": 120}]}, headers=auth).json()
    assert r["name"] == "A2" and [i["exercise_id"] for i in r["items"]] == ["bicep_curl", "squat"]
    assert [i["rest_seconds"] for i in r["items"]] == [90, 120]
    assert client.get("/api/workouts", headers=auth).json()[0]["items"][0]["rest_seconds"] == 90
