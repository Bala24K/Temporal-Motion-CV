import pytest

from motion_engine.synthetic import synth_sequence, synth_standing
from .conftest import register


# Synthetic frames are isotropic; the frame contract requires declaring the source image (square here).
ISO = {"_image": [1, 1]}


def clip(ex, **kw):
    f, t = synth_sequence(ex, **kw)
    return [{**{k: [v[0], v[1], v[2]] for k, v in fr.items()}, **ISO} for fr in f], t


# ---------- auth ----------
def test_auth_flow_and_protection(client):
    assert client.get("/api/progress").status_code == 401
    h = register(client)
    assert client.get("/api/auth/me", headers=h).json()["email"] == "a@example.com"
    assert client.post("/api/auth/register", json={"email": "a@example.com", "password": "password123"}).status_code == 409
    assert client.post("/api/auth/login", json={"email": "a@example.com", "password": "wrongpass1"}).status_code == 401
    ok = client.post("/api/auth/login", json={"email": "a@example.com", "password": "password123"})
    assert ok.status_code == 200 and "session" in ok.cookies
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert client.post("/api/auth/register", json={"email": "bad", "password": "password123"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "b@example.com", "password": "short"}).status_code == 422


def test_admin_endpoints_require_admin(client, auth, admin):
    assert client.get("/api/admin/overview", headers=auth).status_code == 403
    assert client.get("/api/admin/overview", headers=admin).status_code == 200


def test_account_deletion_removes_data(client, auth):
    sid = client.post("/api/sessions", json={}, headers=auth).json()["id"]
    client.post(f"/api/sessions/{sid}/sets", json={"exercise": "squat", "reps": 5}, headers=auth)
    assert client.delete("/api/auth/me", headers=auth).status_code == 204
    assert client.get("/api/auth/me", headers=auth).status_code == 401


# ---------- exercise library / unsupported pipeline ----------
def test_exercise_library_and_filters(client):
    r = client.get("/api/exercises").json()
    assert r["count"] == 9 and "biceps" in r["by_muscle"]
    assert client.get("/api/exercises", params={"muscle": "glutes"}).json()["count"] >= 3
    assert client.get("/api/exercises/squat").json()["form_checks"]
    assert client.get("/api/exercises/nope").status_code == 404


def test_unsupported_exercise_is_honest_and_logged_for_developers(client, auth, admin):
    r = client.post("/api/exercises/resolve", json={"name": "Cable Y Raise"}, headers=auth).json()
    assert r["supported"] is False and r["manual_tracking"] and "isn't fully supported" in r["message"]
    client.post("/api/exercises/resolve", json={"name": "cable y raise"}, headers=auth)
    other = register(client, "c@example.com")
    client.post("/api/exercises/resolve", json={"name": "Cable Y-Raise"}, headers=other)
    ok = client.post("/api/exercises/resolve", json={"name": "overhead press"}, headers=auth).json()
    assert ok["supported"] and ok["exercise"]["id"] == "shoulder_press"
    reqs = client.get("/api/admin/requests", headers=admin).json()
    top = reqs[0]
    assert top["requests"] == 3 and top["distinct_users"] == 2 and top["candidate_pattern"] == "raise"
    m = client.get("/api/metrics").text
    assert "fitness_exercise_unsupported_total 3" in m
    assert client.patch(f"/api/admin/requests/{top['id']}", json={"status": "planned"}, headers=admin).json()["status"] == "planned"
    assert client.patch(f"/api/admin/requests/{top['id']}", json={"status": "bogus"}, headers=admin).status_code == 422


def test_add_exercise_via_config_takes_effect_without_redeploy(client, auth, admin):
    d = client.get("/api/exercises/bicep_curl").json()
    from backend.state import state
    definition = state.registry.get("bicep_curl").model_dump()
    definition.update(id="zottman_curl", name="Zottman Curl", aliases=["zottman"])
    assert client.post("/api/admin/exercises/validate", json={"definition": definition}, headers=admin).json()["valid"]
    assert client.post("/api/admin/exercises", json={"definition": definition}, headers=admin).status_code == 201
    r = client.post("/api/exercises/resolve", json={"name": "zottman"}, headers=auth).json()
    assert r["supported"] and r["exercise"]["id"] == "zottman_curl"
    definition["rep"]["signal"] = {"primitive": "bogus"}
    bad = client.post("/api/admin/exercises", json={"definition": definition}, headers=admin)
    assert bad.status_code == 422
    builtin = state.registry.get("squat").model_dump()
    assert client.post("/api/admin/exercises", json={"definition": builtin}, headers=admin).status_code == 409


# ---------- calibration ----------
def test_calibration_requires_clear_full_body_frames(client, auth):
    f, _ = synth_standing(40)
    frames = [{**{k: list(v) for k, v in fr.items()}, **ISO} for fr in f]
    r = client.post("/api/profile/calibrate", json={"height_cm": 178, "frames": frames}, headers=auth)
    assert r.status_code == 200 and r.json()["calibrated"]
    prof = client.get("/api/profile", headers=auth).json()
    assert prof["calibrated"] and prof["ratios"]["thigh"] > 0
    cut = [{k: v for k, v in fr.items() if "ankle" not in k} for fr in frames]
    r = client.post("/api/profile/calibrate", json={"frames": cut}, headers=auth)
    assert r.status_code == 422
    assert client.post("/api/profile/calibrate", json={"height_cm": 20, "frames": frames}, headers=auth).status_code == 422


# ---------- workouts, sessions, analysis ----------
def test_workout_builder_flags_unsupported_items(client, auth):
    w = client.post("/api/workouts", headers=auth, json={"name": "Push", "items": [
        {"exercise": "Dumbbell Shoulder Press", "sets": 4, "reps": 8, "weight_kg": 20},
        {"exercise": "Cable Y Raise", "sets": 3, "reps": 12}]}).json()
    assert w["items"][0]["exercise_id"] == "shoulder_press" and w["items"][0]["ai_analysis"]
    assert w["items"][1]["exercise_id"] is None and w["unsupported_count"] == 1
    assert "deltoids" in w["muscles"]
    other = register(client, "z@example.com")
    assert client.put(f"/api/workouts/{w['id']}", headers=other, json={"name": "x", "items": [{"exercise": "squat"}]}).status_code == 404
    assert client.get("/api/workouts", headers=other).json() == []


def test_analyze_clip_saves_set_and_builds_baseline(client, auth):
    sid = client.post("/api/sessions", json={"name": "Legs"}, headers=auth).json()["id"]
    frames, ts = clip("squat", reps=5)
    r = client.post("/api/analyze", headers=auth, json={"exercise": "squat", "frames": frames, "timestamps": ts,
                                                         "session_id": sid, "weight_kg": 60})
    assert r.status_code == 200 and r.json()["reps"] == 5 and r.json()["avg_form_score"] == 100.0
    s = client.post(f"/api/sessions/{sid}/finish", headers=auth).json()
    assert s["total_reps"] == 5 and s["volume_kg"] == 300 and s["sets"][0]["source"] == "ai"
    assert client.get("/api/profile", headers=auth).json()["baselines"]["squat"]["n"] == 5
    assert client.post("/api/analyze", headers=auth, json={"exercise": "cable y raise", "frames": frames,
                                                            "timestamps": ts}).status_code == 422
    assert client.post("/api/analyze", headers=auth, json={"exercise": "squat", "frames": frames,
                                                            "timestamps": ts[:-1]}).status_code == 422


def test_ai_set_with_unsupported_exercise_rejected_manual_allowed(client, auth):
    sid = client.post("/api/sessions", json={}, headers=auth).json()["id"]
    assert client.post(f"/api/sessions/{sid}/sets", headers=auth, json={"exercise": "Turkish Get-Up", "reps": 5,
                                                                         "source": "ai"}).status_code == 422
    r = client.post(f"/api/sessions/{sid}/sets", headers=auth, json={"exercise": "Turkish Get-Up", "reps": 5,
                                                                     "weight_kg": 16})
    assert r.status_code == 201 and r.json()["exercise_id"] is None and r.json()["avg_form_score"] is None


def test_session_isolation_between_users(client, auth):
    sid = client.post("/api/sessions", json={}, headers=auth).json()["id"]
    other = register(client, "o@example.com")
    assert client.get(f"/api/sessions/{sid}", headers=other).status_code == 404
    assert client.post(f"/api/sessions/{sid}/sets", headers=other, json={"exercise": "squat", "reps": 1}).status_code == 404


def test_progress_aggregates(client, auth):
    sid = client.post("/api/sessions", json={}, headers=auth).json()["id"]
    for ex, w in (("squat", 60), ("bicep_curl", 12)):
        frames, ts = clip(ex, reps=4)
        client.post("/api/analyze", headers=auth, json={"exercise": ex, "frames": frames, "timestamps": ts,
                                                         "session_id": sid, "weight_kg": w})
    client.post(f"/api/sessions/{sid}/sets", headers=auth, json={"exercise": "Turkish Get-Up", "reps": 3})
    client.post(f"/api/sessions/{sid}/finish", headers=auth)
    p = client.get("/api/progress", params={"range": "week"}, headers=auth).json()
    assert p["workouts"] == 1 and p["sets"] == 3 and p["reps"] == 11
    assert p["volume_kg"] == 4 * 60 + 4 * 12 and p["avg_form_score"] == 100.0
    assert set(p["muscle_sets"]) >= {"quadriceps", "biceps"} and len(p["series"]) == 7
    assert p["unsupported_sets"] == 1 and p["personal_bests"]["Squat"]["max_weight_kg"] == 60
    assert client.get("/api/progress", params={"range": "year"}, headers=auth).status_code == 422
    assert client.get("/api/progress", params={"range": "today"}, headers=auth).json()["workouts"] == 1


# ---------- model registry & promotion gate ----------
LINEAGE = dict(code_commit="abc1234", training_config={"lr": 1e-3}, pose_model_version="mediapipe-lite", feature_version=2,
               evaluation_dataset="ds-test123")
GOOD = dict(kind="movement", name="enc", dataset_version="FitnessDataset-v2", supported_exercises=["squat"],
            limitations="front/side only", lineage=LINEAGE,
            metrics=dict(macro_f1=0.8, subject_disjoint_accuracy=0.78, ece=0.06, latency_ms_p95=12, label_source="expert",
                         ood_auroc=0.95, conformal_coverage=0.9, id_ood_false_flag_rate=0.03))


def test_seeded_registry_is_honest(client, admin):
    models = client.get("/api/admin/models", headers=admin).json()
    rule = next(m for m in models if m["name"] == "definition-rules")
    assert rule["status"] == "production"
    st = next(m for m in models if m["name"] == "stgcn-transformer")
    assert st["status"] == "candidate" and st["metrics"]["label_source"] == "kinematic_rule"
    r = client.post(f"/api/admin/models/{st['id']}/promote", json={}, headers=admin)
    assert r.status_code == 409 and any("not human-verified" in x for x in r.json()["detail"]["blocked_by"])


def test_promotion_gate_blocks_regressions_then_allows_improvement(client, admin):
    mk = lambda v, **m: client.post("/api/admin/models", headers=admin, json={**GOOD, "version": v,
                                    "metrics": {**GOOD["metrics"], **m}}).json()
    v1, v2, v3 = mk(1), mk(2, macro_f1=0.7), mk(3, macro_f1=0.85, ece=0.05, latency_ms_p95=13)
    assert client.post(f"/api/admin/models/{v1['id']}/promote", json={}, headers=admin).json()["promoted"]
    bad = client.post(f"/api/admin/models/{v2['id']}/promote", json={}, headers=admin)
    assert bad.status_code == 409 and any("macro_f1 regressed" in r for r in bad.json()["detail"]["blocked_by"])
    slow = mk(4, latency_ms_p95=40)
    assert client.post(f"/api/admin/models/{slow['id']}/promote", json={}, headers=admin).status_code == 409
    ok = client.post(f"/api/admin/models/{v3['id']}/promote", json={}, headers=admin).json()
    assert ok["promoted"] and ok["replaced"] == 1
    prod = [m for m in client.get("/api/admin/models", headers=admin).json() if m["name"] == "enc" and m["status"] == "production"]
    assert [m["version"] for m in prod] == [3]
    assert client.post("/api/admin/models", headers=admin, json={**GOOD, "version": 3}).status_code == 409
    missing = client.post("/api/admin/models", headers=admin, json={**GOOD, "version": 9, "metrics": {}}).json()
    assert client.post(f"/api/admin/models/{missing['id']}/promote", json={}, headers=admin).status_code == 409


# ---------- live websocket ----------
def _stream(ws, frames, ts):
    msgs = []
    for fr, t in zip(frames, ts):
        ws.send_json({"type": "frame", "t": t, "landmarks": fr})
    return msgs


def _collect_until(ws, kind, limit=2000):
    out = []
    for _ in range(limit):
        m = ws.receive_json()
        out.append(m)
        if m["type"] == kind:
            return out
    raise AssertionError(f"never saw {kind}")


def test_live_ws_counts_reps_gives_feedback_and_saves(client, auth):
    token = auth["Authorization"].split()[1]
    sid = client.post("/api/sessions", json={}, headers=auth).json()["id"]
    frames, ts = clip("bicep_curl", reps=4, fault="drift")
    with client.websocket_connect(f"/ws/live?exercise=bicep_curl&target_reps=4&session_id={sid}&token={token}") as ws:
        for fr, t in zip(frames, ts):
            ws.send_json({"type": "frame", "t": t, "landmarks": fr})
        ws.send_json({"type": "end_set", "weight_kg": 10})
        msgs = _collect_until(ws, "set_saved")
    saved = msgs[-1]
    states = [m for m in msgs if m["type"] == "state"]
    assert saved["saved"] and saved["summary"]["reps"] == 4
    assert any(f["rule"] == "elbow_pinned" and f["status"] == "warning" for m in states for f in m["feedback"])
    s = client.get(f"/api/sessions/{sid}", headers=auth).json()
    assert s["sets"][0]["reps"] == 4 and s["sets"][0]["weight_kg"] == 10


def test_live_ws_auth_and_unsupported(client, auth):
    with client.websocket_connect("/ws/live?exercise=squat&token=bad") as ws:
        assert ws.receive_json()["type"] == "error"
    token = auth["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/live?exercise=cable_y_raise&token={token}") as ws:
        assert ws.receive_json()["type"] == "unsupported"


def test_live_ws_auto_detect_locks_on_and_replays_buffer(client, auth):
    token = auth["Authorization"].split()[1]
    frames, ts = clip("squat", reps=5)
    with client.websocket_connect(f"/ws/live?exercise=auto&token={token}") as ws:
        for fr, t in zip(frames, ts):
            ws.send_json({"type": "frame", "t": t, "landmarks": fr})
        ws.send_json({"type": "end_set"})
        msgs = _collect_until(ws, "set_saved")
    recs = [m for m in msgs if m["type"] == "recognition"]
    assert recs and any(r["action"] == "analyze" and r["top"]["exercise"] == "squat" for r in recs)
    assert msgs[-1]["summary"]["exercise"] == "squat" and msgs[-1]["summary"]["reps"] >= 4


def test_live_ws_auto_detect_asks_to_confirm_twins(client, auth):
    token = auth["Authorization"].split()[1]
    frames, ts = clip("hammer_curl", reps=4)
    with client.websocket_connect(f"/ws/live?exercise=auto&token={token}") as ws:
        for fr, t in zip(frames, ts):
            ws.send_json({"type": "frame", "t": t, "landmarks": fr})
        ws.send_json({"type": "end_set"})
        msgs = _collect_until(ws, "set_saved")
    recs = [m for m in msgs if m["type"] == "recognition"]
    assert recs and all(r["action"] == "confirm" for r in recs)
    assert msgs[-1]["saved"] is False            # never guesses between indistinguishable exercises


def test_demo_sequence_and_health(client):
    d = client.get("/api/demo/sequence", params={"exercise": "squat", "reps": 2}).json()
    assert len(d["frames"]) == len(d["timestamps"]) > 60
    assert client.get("/api/demo/sequence", params={"exercise": "zzz"}).status_code == 404
    assert client.get("/api/health").json()["status"] == "ok"
