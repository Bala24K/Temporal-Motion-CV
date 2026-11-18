import io
import json
import time
from datetime import date, datetime, timedelta, timezone

import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from backend import telemetry
from backend.db import Base, make_engine, migrate
from backend.oidc import OIDCVerifier
from backend.state import state
from motion_engine.synthetic import synth_sequence
from nutrition.energy import classify_intensity, estimate_session_kcal
from nutrition.foods import FOODS, macros_for
from nutrition.targets import ProfileIncomplete, bmr, targets
from .conftest import register
from .test_api import GOOD, LINEAGE, clip


# ============================ nutrition ============================
def test_bmr_matches_published_mifflin_st_jeor_examples():
    assert bmr(70, 175, 30, "male") == pytest.approx(1648.75)            # 700 + 1093.75 - 150 + 5
    assert bmr(60, 165, 25, "female") == pytest.approx(1345.25)


def test_targets_are_ranges_with_a_safety_floor_and_consistent_macros():
    t = targets(80, 180, 1995, "male", "moderate", "build_muscle", today=date(2026, 1, 1))
    assert t["target_kcal"] > t["maintenance_kcal"] and t["kcal_range"][0] < t["target_kcal"] < t["kcal_range"][1]
    assert abs(t["protein_g"] * 4 + t["carbs_g"] * 4 + t["fat_g"] * 9 - t["target_kcal"]) < 25
    tiny = targets(42, 150, 1950, "female", "sedentary", "lose_weight", today=date(2026, 1, 1))
    assert tiny["target_kcal"] == 1200 and tiny["floor_applied"]
    with pytest.raises(ProfileIncomplete, match="weight_kg"):
        targets(None, 170, 1990, "male", "light", None)
    with pytest.raises(ProfileIncomplete, match="14-100"):
        targets(60, 160, 2020, "male", "light", None, today=date(2026, 1, 1))


def test_energy_estimate_is_a_range_and_flags_assumed_weight():
    e = estimate_session_kcal(60, 80, "moderate")
    assert e["kcal_low"] < e["kcal_mid"] == round(4.5 * 80) < e["kcal_high"] and "Estimate" in e["note"]
    assert estimate_session_kcal(60, None)["assumptions"]["weight_assumed"] is True
    assert estimate_session_kcal(0, 80)["kcal_mid"] == 0
    assert classify_intensity(5, 12, 30) == "vigorous" and classify_intensity(3, 8, 120) == "light"


def test_food_table_is_internally_consistent():
    for k, f in FOODS.items():                     # Atwater check: kcal ~ 4P + 4C + 9F (tolerance for fibre/water)
        approx = 4 * f["protein"] + 4 * f["carbs"] + 9 * f["fat"]
        # fibre counts as "carbs" but yields ~2 kcal/g, so allow a small absolute slack for low-calorie fibrous foods
        assert abs(approx - f["kcal"]) / f["kcal"] < 0.25 or abs(approx - f["kcal"]) < 10, k
    assert macros_for("chicken_breast", 200)["protein_g"] == 62.0


def test_nutrition_api_flow(client, auth):
    assert client.get("/api/nutrition/targets", headers=auth).status_code == 422
    r = client.patch("/api/nutrition/profile", headers=auth, json={"sex": "male", "birth_year": 1995, "height_cm": 180,
                                                                  "weight_kg": 80, "activity_level": "moderate"})
    assert r.status_code == 200 and client.patch("/api/nutrition/profile", headers=auth, json={"activity_level": "x"}).status_code == 422
    assert client.get("/api/nutrition/targets", headers=auth).json()["target_kcal"] > 2000
    day = "2026-03-01"
    a = client.post("/api/nutrition/log", headers=auth, json={"day": day, "food_key": "egg", "grams": 100}).json()
    client.post("/api/nutrition/log", headers=auth, json={"day": day, "name": "Mom's curry", "grams": 300, "kcal": 450, "protein_g": 20})
    assert client.post("/api/nutrition/log", headers=auth, json={"food_key": "nope", "grams": 5}).status_code == 422
    assert client.post("/api/nutrition/log", headers=auth, json={"grams": 5}).status_code == 422
    d = client.get("/api/nutrition/day", params={"day": day}, headers=auth).json()
    assert d["totals"]["kcal"] == 143 + 450 and len(d["entries"]) == 2 and d["remaining_kcal"] == d["targets"]["target_kcal"] - 593
    other = register(client, "n@example.com")
    assert client.delete(f"/api/nutrition/log/{a['id']}", headers=other).status_code == 404
    assert client.delete(f"/api/nutrition/log/{a['id']}", headers=auth).status_code == 204
    assert client.get("/api/nutrition/foods", params={"q": "rice"}).json()[0]["key"] == "white_rice"


# ============================ schema migration ============================
def test_migration_adds_new_columns_to_an_old_database(tmp_path):
    url = f"sqlite:///{tmp_path}/old.db"
    eng = make_engine(url)
    with eng.begin() as c:                                           # the v1 schema: no nutrition/lineage columns
        c.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255), password_hash VARCHAR(255), name VARCHAR(120), "
                          "is_admin BOOLEAN, goal VARCHAR(40), weekly_target INTEGER, created_at DATETIME)")
        c.exec_driver_sql("INSERT INTO users (id,email,password_hash,name,is_admin,weekly_target) VALUES (1,'a@b.c','x','n',0,3)")
        c.exec_driver_sql("CREATE TABLE model_versions (id INTEGER PRIMARY KEY, kind VARCHAR(20), name VARCHAR(80), version INTEGER, "
                          "dataset_version VARCHAR(80), status VARCHAR(20), metrics JSON, supported_exercises JSON, limitations TEXT, "
                          "artifact_uri VARCHAR(255), created_at DATETIME, promoted_at DATETIME)")
    added = migrate(eng)
    assert {"users.sex", "users.weight_kg", "model_versions.lineage", "model_versions.promotion_record"} <= set(added)
    assert migrate(eng) == []                                        # idempotent
    with eng.connect() as c:
        assert c.exec_driver_sql("SELECT email, sex FROM users").fetchone() == ("a@b.c", None)   # data preserved


# ============================ model lineage & stricter gate ============================
def test_gate_requires_lineage_ood_and_conformal(client, admin):
    def reg(v, **over):
        body = json.loads(json.dumps(GOOD)); body["version"] = v
        for k, val in over.items():
            body[k] = {**body[k], **val} if isinstance(val, dict) else val
        return client.post("/api/admin/models", headers=admin, json=body).json()
    def promote(m):
        return client.post(f"/api/admin/models/{m['id']}/promote", json={}, headers=admin)
    assert promote(reg(1, lineage={"code_commit": ""})).status_code == 409
    assert "OOD detection too weak" in str(promote(reg(2, metrics={"ood_auroc": 0.6})).json())
    assert "conformal coverage" in str(promote(reg(3, metrics={"conformal_coverage": 0.5})).json())
    assert "in-distribution inputs rejected" in str(promote(reg(4, metrics={"id_ood_false_flag_rate": 0.4})).json())
    ok = promote(reg(5))
    assert ok.status_code == 200
    rec = ok.json()["model"]["promotion_record"]
    assert rec["gate"]["passed"] and rec["metrics_at_promotion"]["macro_f1"] == 0.8 and rec["replaced_version"] is None


def test_why_is_this_model_in_production(client, admin):
    m = client.post("/api/admin/models", headers=admin, json={**GOOD, "version": 7}).json()
    client.post(f"/api/admin/models/{m['id']}/promote", json={}, headers=admin)
    prod = client.get("/api/admin/models/production", headers=admin).json()
    assert any(p["version"] == 7 for p in prod)
    why = client.get(f"/api/admin/models/{m['id']}/why", headers=admin).json()
    assert why["reproducible"] and why["model"]["lineage"]["code_commit"] == "abc1234" and why["model"]["promotion_record"]["promoted_by"]
    seeded = next(x for x in client.get("/api/admin/models", headers=admin).json() if x["name"] == "definition-rules")
    assert client.get(f"/api/admin/models/{seeded['id']}/why", headers=admin).json()["reproducible"] is False   # honest about the seed
    assert client.get("/api/admin/models/9999/why", headers=admin).status_code == 404


def test_register_candidate_from_training_run_copies_lineage(client, admin):
    run = {"id": "r1", "params": {"lr": 0.001, "epochs": 60}, "dataset_version": "ds-abc123", "code_commit": "deadbee",
           "metrics": {**GOOD["metrics"]}, "artifacts": {"onnx": "models/e.onnx"}, "seconds": 812.5}
    body = {"kind": "movement", "name": "enc", "version": 3, "run": run, "sidecar": {"classes": ["squat", "bicep_curl"], "feature_version": 2},
            "limitations": "front/side only", "pose_model_version": "mediapipe-lite", "evaluation_dataset": "ds-abc123", "hardware": "cpu"}
    r = client.post("/api/admin/models/from-run", json=body, headers=admin)
    assert r.status_code == 201 and r.json()["lineage"]["code_commit"] == "deadbee" and r.json()["dataset_version"] == "ds-abc123"
    assert r.json()["lineage"]["training_config"]["epochs"] == 60 and r.json()["supported_exercises"] == ["squat", "bicep_curl"]
    assert client.post(f"/api/admin/models/{r.json()['id']}/promote", json={}, headers=admin).status_code == 200
    assert client.post("/api/admin/models/from-run", json=body, headers=admin).status_code == 409


# ============================ OIDC ============================
@pytest.fixture(scope="module")
def idp():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(key.public_key())); jwk.update(kid="k1", alg="RS256", use="sig")
    def token(claims=None, signer=key, kid="k1", alg="RS256"):
        base = {"iss": "https://idp.test", "aud": "kinetic", "sub": "u1", "exp": int(time.time()) + 300,
                "email": "oidc@example.com", "email_verified": True, "name": "Oidc User"}
        return jwt.encode({**base, **(claims or {})}, signer, algorithm=alg, headers={"kid": kid})
    return {"jwks": {"keys": [jwk]}, "token": token, "other": other}


def verifier(idp, calls=None):
    def prov():
        if calls is not None: calls.append(1)
        return idp["jwks"]
    return OIDCVerifier("https://idp.test", "kinetic", prov)


def test_oidc_verifier_accepts_valid_and_rejects_everything_else(idp):
    v, t = verifier(idp), idp["token"]
    assert v.verify(t())["sub"] == "u1"
    assert v.verify(t({"aud": "someone-else"})) is None and v.verify(t({"iss": "https://evil"})) is None
    assert v.verify(t({"exp": int(time.time()) - 3600})) is None
    assert v.verify(t(signer=idp["other"])) is None                          # wrong signing key
    assert v.verify(t(kid="nope")) is None
    assert v.verify(jwt.encode({"iss": "https://idp.test", "aud": "kinetic", "sub": "x", "exp": int(time.time()) + 99}, "secret", algorithm="HS256", headers={"kid": "k1"})) is None   # alg confusion
    assert v.verify(jwt.encode({"sub": "x"}, None, algorithm="none", headers={"kid": "k1"})) is None
    assert v.verify("not.a.jwt") is None and v.verify(t({"sub": None})) is None


def test_oidc_unknown_kid_does_not_hammer_the_idp(idp):
    calls = []
    v = verifier(idp, calls)
    v.verify(idp["token"]())
    for _ in range(20):
        v.verify(idp["token"](kid="rotating"))
    assert len(calls) <= 2


def test_oidc_login_provisions_user_only_with_verified_email(client, idp, monkeypatch):
    monkeypatch.setattr(state, "oidc", verifier(idp))
    ok = client.get("/api/auth/me", headers={"Authorization": "Bearer " + idp["token"]()})
    assert ok.status_code == 200 and ok.json()["email"] == "oidc@example.com"
    again = client.get("/api/auth/me", headers={"Authorization": "Bearer " + idp["token"]()})
    assert again.json()["id"] == ok.json()["id"]                               # matched, not duplicated
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer " + idp["token"]({"email_verified": False, "email": "x@example.com"})}).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer " + idp["token"]({"email": None})}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "oidc@example.com", "password": "anything123"}).status_code == 401   # no password login for IdP users
    assert client.get("/api/auth/config").json()["oidc"]["issuer"] == "https://idp.test"
    monkeypatch.setattr(state, "oidc", None)
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer " + idp["token"]()}).status_code == 401


def test_oidc_token_works_for_websocket(client, idp, monkeypatch):
    monkeypatch.setattr(state, "oidc", verifier(idp))
    with client.websocket_connect("/ws/live?exercise=squat&token=" + idp["token"]()) as ws:
        ws.send_json({"type": "end_set"})
        assert ws.receive_json()["type"] == "set_saved"


# ============================ telemetry ============================
def test_http_and_recognition_spans_and_client_stats(client, auth):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    mem = InMemorySpanExporter()
    telemetry.provider.add_span_processor(SimpleSpanProcessor(mem))
    token = auth["Authorization"].split()[1]
    client.get(f"/api/health?token={token}")
    frames, ts = clip("squat", reps=4)
    with client.websocket_connect(f"/ws/live?exercise=auto&token={token}") as ws:
        ws.send_json({"type": "client_stats", "camera_fps": 29.5, "pose_ms": 11.0, "e2e_ms": 48.0})
        ws.send_json({"type": "client_stats", "pose_ms": -5, "camera_fps": "bad"})          # junk is ignored
        for fr, t in zip(frames, ts):
            ws.send_json({"type": "frame", "t": t, "landmarks": fr})
        ws.send_json({"type": "end_set"})
        while ws.receive_json()["type"] != "set_saved":
            pass
    names = [s.name for s in mem.get_finished_spans()]
    assert "http.request" in names and "recognition" in names
    http = next(s for s in mem.get_finished_spans() if s.name == "http.request" and s.attributes["http.target"] == "/api/health")
    assert http.attributes["http.status_code"] == 200 and token not in str(dict(http.attributes))
    rec = next(s for s in mem.get_finished_spans() if s.name == "recognition")
    assert rec.attributes["recognition.action"] in ("analyze", "confirm", "fallback", "wait")
    snap = state.metrics.snapshot()["latency_ms"]
    assert snap["client_pose_ms"]["n"] == 1 and snap["client_camera_fps"]["p50"] == 29.5


# ============================ ingestion jobs ============================
class FakeEst:
    name, version = "fake", "1"
    def __init__(self): self.truth, _ = synth_sequence("squat", reps=3, noise=0.0); self.i = 0
    def detect(self, img, t): self.i += 1; return self.truth[(self.i - 1) % len(self.truth)]
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


def mp4_bytes(tmp_path, n=45):
    import cv2
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    if not w.isOpened():
        pytest.skip("no mp4 encoder")
    for _ in range(n):
        w.write(np.full((48, 64, 3), 9, np.uint8))
    w.release()
    return p.read_bytes()


def test_video_ingest_job_end_to_end(client, auth, admin, tmp_path, monkeypatch):
    monkeypatch.setattr(state, "pose_factory", FakeEst)
    monkeypatch.setattr(state, "dataset_dir", str(tmp_path / "ds"))
    data = mp4_bytes(tmp_path)
    files = lambda: {"file": ("c.mp4", io.BytesIO(data), "video/mp4")}
    assert client.post("/api/admin/dataset/ingest", headers=auth, files=files(), data={"subject_id": "s1"}).status_code == 403
    assert client.post("/api/admin/dataset/ingest", headers=admin, files=files(), data={"subject_id": "s1", "view": "bad"}).status_code == 422
    assert client.post("/api/admin/dataset/ingest", headers=admin, files=files(), data={"subject_id": "s1", "exercise_id": "nope"}).status_code == 422
    assert client.post("/api/admin/dataset/ingest", headers=admin, files=files(), data={"subject_id": "s1", "source": "consented_user"}).status_code == 422
    r = client.post("/api/admin/dataset/ingest", headers=admin, files=files(), data={"subject_id": "s1", "exercise_id": "squat", "view": "side"})
    assert r.status_code == 202
    state.jobs.wait(r.json()["job_id"])
    j = client.get(f"/api/admin/jobs/{r.json()['job_id']}", headers=admin).json()
    assert j["status"] == "done" and j["result"]["pose_model"] == "fake:1" and j["result"]["exercise_id"] == "squat"
    rep = client.get("/api/admin/dataset/report", headers=admin).json()
    assert rep["clips"] == 1 and rep["per_exercise"] == {"squat": 1} and rep["version"].startswith("ds-") and not rep["gate_ready"]
    cid = r.json()["clip_id"]
    ls = [{"data": {"clip_id": cid}, "annotations": [{"result": [{"type": "choices", "from_name": "exercise", "value": {"choices": ["squat"]}}]}]},
          {"data": {"clip_id": "ghost"}, "annotations": [{"result": []}]}]
    imp = client.post("/api/admin/dataset/annotations", headers=admin, json=ls).json()
    assert imp["imported"] == 1 and imp["skipped_unknown_clips"] == 1 and imp["dataset_version"] != rep["version"]
    assert client.get("/api/admin/jobs/nope", headers=admin).status_code == 404


def test_failed_ingest_reports_error_and_leaves_no_clip(client, admin, tmp_path, monkeypatch):
    class Never(FakeEst):
        def detect(self, img, t): return None
    monkeypatch.setattr(state, "pose_factory", Never)
    monkeypatch.setattr(state, "dataset_dir", str(tmp_path / "ds"))
    r = client.post("/api/admin/dataset/ingest", headers=admin, data={"subject_id": "s"}, files={"file": ("c.mp4", io.BytesIO(mp4_bytes(tmp_path)), "video/mp4")})
    j = state.jobs.wait(r.json()["job_id"])
    assert j["status"] == "failed" and "pose found" in j["error"]
    assert client.get("/api/admin/dataset/report", headers=admin).json()["clips"] == 0


# ============================ progress analytics ============================
def seed_history(client, auth, weights, days_ago_start=21):
    """One AI set per 'day' via the API, then back-date by editing created_at directly."""
    from backend.db import SetRecord
    sid = client.post("/api/sessions", json={}, headers=auth).json()["id"]
    ids = []
    for w in weights:
        frames, ts = clip("squat", reps=3)
        client.post("/api/analyze", headers=auth, json={"exercise": "squat", "frames": frames, "timestamps": ts, "session_id": sid, "weight_kg": w})
    with state.SessionLocal() as db:
        rows = db.query(SetRecord).order_by(SetRecord.id).all()
        for i, r in enumerate(rows):
            r.created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago_start - 3 * i)
        db.commit()
    return sid


def test_exercise_progress_e1rm_prs_and_trends(client, auth):
    seed_history(client, auth, [40, 50, 60, 55, 70])
    lst = client.get("/api/progress/exercises", headers=auth).json()
    assert lst[0]["name"] == "Squat" and lst[0]["best_e1rm"] == round(70 * (1 + 3 / 30), 1)
    p = client.get("/api/progress/exercise/squat", headers=auth).json()
    assert len(p["series"]) == 5 and [x["e1rm"] for x in p["personal_records"]] == [44.0, 55.0, 66.0, 77.0]   # 55kg day is not a PR
    assert p["trends"]["e1rm"]["direction"] == "improving" and p["trends"]["form"]["direction"] in ("flat", "improving")
    assert p["series"][0]["avg_rom"] > 0.9 and p["ai_analysis"]
    assert client.get("/api/progress/exercise/Turkish Get-Up", headers=auth).status_code == 404


def test_trend_helper_needs_data_and_handles_flat_and_decline():
    from backend.routers.progress import e1rm, trend
    assert trend([1, 2, 3])["direction"] == "insufficient_data" and trend([None, 5, None, 6])["points"] == 2
    assert trend([10, 10.05, 9.98, 10.02, 10.01])["direction"] == "flat"
    assert trend([10, 9, 8, 7, 6])["direction"] == "declining"
    assert e1rm(100, 1) == 100 and e1rm(0, 10) is None and e1rm(60, 10) == 80.0


def test_goal_progress_streak_and_headline(client, auth):
    client.patch("/api/auth/me", headers=auth, json={"goal": "consistency", "weekly_target": 1})
    from backend.db import WorkoutSession
    with state.SessionLocal() as db:
        from backend.db import User
        uid = db.query(User).first().id
        today = datetime.now(timezone.utc).replace(tzinfo=None)
        for wk in (1, 2, 3):                                   # sessions in each of the last three completed weeks
            db.add(WorkoutSession(user_id=uid, started_at=today - timedelta(weeks=wk), finished_at=today - timedelta(weeks=wk) + timedelta(minutes=40)))
        db.commit()
    g = client.get("/api/progress/goal", params={"weeks": 6}, headers=auth).json()
    assert len(g["weeks"]) == 6 and g["streak_weeks"] >= 3 and "streak" in g["headline"]
    assert client.get("/api/progress/goal", params={"weeks": 1}, headers=auth).status_code == 422
    assert "muscle_share" in client.get("/api/progress", headers=auth).json()
