import logging
import numpy as np
import pytest

from exercise_system.registry import ExerciseRegistry
from motion_engine.embedding import build_prototypes, descriptor, match_families
from motion_engine.learned import LearnedRecognizer, feature_names, window_features
from motion_engine.recognition import Candidate, Recognition
from motion_engine.synthetic import EX, _pose, smoothstep, synth_sequence, synth_standing
from backend.recognizers import RecognizerService
from backend.security import RateLimiter
from backend.state import Metrics
from .conftest import register

REG = ExerciseRegistry.load()


def seq(fn, n=240):
    return [dict(_pose(**fn(smoothstep(1 - abs(2 * ((i % 80) / 80) - 1))))) for i in range(n)]


# ---------- few-shot family matching ----------
@pytest.fixture(scope="module")
def protos():
    return build_prototypes(sorted(EX))


def test_descriptor_is_scale_invariant_and_finds_lead_channel():
    a = descriptor(synth_sequence("squat", reps=3, scale=1.0)[0])
    b = descriptor(synth_sequence("squat", reps=3, scale=0.7, seed=4)[0])
    assert a["lead_channel"] == b["lead_channel"] == "knee_flexion"
    assert np.allclose(a["activity"], b["activity"], atol=0.1)
    assert 18 < a["tempo_rpm"] < 26            # 2.7 s per rep ~ 22 rpm


def test_unknown_overhead_arm_raise_maps_to_known_families_not_new_work(protos):
    r = match_families(seq(lambda p: dict(arm_elev=8 + 152 * p, elbow_int=172)), REG, protos)
    assert r["movement_detected"] and r["matches"][0]["family"] in ("shoulder_elevation", "overhead_press")
    assert "primitives likely sufficient" in r["verdict"] or "partial overlap" in r["verdict"]


def test_no_clear_movement_clip_is_reported(protos):
    r = match_families(synth_standing(90)[0], REG, protos)
    assert r["movement_detected"] is False


def test_admin_clip_triage_endpoint(client, auth, admin):
    client.post("/api/exercises/resolve", json={"name": "Cable Y Raise"}, headers=auth)
    rid = client.get("/api/admin/requests", headers=admin).json()[0]["id"]
    frames = [{**{k: [v[0], v[1], v[2]] for k, v in f.items()}, "_image": [1, 1]} for f in seq(lambda p: dict(arm_elev=8 + 152 * p, elbow_int=172))]
    r = client.post(f"/api/admin/requests/{rid}/clip", json={"frames": frames}, headers=admin)
    assert r.status_code == 200 and r.json()["matches"]
    assert client.post(f"/api/admin/requests/{rid}/clip", json={"frames": frames}, headers=auth).status_code == 403
    assert client.post(f"/api/admin/requests/999/clip", json={"frames": frames}, headers=admin).status_code == 404
    assert client.post(f"/api/admin/requests/{rid}/clip", json={"frames": frames[:5]}, headers=admin).status_code == 422


# ---------- ONNX recognizer ----------
@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    from ml.train_recognizer import main
    path = tmp_path_factory.mktemp("m") / "rec.onnx"
    main(str(path), fast=True)
    return path


def test_onnx_export_matches_numpy_forward(tmp_path):
    from ml.train_recognizer import export_onnx, forward
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    params = (rng.normal(size=35), rng.uniform(0.5, 2, 35), rng.normal(size=(35, 8)), rng.normal(size=8),
              rng.normal(size=(8, 4)), rng.normal(size=4))
    export_onnx(params, tmp_path / "m.onnx")
    X = rng.normal(size=(5, 35)).astype(np.float32)
    got = ort.InferenceSession(str(tmp_path / "m.onnx")).run(["logits"], {"x": X})[0]
    assert np.allclose(got, forward(params, X), atol=1e-4)


def test_learned_recognizer_contract(onnx_model):
    L = LearnedRecognizer(onnx_model)
    assert len(window_features(synth_sequence("squat", reps=2)[0])) == len(feature_names())
    r = L.recognize(synth_sequence("squat", reps=4, seed=777, scale=0.85)[0])
    assert r.status == "recognized" and r.top.exercise == "squat"
    r = L.recognize(synth_sequence("hammer_curl", reps=4, seed=777)[0])
    assert r.action == "confirm" and set(r.top.members) == {"bicep_curl", "hammer_curl"}
    assert L.recognize(synth_standing(90)[0]).action == "wait"
    assert L.recognize(synth_standing(10)[0]).action == "wait"


def test_stale_feature_version_refuses_to_load(onnx_model, tmp_path):
    import json, shutil
    shutil.copy(onnx_model, tmp_path / "m.onnx")
    meta = json.loads(onnx_model.with_suffix(".json").read_text()); meta["feature_version"] = 99
    (tmp_path / "m.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        LearnedRecognizer(tmp_path / "m.onnx")


class Stub:
    def __init__(self, members, action="analyze"):
        self.members, self.action = members, action

    def recognize(self, frames, **_):
        top = Candidate(self.members[0], self.members, 1.0, 0.95, len(self.members) == 1) if self.members else None
        return Recognition("recognized" if top else "unsupported", self.action if top else "fallback", 0.95 if top else 0, top)


def svc(mode, learned):
    s = RecognizerService(REG, Metrics(), mode="rules")
    s.mode, s.learned = mode, learned
    return s


def test_consensus_never_analyses_on_disagreement():
    f = synth_sequence("squat", reps=4)[0]
    assert svc("consensus", Stub(["squat"])).recognize(f).action == "analyze"
    out = svc("consensus", Stub(["romanian_deadlift"])).recognize(f)
    assert out.action == "confirm"
    out = svc("consensus", Stub([])).recognize(f)
    assert out.action == "confirm"                      # learned says unknown, rules say squat -> ask
    s = svc("shadow", Stub(["romanian_deadlift"]))
    assert s.recognize(f).action == "analyze" and s.metrics.counters["recognizer_disagree_total"] == 1


def test_broken_learned_model_falls_back_to_rules():
    class Boom:
        def recognize(self, *a, **k): raise RuntimeError("onnx exploded")
    s = svc("consensus", Boom())
    assert s.recognize(synth_sequence("squat", reps=4)[0]).action == "analyze"
    assert s.metrics.counters["learned_recognizer_errors_total"] == 1
    bad = RecognizerService(REG, Metrics(), onnx_path="/nonexistent.onnx", mode="consensus")
    assert bad.mode == "rules" and bad.metrics.counters["learned_recognizer_load_errors_total"] == 1


# ---------- security ----------
def test_security_headers_present(client):
    h = client.get("/api/health").headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert "camera=(self)" in h["permissions-policy"] and "frame-ancestors 'none'" in h["content-security-policy"]
    assert h["x-request-id"]


def test_login_is_rate_limited_per_account(client):
    register(client)
    codes = [client.post("/api/auth/login", json={"email": "a@example.com", "password": "wrongwrong"}).status_code
             for _ in range(10)]
    assert codes[:8] == [401] * 8 and codes[8:] == [429, 429]
    other = client.post("/api/auth/login", json={"email": "other@example.com", "password": "wrongwrong"})
    assert other.status_code == 401          # a different account is not locked out


def test_rate_limiter_window():
    r = RateLimiter(2, 10)
    assert r.allow("k", 0) and r.allow("k", 1) and not r.allow("k", 2) and r.allow("k", 11.5)


def test_ws_token_never_reaches_access_logs(client, auth, caplog):
    token = auth["Authorization"].split()[1]
    with caplog.at_level(logging.INFO, logger="fitness"):
        client.get(f"/api/health?token={token}")
    assert token not in caplog.text


def test_passwords_hashed_and_tokens_tamper_proof(client):
    from backend.security import create_token, decode_token, hash_password, verify_password
    h = hash_password("password123")
    assert "password123" not in h and verify_password("password123", h) and not verify_password("x", h)
    t = create_token(7, "k", 60)
    assert decode_token(t, "k") == 7 and decode_token(t, "other") is None
    head, body, sig = t.split(".")
    assert decode_token(f"{head}.{body[:-2]}xx.{sig}", "k") is None
    assert decode_token(create_token(7, "k", -5), "k") is None
