"""Milestone 2 acceptance tests: calibration v3 (docs/REBUILD_ROADMAP.md "M2 - Calibration v2",
protocol in docs/POSE_CALIBRATION_AUDIT.md section 6). Real-recording targets (success <= 60 s for >= 90% of attempts,
ratio CV <= 5%) need recorded people and are not claimed here."""
import json
from pathlib import Path

import numpy as np
import pytest

from motion_engine import synthetic as S
from motion_engine.calibration import (assess_window, calibrate, check_setup, check_stage_frame, facing_yaw_deg,
                                       missing_stages, setup_fingerprint)
from motion_engine.skeleton import BodyProfile

FIXTURE = Path(__file__).parent / "fixtures" / "qevd_front_frames.json"


def tag(frames, w=1, h=1):
    return [{**f, "_image": [w, h]} for f in frames]


def all_stages(world=True, **kw):
    return {s: tag(S.standing_stage(s, 40, world=world, **kw)) for s in ("front", "arms_out", "side")}


# ---------------------------------------------------------------- facing: geometric, not proportion-based
def test_3d_yaw_reads_facing_and_side_on():
    assert facing_yaw_deg(S.standing_stage("front", 1)[0]) == pytest.approx(0, abs=1)
    assert facing_yaw_deg(S.standing_stage("side", 1)[0]) == pytest.approx(85, abs=1)
    assert facing_yaw_deg(S.standing_stage("front", 1, world=False)[0]) is None


@pytest.mark.parametrize("sw", [0.13, 0.15, 0.17, 0.19, 0.21])      # shoulder/torso 0.59 .. 0.95
def test_any_body_shape_facing_the_camera_passes_the_front_stage(monkeypatch, sw):
    monkeypatch.setattr(S, "SW", sw)
    for world in (True, False):
        f = tag(S.standing_stage("front", 1, world=world))[0]
        assert check_stage_frame("front", f)["ok"], (sw, world, check_stage_frame("front", f)["message"])


def test_narrow_shouldered_user_passes_with_3d_even_below_the_2d_ratio_gate(monkeypatch):
    monkeypatch.setattr(S, "SW", 0.09)                                 # ratio 0.41: the 2D fallback would refuse
    assert not check_stage_frame("front", tag(S.standing_stage("front", 1, world=False))[0])["ok"]
    assert check_stage_frame("front", tag(S.standing_stage("front", 1, world=True))[0])["ok"]


def test_turned_body_is_told_to_face_the_camera():
    turned = tag([S.with_world(f, 50) for f in S.synth_standing(1)[0]])[0]
    r = check_stage_frame("front", turned)
    assert not r["ok"] and r["message"] == "Turn to face the camera."


# ---------------------------------------------------------------- per-joint gating
def test_side_stage_passes_with_the_far_arm_occluded():
    frames = tag(S.standing_stage("side", 30))
    for f in frames:                                                   # far (left) limbs barely visible
        for j in ("left_elbow", "left_wrist", "left_knee", "left_ankle", "left_shoulder", "left_hip"):
            f[j] = (f[j][0], f[j][1], 0.2)
    assert check_stage_frame("side", frames[0])["ok"]
    assert assess_window("side", frames)["ok"]


def test_failure_messages_name_the_joint_and_the_fix():
    f = tag(S.standing_stage("front", 1))[0]
    dim = {**f, "right_wrist": (f["right_wrist"][0], f["right_wrist"][1], 0.2)}
    assert "right wrist" in check_stage_frame("front", dim)["message"]
    off = {**f, "left_ankle": (f["left_ankle"][0], 0.995, 1.0)}
    msg = check_stage_frame("front", off)["message"]
    assert "left ankle" in msg and "step back" in msg


def test_distance_and_centring_guidance():
    far = tag(S.standing_stage("front", 1, scale=0.4))[0]
    assert check_stage_frame("front", far)["message"] == "Come a little closer to the camera."
    shifted = {k: ((v[0] - 0.35, v[1], v[2]) if not k.startswith(("w:", "_")) else v) for k, v in tag(S.standing_stage("front", 1))[0].items()}
    assert check_stage_frame("front", shifted)["message"] == "Move toward the centre of the frame."


def test_arms_out_stage_validates_the_t_pose():
    assert check_stage_frame("arms_out", tag(S.standing_stage("arms_out", 1))[0])["ok"]
    r = check_stage_frame("arms_out", tag(S.standing_stage("front", 1))[0])
    assert not r["ok"] and "arms straight out" in r["message"]


# ---------------------------------------------------------------- stillness over a contiguous window
def test_window_accepts_a_still_body():
    r = assess_window("front", tag(S.standing_stage("front", 45)))
    assert r["ok"] and r["metrics"]["jitter"] < 0.03 and r["metrics"]["drift"] < 0.15


def test_jitter_is_rejected_with_hold_still():
    rng = np.random.default_rng(0)
    frames = [{k: ((v[0] + rng.normal(0, 0.012), v[1] + rng.normal(0, 0.012), v[2]) if not k.startswith(("w:", "_")) else v)
               for k, v in f.items()} for f in tag(S.standing_stage("front", 45))]
    r = assess_window("front", frames)
    assert not r["ok"] and r["message"] == "Hold still for two seconds."


def test_drift_is_rejected_as_movement():
    frames = [{k: ((v[0] + 0.003 * i, v[1], v[2]) if not k.startswith(("w:", "_")) else v) for k, v in f.items()}
              for i, f in enumerate(tag(S.standing_stage("front", 45)))]
    r = assess_window("front", frames)
    assert not r["ok"] and "moved" in r["message"]


def test_window_reports_the_most_common_problem():
    frames = tag(S.standing_stage("front", 30))
    frames = [{**f, "right_ankle": (f["right_ankle"][0], f["right_ankle"][1], 0.1)} for f in frames]
    r = assess_window("front", frames)
    assert not r["ok"] and "right ankle" in r["message"]


@pytest.mark.parametrize("res", ["640x480", "640x360"])
def test_real_facing_frames_pass_the_v3_frame_checks(res):
    """QEVD frames are not contiguous, so stillness is not tested on them, only the per-frame checks."""
    if not FIXTURE.exists():
        pytest.skip("fixture missing")
    frames = [r["landmarks"] for r in json.loads(FIXTURE.read_text())["frames"][res]]
    rate = np.mean([check_stage_frame("front", f)["ok"] for f in frames])
    assert rate >= 0.85, rate


# ---------------------------------------------------------------- profile: quality parts, status, fingerprint
def test_full_calibration_is_good_and_cross_checked_in_3d():
    p, _ = calibrate(all_stages(), 178, camera_label="laptop")
    assert p.calibrated and p.status == "good" and p.calibration_version == 3
    assert p.quality == pytest.approx(min(p.stability, p.confidence, p.coverage))
    assert p.method == "2d+3d" and p.ratio_agreement_3d > 0.9 and not p.warnings
    assert p.setup["aspect"] == 1.0 and p.setup["label"] == "laptop" and p.setup["body_height_frac"] > 0.5
    assert p.upper_arm / p.torso_len == pytest.approx(S.UA / S.T_, rel=0.03)    # arms measured from the T pose
    assert p.thigh / p.torso_len == pytest.approx(S.TH / S.T_, rel=0.03)


def test_front_only_is_usable_but_asks_for_the_arms_pose():
    p, _ = calibrate({"front": all_stages()["front"]})
    assert p.calibrated and p.status == "usable" and p.coverage == 0.6
    assert any("arms-out" in w for w in p.warnings) and missing_stages(p.stages) == ["arms_out", "side"]


def test_legacy_arms_up_stage_still_counts_as_the_arm_stage():
    st = all_stages()
    p, _ = calibrate({"front": st["front"], "arms_up": tag(S.standing_stage("arms_up", 40))})
    assert p.coverage == 1.0 and missing_stages(p.stages) == ["side"]


def test_ratios_do_not_depend_on_camera_aspect():
    a, _ = calibrate(all_stages())
    st = {k: [{**{j: ((0.5 + (v[0] - 0.5) * 9 / 16, v[1], v[2]) if not j.startswith(("w:", "_")) else v) for j, v in f.items()},
                "_image": [1280, 720]} for f in fr] for k, fr in all_stages().items()}
    b, _ = calibrate(st)
    assert b.ratios() == pytest.approx(a.ratios(), rel=0.01) and b.setup["aspect"] == pytest.approx(16 / 9, rel=1e-3)


def test_setup_check_detects_camera_and_distance_changes():
    p, _ = calibrate(all_stages())
    same = tag(S.standing_stage("front", 30))
    assert check_setup(p, same)["status"] == "ok"
    assert check_setup(p, tag(S.standing_stage("front", 30), 1280, 720))["status"] == "camera_changed"
    assert check_setup(p, tag(S.standing_stage("front", 30, scale=0.7)))["status"] == "distance_changed"
    assert check_setup(None, same)["status"] == "uncalibrated"
    assert check_setup(BodyProfile(), same)["status"] == "uncalibrated"


def test_fingerprint_roll_is_level_for_a_level_camera():
    assert abs(setup_fingerprint(tag(S.standing_stage("front", 10)))["ankle_roll_deg"]) < 3


# ---------------------------------------------------------------- API
def _json(frames):
    return [{k: list(v) for k, v in f.items()} for f in frames]


def test_assess_endpoint_and_protocol(client, auth):
    proto = client.get("/api/profile/calibrate/protocol").json()["stages"]
    assert [s["id"] for s in proto] == ["front", "arms_out", "side"] and proto[2]["optional"]
    ok = client.post("/api/profile/calibrate/assess", json={"stage": "front", "frames": _json(tag(S.standing_stage("front", 45)))}).json()
    assert ok["ok"] and {c["id"] for c in ok["checks"]} >= {"visibility", "in_frame", "facing", "pose", "still"}
    bad = client.post("/api/profile/calibrate/assess", json={"stage": "arms_out", "frames": _json(tag(S.standing_stage("front", 45)))}).json()
    assert not bad["ok"] and "arms straight out" in bad["message"]


def test_staged_endpoint_returns_quality_breakdown_and_stores_setup(client, auth):
    body = {"height_cm": 178, "camera_label": "test cam", "stages": {k: _json(v) for k, v in all_stages().items()}}
    r = client.post("/api/profile/calibrate/staged", json=body, headers=auth).json()
    assert r["calibrated"] and r["status"] == "good" and r["missing_stages"] == [] and r["method"] == "2d+3d"
    assert set(r) >= {"stability", "confidence", "coverage", "plausibility", "warnings", "setup"}
    prof = client.get("/api/profile", headers=auth).json()
    assert prof["calibration"]["setup"]["label"] == "test cam" and prof["missing_stages"] == []


def test_live_set_warns_when_the_camera_changed_since_calibration(client, auth):
    body = {"stages": {k: _json(v) for k, v in all_stages().items()}}
    assert client.post("/api/profile/calibrate/staged", json=body, headers=auth).status_code == 200
    frames, ts = S.synth_sequence("squat", reps=1)
    with client.websocket_connect(f"/ws/live?exercise=squat&token={auth['Authorization'][7:]}") as ws:
        got = None
        for f, t in zip(frames[:40], ts):
            g = {**{k: list(v) for k, v in f.items()}, "_image": [1280, 720]}
            ws.send_json({"type": "frame", "t": t, "landmarks": g})
            m = ws.receive_json()
            while m["type"] != "state":
                got = m if m["type"] == "setup" else got
                m = ws.receive_json()
    assert got and got["status"] == "camera_changed"


def test_demo_stages_include_world_landmarks(client):
    for st in ("front", "arms_out", "side", "arms_up"):
        f = client.get(f"/api/demo/standing?stage={st}&n=10").json()["frames"][0]
        assert "w:left_shoulder" in f and f["_image"] == [1, 1]
