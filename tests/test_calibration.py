import numpy as np
import pytest

from exercise_system.registry import ExerciseRegistry
from motion_engine.calibration import CalibrationError, calibrate, check_stage_frame
from motion_engine.session import analyze_sequence
from motion_engine.skeleton import BodyProfile
from motion_engine.synthetic import _pose, synth_sequence, synth_standing, yaw_project

REG = ExerciseRegistry.load()


def stages(n=30, scale=1.0):
    front, _ = synth_standing(n, scale=scale)
    return {"front": front, "side": [yaw_project(f, 85) for f in front],
            "arms_up": [dict(_pose(scale=scale, arm_elev=170, elbow_int=175)) for _ in range(n)]}


@pytest.fixture(scope="module")
def profile():
    return calibrate(stages(), height_cm=178)[0]


def test_staged_calibration_builds_personal_profile(profile):
    assert profile.calibrated and profile.stages == ["front", "side", "arms_up"] and profile.calibration_version == 3
    assert profile.quality > 0.6 and profile.plausibility > 0.5 and profile.shoulder_ratio > 0.5
    assert profile.cm_per_torso > 0


def test_physical_scale_is_independent_of_distance_to_camera():
    near = calibrate(stages(scale=1.0), 178)[0]
    far = calibrate(stages(scale=0.6), 178)[0]
    assert far.torso_len < near.torso_len
    assert far.cm_per_torso == pytest.approx(near.cm_per_torso, rel=0.03)
    assert far.ratios() == pytest.approx(near.ratios(), rel=0.03)


def test_more_stages_give_higher_quality():
    s = stages()
    q3 = calibrate(s, 178)[0].quality
    q1 = calibrate({"front": s["front"]}, 178)[0].quality
    assert q1 < q3


def test_each_stage_validates_what_it_is_for():
    front = synth_standing(5)[0][0]
    assert check_stage_frame("front", front)["ok"]
    assert "sideways" in check_stage_frame("side", front)["message"]
    assert "arms" in check_stage_frame("arms_up", front)["message"]
    side = yaw_project(front, 85)
    assert check_stage_frame("side", side)["ok"] and "face the camera" in check_stage_frame("front", side)["message"].lower()
    with pytest.raises(ValueError):
        check_stage_frame("nope", front)


def test_user_who_just_faces_the_camera_three_times_gets_a_flagged_partial_profile():
    front = synth_standing(30)[0]
    prof, res = calibrate({"front": front, "side": front, "arms_up": front}, 178)
    by = {r.stage: r for r in res}
    assert by["front"].ok and not by["side"].ok and not by["arms_up"].ok and "sideways" in by["side"].message
    assert prof.stages == ["front"]


def test_front_stage_is_mandatory_and_clipped_body_rejected():
    with pytest.raises(CalibrationError):
        calibrate({"side": synth_standing(30)[0]}, 178)
    cut = [{k: v for k, v in f.items() if "ankle" not in k} for f in synth_standing(30)[0]]
    with pytest.raises(CalibrationError, match="whole body"):
        calibrate({"front": cut}, 178)


def test_implausible_proportions_lower_plausibility(profile):
    s = stages()
    for f in s["front"]:                       # absurdly long forearms (e.g. wrong person/camera tilt)
        for side in ("left", "right"):
            ex, ey, v = f[f"{side}_elbow"]; wx, wy, _ = f[f"{side}_wrist"]
            f[f"{side}_wrist"] = (ex + 3 * (wx - ex), ey + 3 * (wy - ey), v)
    bad = calibrate({"front": s["front"]}, 178)[0]
    assert bad.plausibility < profile.plausibility


@pytest.mark.parametrize("ex,depth", [("lateral_raise", 0.8), ("shoulder_press", 0.8)])
def test_yaw_compensation_makes_rom_view_invariant_for_frontal_motion(profile, ex, depth):
    d = REG.get(ex)
    assert d.motion_plane == "frontal"
    raw_def = d.model_copy(update={"motion_plane": "mixed"})
    f, t = synth_sequence(ex, reps=4, seed=2, depth=depth)
    straight = analyze_sequence(d, f, t, profile=profile)["avg_rom"]
    yawed = [yaw_project(x, 50) for x in f]
    comp = analyze_sequence(d, yawed, t, profile=profile)["avg_rom"]
    raw = analyze_sequence(raw_def, yawed, t, profile=profile)["avg_rom"]
    assert abs(comp - straight) < 0.02
    assert abs(raw - straight) > 0.05                  # without it the camera angle changes the verdict


def test_sagittal_exercises_are_not_yaw_corrected():
    assert all(REG.get(e).motion_plane == "sagittal" for e in ("squat", "bicep_curl", "romanian_deadlift"))


def test_profile_roundtrip_keeps_v2_fields(profile):
    p2 = BodyProfile.from_dict(profile.to_dict())
    assert p2.stages == profile.stages and p2.shoulder_ratio == profile.shoulder_ratio


def test_staged_api(client, auth):
    s = stages()
    fr = lambda lst: [{**{k: list(v) for k, v in f.items()}, "_image": [1, 1]} for f in lst]
    body = {"height_cm": 178, "stages": {k: fr(v) for k, v in s.items()}}
    r = client.post("/api/profile/calibrate/staged", json=body, headers=auth)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["calibrated"] and j["missing_stages"] == [] and j["segments_cm"]["thigh"] > 0 and j["plausibility"] > 0.5
    prof = client.get("/api/profile", headers=auth).json()
    assert prof["stages"] == ["front", "side", "arms_up"] and prof["segments_cm"] and "side" in prof["instructions"]
    assert client.post("/api/profile/calibrate/staged", json={"stages": {"side": body["stages"]["side"]}}, headers=auth).status_code == 422
    assert client.post("/api/profile/calibrate/staged", json={"stages": {"bogus": []}}, headers=auth).status_code == 422
    c = client.post("/api/profile/calibrate/stage-check", json={"stage": "side", "frame": fr(s["front"])[0]}, headers=auth)
    assert c.json()["ok"] is False and "sideways" in c.json()["message"]
    assert client.post("/api/profile/calibrate/stage-check", json={"stage": "x", "frame": {}}, headers=auth).status_code == 422
