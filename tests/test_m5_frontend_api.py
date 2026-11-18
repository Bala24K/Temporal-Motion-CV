"""Milestone 5: API fields the redesigned UI relies on (live HUD, library honesty, progress copy)."""


def test_exercise_list_exposes_what_the_live_hud_draws(client):
    ex = {e["id"]: e for e in client.get("/api/exercises").json()["exercises"]}
    assert ex["bicep_curl"]["rep_joints"] == ["shoulder", "elbow", "wrist"]
    assert ex["squat"]["rep_joints"] == ["hip", "knee", "ankle"] and ex["squat"]["laterality"] == "visible_side"
    assert 0 < ex["squat"]["count_progress"] < 1
    assert ex["squat"]["camera"]["preferred"] == "side" and ex["squat"]["camera"]["full_body"] is True


def test_exercise_detail_states_its_limitations(client):
    curl = client.get("/api/exercises/bicep_curl").json()
    assert any("Hammer Curl" in l for l in curl["limitations"])              # grip isn't visible to the camera
    assert any(v["views"] == ["front", "45deg"] for v in curl["view_limited_checks"])
    assert client.get("/api/exercises/squat").json()["limitations"] == []


def test_goal_headline_counts_training_days_and_pluralises(client, auth):
    """The weekly figure counts distinct training days (two sessions on one day = one day), and says so."""
    client.patch("/api/auth/me", json={"goal": "improve_fitness"}, headers=auth)
    client.post("/api/sessions", json={"name": "x"}, headers=auth)
    assert client.get("/api/progress/goal", headers=auth).json()["headline"].startswith("1 training day in")
    client.post("/api/sessions", json={"name": "y"}, headers=auth)
    assert client.get("/api/progress/goal", headers=auth).json()["headline"].startswith("1 training day in")
