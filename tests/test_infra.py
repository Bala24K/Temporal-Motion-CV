"""Guards against silent drift between code and its operational config."""
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = "\n".join(p.read_text() for p in (ROOT / "backend").rglob("*.py"))


def metric_names(text):
    return set(re.findall(r"fitness_[a-z_]+", text))


def emitted_counter(name):          # fitness_foo_total  <-  state.metrics.inc("foo_total")
    return f'"{name.removeprefix("fitness_")}"' in SRC


def emitted_latency_op(op):         # observe("frame_process", ...) or the client_stats tuple table
    return f'"{op}"' in SRC


def referenced(text):
    return metric_names(text), set(re.findall(r'op="([a-z_]+)"', text))


def test_alerts_and_dashboard_only_reference_metrics_the_app_emits():
    alerts = yaml.safe_load((ROOT / "infrastructure/alerts.yml").read_text())
    exprs = [r["expr"] for g in alerts["groups"] for r in g["rules"]]
    dash = json.loads((ROOT / "infrastructure/grafana/dashboards/kinetic.json").read_text())
    exprs += [t["expr"] for p in dash["panels"] for t in p["targets"]]
    assert len(exprs) >= 12
    for e in exprs:
        names, ops = referenced(e)
        for n in names - {"fitness_latency_ms"}:
            assert emitted_counter(n), f"{n} is referenced by alerts/dashboard but never emitted ({e})"
        for op in ops:
            assert emitted_latency_op(op), f"latency op {op!r} is referenced but never observed ({e})"


def test_configs_parse_and_are_wired_together():
    for f in ("infrastructure/prometheus.yml", "infrastructure/otel-collector.yml", "docker-compose.yml",
              "docker-compose.observability.yml", ".github/workflows/ci.yml", "dvc.yaml", "train.yaml"):
        assert yaml.safe_load((ROOT / f).read_text()), f
    prom = yaml.safe_load((ROOT / "infrastructure/prometheus.yml").read_text())
    assert prom["scrape_configs"][0]["metrics_path"] == "/api/metrics" and "/etc/prometheus/alerts.yml" in prom["rule_files"]
    obs = yaml.safe_load((ROOT / "docker-compose.observability.yml").read_text())
    assert {"prometheus", "grafana", "otel-collector"} <= set(obs["services"])
    json.loads((ROOT / "infrastructure/grafana/dashboards/kinetic.json").read_text())


def test_metrics_endpoint_emits_what_the_dashboard_queries(client, auth):
    token = auth["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/live?exercise=squat&token={token}") as ws:
        ws.send_json({"type": "client_stats", "camera_fps": 30, "pose_ms": 9, "e2e_ms": 40})
        from motion_engine.synthetic import synth_sequence
        f, t = synth_sequence("squat", reps=1)
        for fr, ti in zip(f, t):
            ws.send_json({"type": "frame", "t": ti, "landmarks": {**{k: list(v) for k, v in fr.items()}, "_image": [1, 1]}})
        ws.send_json({"type": "end_set"})
        while ws.receive_json()["type"] != "set_saved":
            pass
    client.post("/api/exercises/resolve", json={"name": "squat"}, headers=auth)
    text = client.get("/api/metrics").text
    for want in ('op="frame_process"', 'op="client_pose_ms"', 'op="client_camera_fps"', 'op="client_e2e_ms"', 'op="http"',
                 "fitness_frames_total", "fitness_exercise_resolve_total"):
        assert want in text, want


def test_dockerfile_ships_everything_the_server_imports():
    df = (ROOT / "Dockerfile").read_text()
    for needed in ("COPY backend", "COPY motion_engine", "COPY exercise_system", "COPY nutrition", "ml/data", "web/out"):
        assert needed in df, needed
    assert "torch_models" in df and "not shipped" in df            # training code deliberately excluded
    # every top-level package the backend imports must be copied into the image
    imported = set(re.findall(r"^(?:from|import) (ml|nutrition|motion_engine|exercise_system)\b", SRC, re.M))
    assert imported <= {"ml", "nutrition", "motion_engine", "exercise_system"} and all(f"COPY {m}" in df or f"{m}/" in df for m in imported)


def test_served_ml_modules_do_not_import_torch():
    """The serving image has no torch: anything the backend imports from ml/ must work without it."""
    served = list((ROOT / "ml/data").glob("*.py")) + [ROOT / "ml/seq.py", ROOT / "ml/tracking.py", ROOT / "ml/__init__.py"]
    for f in served:
        assert "import torch" not in f.read_text(), f


@pytest.mark.skipif(not (ROOT / "web/out/index.html").exists(), reason="run `npm run build` in web/ first")
def test_static_export_is_served_with_clean_routes(client):
    for path in ("/", "/workout/", "/live/", "/exercises/", "/progress/", "/nutrition/", "/profile/", "/admin/", "/login/"):
        r = client.get(path)
        assert r.status_code == 200 and "Kinetic" in r.text, path
    assert client.get("/api/health").json()["status"] == "ok"            # API still wins over the static mount
    assert client.get("/legacy/").status_code == 200
