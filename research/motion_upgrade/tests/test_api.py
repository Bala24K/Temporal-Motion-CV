"""
Tests for the Motion ST-GCN FastAPI Inference Service.

Covers:
  - /health endpoint
  - /model endpoint
  - /predict with valid and malformed input
  - /analyze with valid input
  - Model loading failure behaviour
  - API output schema validation
  - Local-vs-API prediction consistency
"""

import os
import sys
import time

import numpy as np
import pytest

# Ensure both motion-upgrade and Ver-1 are on the path
_test_dir = os.path.dirname(os.path.abspath(__file__))
_motion_dir = os.path.dirname(_test_dir)
_workspace_dir = os.path.dirname(_motion_dir)
_ver1_dir = os.path.join(_workspace_dir, "Ver-1")
for p in [_ver1_dir, _motion_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient

from pose.landmarks import Landmark, PoseFrame
from motion.recognizer import NeuralExerciseRecognizer


# ── Synthetic pose data ──────────────────────────────────────────────

def _make_pose_sequence(count: int = 20) -> list:
    """Builds a synthetic pose sequence matching the API schema.
    Uses the NEUTRAL standing pose from motion_fixtures with slight
    elbow motion to simulate movement.
    """
    frames = []
    for t in range(count):
        frames.append({
            "landmarks": {
                "nose":            {"x": 0.50, "y": 0.12, "z": 0.0, "visibility": 0.95},
                "left_shoulder":   {"x": 0.42, "y": 0.28, "z": 0.0, "visibility": 0.95},
                "right_shoulder":  {"x": 0.58, "y": 0.28, "z": 0.0, "visibility": 0.95},
                "left_elbow":      {"x": 0.40, "y": 0.42 + 0.005 * t, "z": 0.0, "visibility": 0.95},
                "right_elbow":     {"x": 0.60, "y": 0.42 + 0.005 * t, "z": 0.0, "visibility": 0.95},
                "left_wrist":      {"x": 0.39, "y": 0.56 + 0.01 * t, "z": 0.0, "visibility": 0.95},
                "right_wrist":     {"x": 0.61, "y": 0.56 + 0.01 * t, "z": 0.0, "visibility": 0.95},
                "left_hip":        {"x": 0.45, "y": 0.55, "z": 0.0, "visibility": 0.95},
                "right_hip":       {"x": 0.55, "y": 0.55, "z": 0.0, "visibility": 0.95},
                "left_knee":       {"x": 0.45, "y": 0.75, "z": 0.0, "visibility": 0.95},
                "right_knee":      {"x": 0.55, "y": 0.75, "z": 0.0, "visibility": 0.95},
                "left_ankle":      {"x": 0.45, "y": 0.94, "z": 0.0, "visibility": 0.95},
                "right_ankle":     {"x": 0.55, "y": 0.94, "z": 0.0, "visibility": 0.95},
            },
            "timestamp": float(t) / 7.5,
        })
    return frames


def _make_internal_pose_frames(count: int = 20) -> list:
    """Builds the exact same pose data as internal PoseFrame objects
    for consistency testing."""
    frames = []
    for t in range(count):
        pf = PoseFrame(timestamp=float(t) / 7.5)
        pf.landmarks = {
            "nose":            Landmark(0.50, 0.12, 0.0, 0.95),
            "left_shoulder":   Landmark(0.42, 0.28, 0.0, 0.95),
            "right_shoulder":  Landmark(0.58, 0.28, 0.0, 0.95),
            "left_elbow":      Landmark(0.40, 0.42 + 0.005 * t, 0.0, 0.95),
            "right_elbow":     Landmark(0.60, 0.42 + 0.005 * t, 0.0, 0.95),
            "left_wrist":      Landmark(0.39, 0.56 + 0.01 * t, 0.0, 0.95),
            "right_wrist":     Landmark(0.61, 0.56 + 0.01 * t, 0.0, 0.95),
            "left_hip":        Landmark(0.45, 0.55, 0.0, 0.95),
            "right_hip":       Landmark(0.55, 0.55, 0.0, 0.95),
            "left_knee":       Landmark(0.45, 0.75, 0.0, 0.95),
            "right_knee":      Landmark(0.55, 0.75, 0.0, 0.95),
            "left_ankle":      Landmark(0.45, 0.94, 0.0, 0.95),
            "right_ankle":     Landmark(0.55, 0.94, 0.0, 0.95),
        }
        frames.append(pf)
    return frames


# ── Fixtures ─────────────────────────────────────────────────────────

_MODEL_PATH = os.path.join(_motion_dir, "models", "motion_stgcn.pt")
_MODEL_EXISTS = os.path.exists(_MODEL_PATH)


@pytest.fixture()
def client(monkeypatch):
    """Returns a TestClient with the real ST-GCN model loaded."""
    if not _MODEL_EXISTS:
        pytest.skip(f"ST-GCN checkpoint not found at {_MODEL_PATH}")

    monkeypatch.setenv("MODEL_PATH", _MODEL_PATH)

    import api.config
    import importlib
    importlib.reload(api.config)

    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_no_model(tmp_path, monkeypatch):
    """Returns a TestClient with a nonexistent model path."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nonexistent.pt"))

    import api.config
    import importlib
    importlib.reload(api.config)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── /health tests ────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_name"] == "MotionSTGCN"

    def test_health_degraded_when_model_missing(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


# ── /model tests ─────────────────────────────────────────────────────


class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "MotionSTGCN"
        assert set(data["classes"]) == {"bicep_curl", "squat", "shoulder_press", "lateral_raise"}
        assert "streams" in data
        assert isinstance(data["window_frames"], int)
        assert isinstance(data["temperature"], float)
        assert isinstance(data["min_confidence"], float)

    def test_model_info_unavailable(self, client_no_model):
        resp = client_no_model.get("/model")
        assert resp.status_code == 503


# ── /predict tests ───────────────────────────────────────────────────


class TestPredict:
    def test_predict_valid(self, client):
        seq = _make_pose_sequence(20)
        resp = client.post("/predict", json={"sequence": seq})
        assert resp.status_code == 200
        data = resp.json()

        assert "class" in data
        # exercise is one of the classes or "unknown" / "other" if abstained
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "probabilities" in data
        assert data["frame_count"] == 20
        assert "abstained" in data
        assert isinstance(data["abstained"], bool)
        assert "valid_pose_ratio" in data
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0.0

    def test_predict_single_frame(self, client):
        seq = _make_pose_sequence(1)
        resp = client.post("/predict", json={"sequence": seq})
        assert resp.status_code == 200
        assert resp.json()["frame_count"] == 1

    def test_predict_empty_sequence(self, client):
        resp = client.post("/predict", json={"sequence": []})
        assert resp.status_code == 422  # Pydantic validation error

    def test_predict_missing_body(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_predict_malformed_landmarks(self, client):
        resp = client.post("/predict", json={
            "sequence": [{"landmarks": {"invalid_joint": {"x": 0, "y": 0}}}]
        })
        # Should not crash — handles missing/unknown joints gracefully
        assert resp.status_code == 200

    def test_predict_model_not_loaded(self, client_no_model):
        seq = _make_pose_sequence(5)
        resp = client_no_model.post("/predict", json={"sequence": seq})
        assert resp.status_code == 503

    def test_predict_response_schema(self, client):
        """Validates the exact shape of the /predict response."""
        seq = _make_pose_sequence(20)
        resp = client.post("/predict", json={"sequence": seq})
        data = resp.json()
        required_keys = {"class", "confidence", "probabilities", "frame_count",
                         "valid_pose_ratio", "abstained", "latency_ms"}
        assert required_keys.issubset(set(data.keys()))


# ── /analyze tests ───────────────────────────────────────────────────


class TestAnalyze:
    def test_analyze_valid(self, client):
        seq = _make_pose_sequence(20)
        resp = client.post("/analyze", json={"sequence": seq})
        assert resp.status_code == 200
        data = resp.json()

        assert "exercise" in data
        assert "exercise_id" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "confidence_is_calibrated" in data
        assert "abstained" in data
        assert isinstance(data["rep_count"], int)
        assert isinstance(data["learned_rep_count"], int)
        assert isinstance(data["state_machine_rep_count"], int)
        assert "rep_count_agreement" in data
        assert "phase" in data
        assert "learned_phase" in data
        assert "form_feedback" in data
        assert isinstance(data["form_feedback"], list)
        assert "probabilities" in data
        assert "latency_ms" in data
        assert "timing" in data

    def test_analyze_with_exercise_mode(self, client):
        seq = _make_pose_sequence(20)
        resp = client.post("/analyze", json={
            "sequence": seq,
            "exercise_mode": "squat",
        })
        assert resp.status_code == 200

    def test_analyze_model_not_loaded(self, client_no_model):
        seq = _make_pose_sequence(5)
        resp = client_no_model.post("/analyze", json={"sequence": seq})
        assert resp.status_code == 503


# ── Consistency tests ────────────────────────────────────────────────


class TestLocalVsAPIConsistency:
    """Verifies that local and API inference produce identical results
    for the same input data."""

    def test_predict_matches_local_inference(self, client):
        """The API /predict result must match NeuralExerciseRecognizer
        called directly with the same PoseFrame data."""
        # Local inference
        local_rec = NeuralExerciseRecognizer.load(_MODEL_PATH)
        local_frames = _make_internal_pose_frames(20)
        local_analysis = local_rec.analyze(local_frames)

        # API inference
        api_seq = _make_pose_sequence(20)
        resp = client.post("/predict", json={"sequence": api_seq})
        assert resp.status_code == 200
        api_result = resp.json()

        if local_analysis is None:
            # Both should indicate no valid result
            assert api_result["abstained"] is True
        else:
            # Same exercise prediction
            assert api_result["class"] == local_analysis.exercise
            # Same confidence (within floating-point tolerance)
            assert abs(api_result["confidence"] - local_analysis.confidence) < 1e-5
            # Same probabilities
            for cls in local_rec.classes:
                assert abs(
                    api_result["probabilities"].get(cls, 0.0) - local_analysis.probabilities.get(cls, 0.0)
                ) < 1e-5
