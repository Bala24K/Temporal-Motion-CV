"""
Unit tests for ExerciseRecognizer inference API and model bundle serialization.
"""

import os
import tempfile
import numpy as np
import pytest

from pose.landmarks import Landmark, PoseFrame
from recognition.feature_engineering import FeatureConfig
from recognition.exercise_recognizer import (
    ExerciseRecognizer,
    ExerciseRecognitionResult,
    train_and_save_recognizer,
)
from recognition.dataset_synthesis import EXERCISES
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def _make_mock_pose_frames(count: int = 10) -> list:
    frames = []
    for t in range(count):
        pf = PoseFrame()
        pf.landmarks = {
            "left_shoulder": Landmark(0.4, 0.3, 0.0),
            "right_shoulder": Landmark(0.6, 0.3, 0.0),
            "left_elbow": Landmark(0.35, 0.5 + 0.01 * t, 0.0),
            "right_elbow": Landmark(0.65, 0.5 + 0.01 * t, 0.0),
            "left_wrist": Landmark(0.3, 0.6, 0.0),
            "right_wrist": Landmark(0.7, 0.6, 0.0),
            "left_hip": Landmark(0.45, 0.7, 0.0),
            "right_hip": Landmark(0.55, 0.7, 0.0),
            "left_knee": Landmark(0.45, 0.85, 0.0),
            "right_knee": Landmark(0.55, 0.85, 0.0),
            "left_ankle": Landmark(0.45, 0.95, 0.0),
            "right_ankle": Landmark(0.55, 0.95, 0.0),
            "nose": Landmark(0.5, 0.2, 0.0),
        }
        frames.append(pf)
    return frames


def test_exercise_recognizer_prediction_format():
    # Fit simple model on dummy data
    X_dummy = np.random.randn(20, 870)
    y_dummy = np.array(EXERCISES * 5)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)

    scaler = StandardScaler()
    scaler.fit(X_dummy)

    rec = ExerciseRecognizer(
        model=model,
        classes=list(EXERCISES),
        feature_config=FeatureConfig(),
        scaler=scaler,
        seed=42,
    )

    frames = _make_mock_pose_frames(10)
    res = rec.predict_sequence(frames)

    assert isinstance(res, ExerciseRecognitionResult)
    assert res.exercise in EXERCISES
    assert 0.0 <= res.confidence <= 1.0
    assert set(res.probabilities.keys()) == set(EXERCISES)
    assert res.frame_count == 10

    d = res.to_dict()
    assert "exercise" in d
    assert "confidence" in d
    assert "probabilities" in d


def test_exercise_recognizer_empty_and_invalid_pose():
    X_dummy = np.random.randn(20, 870)
    y_dummy = np.array(EXERCISES * 5)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)

    rec = ExerciseRecognizer(model=model, classes=list(EXERCISES))

    # Test empty sequence
    res_empty = rec.predict_sequence([])
    assert res_empty.frame_count == 0
    assert res_empty.confidence == 0.0
    assert len(res_empty.probabilities) == 4


def test_bundle_serialization_deserialization():
    X_dummy = np.random.randn(20, 870)
    y_dummy = np.array(EXERCISES * 5)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    scaler = StandardScaler().fit(X_dummy)

    rec_orig = ExerciseRecognizer(
        model=model,
        classes=list(EXERCISES),
        feature_config=FeatureConfig(use_velocity=True),
        scaler=scaler,
        seed=123,
        model_name="TestRF",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = os.path.join(tmpdir, "test_bundle.joblib")
        rec_orig.save_bundle(bundle_path)
        assert os.path.exists(bundle_path)

        rec_loaded = ExerciseRecognizer.load_bundle(bundle_path)
        assert rec_loaded.seed == 123
        assert rec_loaded.model_name == "TestRF"
        assert rec_loaded.feature_config.use_velocity is True

        frames = _make_mock_pose_frames(10)
        res_orig = rec_orig.predict_sequence(frames)
        res_loaded = rec_loaded.predict_sequence(frames)

        assert res_orig.exercise == res_loaded.exercise
        assert np.isclose(res_orig.confidence, res_loaded.confidence)
