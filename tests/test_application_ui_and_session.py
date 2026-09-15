"""
Automated Unit Tests for AppConfig, SessionTracker, and Dashboard UI Overlay Renderer.
Runs headlessly without requiring physical camera hardware.
"""

import json
import os
import tempfile
import numpy as np
import pytest

from config import AppConfig
from recognition.session import SessionTracker, SessionState
from ui import dashboard


def test_app_config_defaults_and_properties():
    """Verify AppConfig initialization and video source property."""
    cfg = AppConfig(camera_id=1, width=1920, height=1080)
    assert cfg.camera_id == 1
    assert cfg.width == 1920
    assert cfg.height == 1080
    assert cfg.is_video_source is False

    cfg_vid = AppConfig(video_file="sample.mp4")
    assert cfg_vid.is_video_source is True


def test_session_tracker_state_transitions_and_timer():
    """Verify SessionTracker state machine and active elapsed time calculation."""
    session = SessionTracker(exercise_mode="BICEP CURL")
    assert session.state == SessionState.SETUP
    assert session.get_elapsed_seconds() == 0.0

    session.start_session()
    assert session.state == SessionState.RUNNING
    assert session.start_time is not None

    session.pause_session()
    assert session.state == SessionState.PAUSED

    session.pause_session()  # unpause
    assert session.state == SessionState.RUNNING

    session.end_session()
    assert session.state == SessionState.SUMMARY


def test_session_tracker_metrics_and_rep_splash():
    """Verify frame metrics logging, rep completion splash trigger, and form counters."""
    session = SessionTracker(exercise_mode="AUTO")
    session.start_session()

    engine_res_1 = {
        "valid_pose": True,
        "exercise": "bicep_curl",
        "confidence": 0.95,
        "rep_count": 0,
        "phase": "EXTENDED",
        "form_feedback": [{"message": "Controlled movement", "status": "good"}],
    }

    session.update_frame_metrics(engine_res_1, fps=50.0, latency_ms=18.5, now=100.0)
    assert session.active_exercise_name == "bicep_curl"
    assert session.good_form_count == 1

    # Simulate rep completion
    engine_res_2 = {
        "valid_pose": True,
        "exercise": "bicep_curl",
        "confidence": 0.96,
        "rep_count": 1,
        "phase": "FLEXED",
        "tempo": 2.1,
        "rom": 125.0,
        "form_feedback": [{"message": "Good range of motion", "status": "good"}],
    }
    session.update_frame_metrics(engine_res_2, fps=51.0, latency_ms=19.0, now=101.0)
    assert session.last_rep_count == 1
    assert session.splash_banner_text == "REP 1 \u2713"
    assert len(session.rep_events) == 1


def test_session_tracker_json_export():
    """Verify session summary JSON serialization and local file writing."""
    session = SessionTracker(exercise_mode="SQUAT")
    session.start_session()

    session.update_frame_metrics({
        "valid_pose": True,
        "exercise": "squat",
        "confidence": 0.91,
        "rep_count": 2,
        "form_feedback": [],
    }, fps=48.0, latency_ms=20.0, now=10.0)

    session.end_session()
    summary = session.get_summary()

    assert summary["exercise_mode"] == "SQUAT"
    assert summary["total_reps"] == 2
    assert summary["avg_confidence_pct"] == 91.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = session.save_summary_json(output_dir=tmp_dir)
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["exercise_mode"] == "SQUAT"
            assert data["total_reps"] == 2


def test_dashboard_ui_renderers_headless():
    """Verify OpenCV dashboard overlay rendering functions execute without exceptions."""
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Setup screen
    f1 = dashboard.draw_setup_screen(dummy_frame.copy(), has_person=True)
    assert f1.shape == (720, 1280, 3)

    # 2. Tracking loss banner
    f2 = dashboard.draw_tracking_loss(dummy_frame.copy())
    assert f2.shape == (720, 1280, 3)

    # 3. Dashboard panel
    session = SessionTracker(exercise_mode="AUTO")
    cfg = AppConfig()
    engine_res = {
        "valid_pose": True,
        "exercise": "shoulder_press",
        "confidence": 0.88,
        "rep_count": 5,
        "phase": "CONCENTRIC",
        "form_feedback": [{"message": "Controlled movement", "status": "good"}],
    }

    f3 = dashboard.draw_dashboard_panel(dummy_frame.copy(), engine_res, session, cfg)
    assert f3.shape == (720, 1280, 3)

    # 4. Rep splash
    f4 = dashboard.draw_rep_splash(dummy_frame.copy(), "REP 5 \u2713")
    assert f4.shape == (720, 1280, 3)

    # 5. Bottom HUD
    f5 = dashboard.draw_bottom_hud(dummy_frame.copy(), fps=52.0, latency_ms=19.1, session_tracker=session, config=cfg)
    assert f5.shape == (720, 1280, 3)

    # 6. Session Summary screen
    summary_data = session.get_summary()
    f6 = dashboard.draw_session_summary(dummy_frame.copy(), summary_data)
    assert f6.shape == (720, 1280, 3)
