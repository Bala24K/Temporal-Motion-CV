"""
Modern Dashboard UI & Pose Visualization Renderer for OpenCV.

Render Components:
  - Setup & Positioning Calibration Guide (draw_setup_screen)
  - Modular Clean / Debug Pose Skeleton (draw_skeleton)
  - Glassmorphic Side Dashboard Panel (draw_dashboard_panel)
  - Rep Completion Animated Splash Banner (draw_rep_splash)
  - Tracking Loss & Positioning Banner (draw_tracking_loss)
  - Bottom Performance HUD Bar (draw_bottom_hud)
  - Post-Session Summary Screen (draw_session_summary)
"""

import time
from typing import Any, Dict, Optional, List
import cv2
import numpy as np

# Color Palette (BGR for OpenCV)
COLOR_BG_PANEL = (25, 28, 36)
COLOR_BORDER = (55, 62, 78)
COLOR_TEXT_PRIMARY = (245, 245, 245)
COLOR_TEXT_MUTED = (160, 168, 185)
COLOR_ACCENT_GREEN = (80, 220, 100)
COLOR_ACCENT_BLUE = (255, 180, 60)
COLOR_ACCENT_AMBER = (50, 180, 255)
COLOR_ACCENT_RED = (70, 70, 240)
COLOR_SKELETON_LINE = (230, 180, 70)
COLOR_SKELETON_JOINT = (255, 235, 120)
COLOR_HIGHLIGHT_JOINT = (100, 255, 100)

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

EXERCISE_KEY_JOINTS = {
    "bicep_curl": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
    "squat": ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"],
    "shoulder_press": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
    "lateral_raise": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
}


def _to_px(landmark, width: int, height: int, mirrored: bool = True):
    x = landmark.x
    if mirrored:
        x = 1.0 - x
    return int(x * width), int(landmark.y * height)


def draw_setup_screen(frame: np.ndarray, has_person: bool = False, mirrored: bool = True) -> np.ndarray:
    """Renders visual camera setup & full-body calibration overlay."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Center guide box / silhouette bounds
    box_w = int(w * 0.45)
    box_h = int(h * 0.75)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Translucent guide box
    box_color = COLOR_ACCENT_GREEN if has_person else COLOR_ACCENT_AMBER
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

    # Draw body silhouette guide inside box
    head_center = (w // 2, y1 + int(box_h * 0.18))
    head_radius = int(box_h * 0.08)
    cv2.circle(overlay, head_center, head_radius, box_color, 2, cv2.LINE_AA)

    torso_top = (w // 2, head_center[1] + head_radius)
    torso_bottom = (w // 2, y1 + int(box_h * 0.55))
    cv2.line(overlay, torso_top, torso_bottom, box_color, 2, cv2.LINE_AA)

    # Arms guide
    elbow_l = (w // 2 - int(box_w * 0.3), torso_top[1] + int(box_h * 0.15))
    wrist_l = (w // 2 - int(box_w * 0.35), torso_top[1] + int(box_h * 0.3))
    cv2.line(overlay, torso_top, elbow_l, box_color, 2, cv2.LINE_AA)
    cv2.line(overlay, elbow_l, wrist_l, box_color, 2, cv2.LINE_AA)

    elbow_r = (w // 2 + int(box_w * 0.3), torso_top[1] + int(box_h * 0.15))
    wrist_r = (w // 2 + int(box_w * 0.35), torso_top[1] + int(box_h * 0.3))
    cv2.line(overlay, torso_top, elbow_r, box_color, 2, cv2.LINE_AA)
    cv2.line(overlay, elbow_r, wrist_r, box_color, 2, cv2.LINE_AA)

    # Legs guide
    ankle_l = (w // 2 - int(box_w * 0.2), y2 - int(box_h * 0.05))
    ankle_r = (w // 2 + int(box_w * 0.2), y2 - int(box_h * 0.05))
    cv2.line(overlay, torso_bottom, ankle_l, box_color, 2, cv2.LINE_AA)
    cv2.line(overlay, torso_bottom, ankle_r, box_color, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Header title banner
    banner_h = 90
    cv2.rectangle(frame, (0, 0), (w, banner_h), COLOR_BG_PANEL, -1)
    cv2.line(frame, (0, banner_h), (w, banner_h), COLOR_BORDER, 2)

    cv2.putText(frame, "CAMERA SETUP & POSITIONING", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)

    sub_txt = "Stand 2-3 meters back | Keep your full body visible inside the guide frame"
    cv2.putText(frame, sub_txt, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Footer instruction prompt
    status_txt = "FULL BODY DETECTED — Press [SPACE] to Start Session" if has_person else "POSITION YOURSELF IN FRAME — Press [SPACE] to Start"
    status_col = COLOR_ACCENT_GREEN if has_person else COLOR_ACCENT_AMBER
    cv2.rectangle(frame, (0, h - 60), (w, h), COLOR_BG_PANEL, -1)
    cv2.line(frame, (0, h - 60), (w, h - 60), COLOR_BORDER, 2)
    cv2.putText(frame, status_txt, (30, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_col, 2, cv2.LINE_AA)

    return frame


def draw_skeleton(frame: np.ndarray, pose_frame: Any, exercise: str = "bicep_curl",
                  mirrored: bool = True, debug_mode: bool = False, min_visibility: float = 0.4) -> np.ndarray:
    """Draws sleek pose skeleton overlay with exercise-specific joint highlighting."""
    if pose_frame is None:
        return frame

    h, w = frame.shape[:2]
    key_joints = EXERCISE_KEY_JOINTS.get(exercise.lower().replace(" ", "_"), [])

    # Skeleton connections
    for a_name, b_name in SKELETON_CONNECTIONS:
        a = pose_frame.get(a_name)
        b = pose_frame.get(b_name)
        if a is None or b is None:
            continue
        if a.visibility < min_visibility or b.visibility < min_visibility:
            continue
        pa = _to_px(a, w, h, mirrored)
        pb = _to_px(b, w, h, mirrored)
        cv2.line(frame, pa, pb, COLOR_SKELETON_LINE, 3, cv2.LINE_AA)

    # Landmarks
    for name, lm in pose_frame.landmarks.items():
        if lm.visibility < min_visibility:
            continue
        p = _to_px(lm, w, h, mirrored)
        is_highlight = name in key_joints
        radius = 6 if is_highlight else 4
        color = COLOR_HIGHLIGHT_JOINT if is_highlight else COLOR_SKELETON_JOINT

        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

        if debug_mode:
            lbl = f"{name[:6]} ({lm.visibility:.2f})"
            cv2.putText(frame, lbl, (p[0] + 8, p[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_dashboard_panel(frame: np.ndarray, engine_res: Dict[str, Any], session_tracker: Any,
                         config: Any) -> np.ndarray:
    """Renders right-side glassmorphism dashboard overlay panel."""
    h, w, _ = frame.shape
    panel_w = 330
    x0 = w - panel_w

    # Glassmorphism dark background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), COLOR_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (x0, 0), (x0, h), COLOR_BORDER, 2)

    y = 30

    # Header: Title & Status Indicator
    cv2.putText(frame, "FITNESS CV", (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)

    is_running = session_tracker.state.value == "RUNNING"
    is_paused = session_tracker.state.value == "PAUSED"
    dot_color = COLOR_ACCENT_GREEN if is_running else (COLOR_ACCENT_AMBER if is_paused else COLOR_TEXT_MUTED)
    status_lbl = "LIVE" if is_running else ("PAUSED" if is_paused else "SETUP")

    cv2.circle(frame, (w - 75, y - 8), 6, dot_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status_lbl, (w - 62, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 1, cv2.LINE_AA)

    y += 35

    # Section Divider
    cv2.line(frame, (x0 + 15, y), (w - 15, y), COLOR_BORDER, 1)
    y += 25

    # Card 1: EXERCISE
    cv2.putText(frame, "EXERCISE", (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    y += 24

    valid_pose = engine_res.get("valid_pose", False)
    ex_name = engine_res.get("exercise", "Detecting...") if valid_pose else "TRACKING LOST"
    cv2.putText(frame, ex_name.upper(), (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT_PRIMARY if valid_pose else COLOR_ACCENT_AMBER, 2, cv2.LINE_AA)
    y += 30

    # Card 2: MODEL SCORE
    model_score_val = engine_res.get("model_score", engine_res.get("confidence", 0.0)) * 100.0

    cv2.putText(frame, "MODEL SCORE", (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    conf_str = f"{model_score_val:.1f}%" if valid_pose else "0.0%"
    cv2.putText(frame, conf_str, (x0 + 200, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT_GREEN if (valid_pose and model_score_val >= 50) else COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    y += 12

    # Model Score progress bar
    bar_w = panel_w - 40
    bar_fill = int((model_score_val / 100.0) * bar_w) if valid_pose else 0
    cv2.rectangle(frame, (x0 + 20, y), (x0 + 20 + bar_w, y + 8), COLOR_BORDER, -1)
    if bar_fill > 0 and valid_pose:
        fill_col = COLOR_ACCENT_GREEN if model_score_val >= 60 else COLOR_ACCENT_AMBER
        cv2.rectangle(frame, (x0 + 20, y), (x0 + 20 + bar_fill, y + 8), fill_col, -1)
    y += 32

    # Card 3: REPS COMPLETED
    rep_count = engine_res.get("rep_count", 0)
    card_h = 75
    cv2.rectangle(frame, (x0 + 20, y), (w - 20, y + card_h), (35, 40, 52), -1)
    cv2.rectangle(frame, (x0 + 20, y), (w - 20, y + card_h), COLOR_BORDER, 1)

    cv2.putText(frame, "REPS COMPLETED", (x0 + 35, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    rep_disp = f"{rep_count:02d}" if (valid_pose or rep_count > 0) else "--"
    cv2.putText(frame, rep_disp, (x0 + 35, y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT_PRIMARY if valid_pose else COLOR_TEXT_MUTED, 3, cv2.LINE_AA)
    y += card_h + 25

    # Card 4: MOVEMENT PHASE
    phase_str = engine_res.get("phase", "TRACKING LOST") if valid_pose else "TRACKING LOST"
    cv2.putText(frame, "MOVEMENT PHASE", (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    y += 24
    phase_col = COLOR_ACCENT_BLUE if valid_pose else COLOR_ACCENT_RED
    cv2.putText(frame, phase_str.upper(), (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_col, 2, cv2.LINE_AA)
    y += 35

    # Section Divider
    cv2.line(frame, (x0 + 15, y), (w - 15, y), COLOR_BORDER, 1)
    y += 25

    # Card 5: FORM FEEDBACK PANEL
    cv2.putText(frame, "FORM CHECK", (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    y += 26

    feedback_list = engine_res.get("form_feedback", [])
    if not feedback_list or not valid_pose:
        cv2.putText(frame, "- Position body in frame", (x0 + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        y += 24
    else:
        for fb in feedback_list[:3]:
            msg = fb.get("message", "")
            status = fb.get("status", "good")
            if status in ("good", "ok"):
                tag = "[OK]"
                col = COLOR_ACCENT_GREEN
            else:
                tag = "[WARN]"
                col = COLOR_ACCENT_AMBER

            txt = f"{tag} {msg}"
            cv2.putText(frame, txt, (x0 + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
            y += 24

    y += 10
    # Disclaimer note
    cv2.putText(frame, "* Movement heuristic checks", (x0 + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 125, 140), 1, cv2.LINE_AA)

    return frame


def draw_rep_splash(frame: np.ndarray, text: str) -> np.ndarray:
    """Renders prominent animated splash banner when a rep is completed."""
    h, w, _ = frame.shape
    banner_w = 280
    banner_h = 60
    x1 = (w - 330 - banner_w) // 2
    y1 = 30
    x2 = x1 + banner_w
    y2 = y1 + banner_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 140, 60), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ACCENT_GREEN, 2, cv2.LINE_AA)

    cv2.putText(frame, text, (x1 + 35, y1 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


def draw_tracking_loss(frame: np.ndarray) -> np.ndarray:
    """Renders tracking loss warning banner when person is not detected."""
    h, w, _ = frame.shape
    panel_w = 330
    center_x = (w - panel_w) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (center_x - 220, h // 2 - 35), (center_x + 220, h // 2 + 35), (20, 20, 28), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (center_x - 220, h // 2 - 35), (center_x + 220, h // 2 + 35), COLOR_ACCENT_AMBER, 2, cv2.LINE_AA)

    msg = "PERSON NOT DETECTED"
    sub = "Move 2-3m back so your full body is visible"
    cv2.putText(frame, msg, (center_x - 170, h // 2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT_AMBER, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (center_x - 205, h // 2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    return frame


def draw_bottom_hud(frame: np.ndarray, fps: float, latency_ms: float, session_tracker: Any, config: Any) -> np.ndarray:
    """Renders bottom performance HUD bar."""
    h, w, _ = frame.shape
    bar_h = 36
    y0 = h - bar_h

    cv2.rectangle(frame, (0, y0), (w, h), COLOR_BG_PANEL, -1)
    cv2.line(frame, (0, y0), (w, y0), COLOR_BORDER, 1)

    duration_str = session_tracker.get_elapsed_formatted()
    mode_str = config.exercise_mode.upper()
    hud_str = f"FPS: {fps:4.1f} | LATENCY: {latency_ms:4.1f}ms | DURATION: {duration_str} | MODE: {mode_str}"

    cv2.putText(frame, hud_str, (20, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

    ctrl_str = "[SPACE] Start/Pause | [R] Reset | [M] Mode | [D] Debug | [E] End"
    cv2.putText(frame, ctrl_str, (w - 480, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    return frame


def draw_session_summary(frame: np.ndarray, summary_data: Dict[str, Any], saved_file: Optional[str] = None) -> np.ndarray:
    """Renders post-session summary overlay screen."""
    h, w, _ = frame.shape

    # Dark background modal
    box_w = 600
    box_h = 460
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ACCENT_GREEN, 2, cv2.LINE_AA)

    y = y1 + 45
    cv2.putText(frame, "SESSION COMPLETE", (x1 + 40, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_ACCENT_GREEN, 2, cv2.LINE_AA)

    y += 35
    cv2.line(frame, (x1 + 30, y), (x2 - 30, y), COLOR_BORDER, 1)
    y += 35

    # Metrics Grid
    rows = [
        ("Exercise Mode:", f"{summary_data.get('exercise_mode', 'AUTO')} ({summary_data.get('detected_exercise', '-')})"),
        ("Total Reps Completed:", f"{summary_data.get('total_reps', 0)}"),
        ("Active Duration:", f"{summary_data.get('duration_formatted', '00:00')}"),
        ("Average Confidence:", f"{summary_data.get('avg_confidence_pct', 0.0):.1f}%"),
        ("Average System FPS:", f"{summary_data.get('avg_fps', 0.0):.1f} FPS"),
        ("Average Pipeline Latency:", f"{summary_data.get('avg_latency_ms', 0.0):.1f} ms"),
        ("Form Quality Checks:", f"{summary_data.get('good_form_checks', 0)} Good / {summary_data.get('warning_form_checks', 0)} Warnings"),
    ]

    for label, val in rows:
        cv2.putText(frame, label, (x1 + 40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, val, (x1 + 280, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)
        y += 32

    y += 15
    if saved_file:
        save_msg = f"Saved: {saved_file}"
        cv2.putText(frame, save_msg, (x1 + 40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_ACCENT_GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Press [S] to Save Session to JSON File", (x1 + 40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT_AMBER, 1, cv2.LINE_AA)

    y += 45
    cv2.putText(frame, "Press [SPACE] for New Session | Press [Q] to Exit", (x1 + 40, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)

    return frame
