"""
Bare OpenCV overlay UI — deliberately not React yet (see V1 plan step 1:
prove camera -> pose -> calculations -> feedback before spending time on
UI polish).

MIRRORING NOTE: We flip the camera frame horizontally before display so it
feels like a mirror to the user. PoseFrame landmarks are anatomical
(subject's real left/right) and normalized against the UNFLIPPED frame
MediaPipe actually saw. So every pixel conversion here must also flip the
x coordinate — that's what `_to_px` does. If you ever draw landmarks
without going through `_to_px`, the skeleton will not line up with the
mirrored video, and left/right will look swapped.
"""
import cv2
import numpy as np

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

COLOR_OK = (100, 220, 100)
COLOR_WARN = (60, 180, 250)
COLOR_BAD = (60, 60, 230)
COLOR_TEXT = (235, 235, 235)
COLOR_SKELETON = (255, 200, 80)
COLOR_JOINT = (80, 220, 255)


def _to_px(landmark, width, height, mirrored=True):
    x = landmark.x
    if mirrored:
        x = 1.0 - x
    return int(x * width), int(landmark.y * height)


def draw_skeleton(frame, pose_frame, mirrored=True, min_visibility=0.4):
    if pose_frame is None:
        return frame
    h, w = frame.shape[:2]
    for a_name, b_name in SKELETON_CONNECTIONS:
        a = pose_frame.get(a_name)
        b = pose_frame.get(b_name)
        if a is None or b is None:
            continue
        if a.visibility < min_visibility or b.visibility < min_visibility:
            continue
        pa = _to_px(a, w, h, mirrored)
        pb = _to_px(b, w, h, mirrored)
        cv2.line(frame, pa, pb, COLOR_SKELETON, 2, cv2.LINE_AA)

    for name, lm in pose_frame.landmarks.items():
        if lm.visibility < min_visibility:
            continue
        p = _to_px(lm, w, h, mirrored)
        cv2.circle(frame, p, 4, COLOR_JOINT, -1, cv2.LINE_AA)

    return frame


def draw_calibration_prompt(frame, message: str):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 40), (w, h // 2 + 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, message, (24, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, COLOR_WARN, 2, cv2.LINE_AA)
    return frame


def draw_stats_panel(frame, output: dict, fps: float, latency_ms: float):
    """output: the dict returned by an ExerciseAnalyzer.process(...) call."""
    h, w = frame.shape[:2]
    panel_w = 260
    x0 = w - panel_w
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 34
    cv2.putText(frame, output.get("exercise", "—"), (x0 + 16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    y += 40

    rep = output.get("rep_count", 0)
    cv2.putText(frame, f"REP        {rep:02d}", (x0 + 16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    y += 28

    phase = output.get("phase", "—")
    cv2.putText(frame, f"PHASE      {phase}", (x0 + 16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    y += 28

    rom = output.get("rom")
    cv2.putText(frame, f"ROM        {rom:.0f} deg" if rom is not None else "ROM        --",
                (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    y += 28

    tempo = output.get("tempo")
    cv2.putText(frame, f"TEMPO      {tempo:.1f}s" if tempo is not None else "TEMPO      --",
                (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    y += 36

    for flag in output.get("form_flags", []):
        # form_flags are {"level": "ok"|"warn", "message": str} dicts, not
        # pre-formatted strings with glyphs — cv2.putText only supports
        # Hershey fonts, which can't render a checkmark/warning symbol (it
        # renders as "???"). We draw the indicator as a shape instead.
        level = flag.get("level", "ok")
        message = flag.get("message", "")
        color = COLOR_OK if level == "ok" else COLOR_WARN
        dot_center = (x0 + 20, y - 5)
        if level == "ok":
            cv2.circle(frame, dot_center, 5, color, -1, cv2.LINE_AA)
        else:
            pts = np.array([[dot_center[0], dot_center[1] - 6],
                             [dot_center[0] - 6, dot_center[1] + 5],
                             [dot_center[0] + 6, dot_center[1] + 5]], dtype=np.int32)
            cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
        cv2.putText(frame, message, (x0 + 34, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, color, 1, cv2.LINE_AA)
        y += 24

    # bottom status bar: live indicator, FPS, latency
    bar_y = h - 20
    cv2.circle(frame, (x0 + 20, bar_y), 5, (60, 60, 230), -1)
    cv2.putText(frame, "LIVE", (x0 + 32, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS   {latency_ms:5.0f} ms", (x0 + 90, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1, cv2.LINE_AA)

    return frame
