"""
Fitness CV Application — Primary Entry Point.

Provides a production-grade, real-time computer vision fitness coaching application supporting:
  - External camera feeds (iPhone via Camo, DroidCam, EpocCam) & USB/built-in webcams
  - Camera setup/calibration screen with visual body positioning guide
  - Real-time exercise recognition, rep counting, and form feedback
  - Interactive keyboard controls (SPACE, p, r, m, d, f, s, e/q)
  - Post-session summary screen with local JSON export
  - Video recording mode (--record-session)
"""

import argparse
import os
import sys
import time
from collections import deque
import cv2

from camera.capture import Camera
from camera.sources import list_available_cameras, run_camera_test
from config import AppConfig
from pose.detector import PoseDetector
from pose.landmarks import PoseSequence
from recognition.fitness_pipeline import FitnessCVEngine
from recognition.session import SessionTracker, SessionState
from ui import dashboard


def run_app(config: AppConfig):
    """Executes the main Fitness CV real-time application."""
    print("=" * 60)
    print("  FITNESS CV — REAL-TIME COMPUTER VISION APPLICATION")
    print("=" * 60)

    # Initialize Fitness CV Engine
    print(f"Loading Fitness CV ML Engine ('{config.model_bundle_path}')...")
    try:
        engine = FitnessCVEngine(bundle_path=config.model_bundle_path)
    except Exception as e:
        print(f"Warning: Could not load bundle '{config.model_bundle_path}' ({e}). Fallback to rule-based engine.")
        engine = FitnessCVEngine(bundle_path=None)

    # Initialize Camera
    source_val = config.video_file if config.is_video_source else config.camera_id
    print(f"Opening camera source '{source_val}' ({config.width}x{config.height} @ {config.fps} FPS)...")
    try:
        cam = Camera(source=source_val, width=config.width, height=config.height, fps=config.fps).start()
    except Exception as e:
        print(f"Error opening camera source '{source_val}': {e}")
        sys.exit(1)

    # Initialize MediaPipe Pose Detector
    print(f"Initializing MediaPipe Pose detector ('{config.pose_model_path}')...")
    try:
        detector = PoseDetector(model_path=config.pose_model_path)
    except Exception as e:
        print(f"Error initializing MediaPipe Pose detector: {e}")
        cam.stop()
        sys.exit(1)

    # Session tracker & rolling pose window
    session = SessionTracker(exercise_mode=config.exercise_mode)
    pose_seq = PoseSequence(max_len=45)

    # Video Recorder (if opted-in)
    writer = None
    if config.record_session:
        os.makedirs(config.recordings_dir, exist_ok=True)
        rec_path = os.path.join(config.recordings_dir, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(rec_path, fourcc, config.fps, (config.width, config.height))
        print(f"Recording session to '{rec_path}'...")

    # Performance monitoring
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0
    saved_json_path = None

    print("\nApplication Started Successfully!")
    print("Controls:")
    print("  [SPACE] Start / Pause Session")
    print("  [R]     Reset Reps & Timer")
    print("  [M]     Cycle Exercise Mode (Auto -> Bicep Curl -> Squat -> Press -> Raise)")
    print("  [D]     Toggle Debug Overlay")
    print("  [F]     Toggle Horizontal Mirror Flip")
    print("  [S]     Save Session Summary (in Summary View)")
    print("  [E]/[Q] End Session / Exit\n")

    try:
        while True:
            loop_start = time.time()
            frame, cap_ts = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Calculate measured FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            # Optional horizontal mirror flip
            if config.horizontal_flip:
                display_frame = cv2.flip(frame, 1)
            else:
                display_frame = frame.copy()

    # Signal CSV logger if opted-in
    signal_log_file = None
    if config.debug_signals:
        os.makedirs(config.logs_dir, exist_ok=True)
        sig_path = os.path.join(config.logs_dir, f"signals_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        signal_log_file = open(sig_path, "w", encoding="utf-8")
        signal_log_file.write("timestamp,valid_person,exercise,model_score,angle,phase,rep_count\n")
        print(f"Logging real-time joint signals to '{sig_path}'...")

    last_valid_frame_time = time.time()

    try:
        while True:
            loop_start = time.time()
            frame, cap_ts = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Calculate measured FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            # Optional horizontal mirror flip
            if config.horizontal_flip:
                display_frame = cv2.flip(frame, 1)
            else:
                display_frame = frame.copy()

            now = time.time()

            # Pose Detection & Tracking Loss Management
            pf = detector.detect(frame)
            valid_person = (pf is not None and len(pf.landmarks) >= 5)

            if valid_person:
                pose_seq.push(pf)
                last_valid_frame_time = now
            else:
                # Clear historical frames after 500ms tracking loss grace period
                if (now - last_valid_frame_time) > 0.50:
                    pose_seq.clear()

            # Manual Exercise Mode override logic
            if config.exercise_mode != "AUTO":
                mapped_ex = config.exercise_mode.lower().replace(" ", "_")
                engine.active_exercise = mapped_ex

            # Pipeline Inference
            res = engine.process_frame_sequence(pose_seq.frames, now=now)
            latency_ms = (time.time() - loop_start) * 1000.0

            # Signal logger
            if signal_log_file:
                ang_str = f"{res.get('current_angle'):.1f}" if res.get('current_angle') is not None else ""
                signal_log_file.write(
                    f"{now:.3f},{valid_person},{res.get('exercise')},{res.get('model_score', 0.0):.2f},"
                    f"{ang_str},{res.get('phase')},{res.get('rep_count', 0)}\n"
                )

            # Update session metrics
            session.update_frame_metrics(res, fps=current_fps, latency_ms=latency_ms, now=now)

            # Render UI according to state
            if session.state == SessionState.SETUP:
                display_frame = dashboard.draw_setup_screen(display_frame, has_person=valid_person, mirrored=config.horizontal_flip)

            elif session.state in (SessionState.RUNNING, SessionState.PAUSED):
                # 1. Pose Skeleton
                display_frame = dashboard.draw_skeleton(
                    display_frame,
                    pose_frame=pf,
                    exercise=res.get("exercise", "bicep_curl"),
                    mirrored=config.horizontal_flip,
                    debug_mode=config.debug_overlay,
                )

                # 2. Tracking Loss Warning
                if not valid_person:
                    display_frame = dashboard.draw_tracking_loss(display_frame)

                # 3. Dashboard Panel
                display_frame = dashboard.draw_dashboard_panel(
                    display_frame,
                    engine_res=res,
                    session_tracker=session,
                    config=config,
                )

                # 4. Rep completion splash banner
                if session.splash_banner_text and now < session.splash_banner_until:
                    display_frame = dashboard.draw_rep_splash(display_frame, session.splash_banner_text)

                # 5. Bottom HUD
                display_frame = dashboard.draw_bottom_hud(
                    display_frame,
                    fps=current_fps,
                    latency_ms=latency_ms,
                    session_tracker=session,
                    config=config,
                )

            elif session.state == SessionState.SUMMARY:
                summary_data = session.get_summary()
                display_frame = dashboard.draw_session_summary(display_frame, summary_data, saved_file=saved_json_path)

            # Write recording frame if active
            if writer:
                writer.write(display_frame)

            # Render Window
            cv2.imshow("Fitness CV — Real-Time Exercise Intelligence", display_frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                if session.state != SessionState.SUMMARY:
                    session.end_session()
                else:
                    break
            elif key == ord('e'):
                session.end_session()
            elif key in (32, ord(' ')):  # SPACE key
                if session.state == SessionState.SETUP:
                    session.start_session()
                elif session.state in (SessionState.RUNNING, SessionState.PAUSED):
                    session.pause_session()
                elif session.state == SessionState.SUMMARY:
                    session.reset_session()
                    session.start_session()
                    saved_json_path = None
            elif key == ord('p'):
                if session.state in (SessionState.RUNNING, SessionState.PAUSED):
                    session.pause_session()
            elif key == ord('r'):
                session.reset_session()
                engine.reset_state()
                print("Session & Rep Counters Reset.")
            elif key == ord('m'):
                curr_idx = AppConfig.EXERCISE_OPTIONS.index(config.exercise_mode)
                next_idx = (curr_idx + 1) % len(AppConfig.EXERCISE_OPTIONS)
                config.exercise_mode = AppConfig.EXERCISE_OPTIONS[next_idx]
                session.exercise_mode = config.exercise_mode
                print(f"Exercise Mode set to: {config.exercise_mode}")
            elif key == ord('d'):
                config.debug_overlay = not config.debug_overlay
                print(f"Debug Overlay: {'ON' if config.debug_overlay else 'OFF'}")
            elif key == ord('f'):
                config.horizontal_flip = not config.horizontal_flip
                print(f"Horizontal Flip: {'ON' if config.horizontal_flip else 'OFF'}")
            elif key == ord('s'):
                if session.state == SessionState.SUMMARY:
                    saved_json_path = session.save_summary_json(output_dir=config.output_dir)

    finally:
        cam.stop()
        detector.close()
        if writer:
            writer.release()
        if signal_log_file:
            signal_log_file.close()
        cv2.destroyAllWindows()
        print("\nFitness CV session terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Fitness CV — Production Real-Time Exercise Intelligence")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (alias for --camera-id)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (e.g. 0 for built-in webcam, 1 for iPhone/Camo)")
    parser.add_argument("--list-cameras", action="store_true", help="Enumerate available video capture devices")
    parser.add_argument("--camera-test", action="store_true", help="Run live camera diagnostic preview")
    parser.add_argument("--video", type=str, default=None, help="Path to input .mp4 video file")
    parser.add_argument("--video-file", type=str, default=None, help="Path to input .mp4 video file (alias)")
    parser.add_argument("--demo-video", type=str, default=None, help="Path to input .mp4 video file for demo mode")
    parser.add_argument("--exercise", type=str, default="AUTO", choices=["AUTO", "BICEP CURL", "SQUAT", "SHOULDER PRESS", "LATERAL RAISE"])
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height")
    parser.add_argument("--fps", type=int, default=30, help="Requested frame rate")
    parser.add_argument("--flip-h", action="store_true", default=True, help="Enable horizontal mirror flip")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal mirror flip")
    parser.add_argument("--debug-overlay", action="store_true", help="Display diagnostic pose landmarks and angles")
    parser.add_argument("--debug-signals", action="store_true", help="Log real-time joint angle signals and state transitions to CSV")
    parser.add_argument("--record-session", action="store_true", help="Record application output to local MP4 video file")
    args = parser.parse_args()

    if args.list_cameras:
        cams = list_available_cameras()
        if not cams:
            print("No camera devices detected.")
        else:
            print("\nAvailable Camera Devices:")
            for c in cams:
                print(f"  ID {c['id']}: {c['name']} (Default Res: {c['width']}x{c['height']})")
        sys.exit(0)

    cam_id = args.camera if args.camera is not None else args.camera_id

    if args.camera_test:
        run_camera_test(camera_id=cam_id, width=args.width, height=args.height, fps=args.fps)
        sys.exit(0)

    video_path = args.video or args.video_file or args.demo_video

    config = AppConfig(
        camera_id=cam_id,
        video_file=video_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        horizontal_flip=not args.no_flip,
        exercise_mode=args.exercise,
        debug_overlay=args.debug_overlay,
        debug_signals=args.debug_signals,
        record_session=args.record_session,
    )

    run_app(config)


if __name__ == "__main__":
    main()
