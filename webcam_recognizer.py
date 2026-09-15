"""
Real-Time Fitness-CV Webcam Demo Module.

Delegates execution to app.py and AppConfig for unified modern dashboard rendering,
camera setup/calibration guide, and session tracking.
"""

import argparse
import sys
from app import main as app_main, run_app
from camera.sources import list_available_cameras, run_camera_test
from config import AppConfig


def run_fitness_cv_webcam_demo(bundle_path: str = "models/exercise_recognizer_bundle.joblib",
                                model_path: str = "models/pose_landmarker.task",
                                source: int = 0,
                                width: int = 1280,
                                height: int = 720,
                                fps: int = 30,
                                window_size: int = 45):
    """
    Backward compatible entry point delegating to AppConfig and run_app.
    """
    cam_id = source if isinstance(source, int) else 0
    video_file = source if isinstance(source, str) else None

    config = AppConfig(
        camera_id=cam_id,
        video_file=video_file,
        width=width,
        height=height,
        fps=fps,
        model_bundle_path=bundle_path,
        pose_model_path=model_path,
    )
    run_app(config)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Fitness-CV Demo & Camera Manager")
    parser.add_argument("--bundle-path", type=str, default="models/exercise_recognizer_bundle.joblib")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (e.g. 0 for webcam, 1 for iPhone/Camo)")
    parser.add_argument("--list-cameras", action="store_true", help="Enumerate available camera devices")
    parser.add_argument("--camera-test", action="store_true", help="Run camera diagnostic feed test")
    parser.add_argument("--video-file", type=str, default=None, help="Path to input .mp4 video file")
    parser.add_argument("--exercise", type=str, default="AUTO", choices=["AUTO", "BICEP CURL", "SQUAT", "SHOULDER PRESS", "LATERAL RAISE"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--record-session", action="store_true")
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

    if args.camera_test:
        run_camera_test(camera_id=args.camera_id, width=args.width, height=args.height, fps=args.fps)
        sys.exit(0)

    source_val = args.video_file if args.video_file else args.camera_id
    run_fitness_cv_webcam_demo(
        bundle_path=args.bundle_path,
        source=source_val,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
