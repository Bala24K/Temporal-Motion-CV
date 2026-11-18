"""
Live entry point for the learned pipeline.

    # measure where the time actually goes, no camera needed
    python -m motion.live --profile --seconds 20

    # run against the webcam with a trained checkpoint
    python -m motion.live --checkpoint models/motion_stgcn.pt

    # or against a recorded clip
    python -m motion.live --video recordings/session.mp4 --checkpoint models/motion_stgcn.pt

The profiling mode is the one to run first. It drives the whole pipeline
with a synthetic camera and prints per-stage p50/p95 latency, so the
question "what is actually slow" has an answer before anything is
optimised. It needs no webcam, no checkpoint and no torch — the
recogniser is simply absent, which is itself the useful baseline: it
shows what pose estimation alone costs.
"""
import argparse
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from config import AppConfig
from motion.fusion import FusionConfig, HybridMovementEngine
from motion.runtime import AsyncMotionPipeline, PipelineConfig


@dataclass
class LiveStack:
    """The assembled runtime, kept together so callers stop it as a unit."""
    pipeline: AsyncMotionPipeline
    engine: HybridMovementEngine

    def start(self) -> "LiveStack":
        self.pipeline.start()
        return self

    def stop(self) -> None:
        self.pipeline.stop()

    def state(self) -> Dict[str, Any]:
        """The fused view for the UI, with live pipeline health attached."""
        pipeline_state = self.pipeline.state()
        fused = self.engine.state()
        fused.update({
            "capture_fps": round(pipeline_state.capture_fps, 1),
            "pose_fps": round(pipeline_state.pose_fps, 1),
            "inference_fps": round(pipeline_state.inference_fps, 1),
            "frames_dropped": pipeline_state.frames_dropped,
        })
        return fused

    def __enter__(self) -> "LiveStack":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False


def build_live_stack(
    source: Any,
    detector: Any,
    recognizer: Optional[Any] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    fusion_config: Optional[FusionConfig] = None,
) -> LiveStack:
    """
    Wire the four pieces together.

    Everything is injected, so this same function builds the real webcam
    stack, the video-file stack and the synthetic profiling stack. The
    alternative — constructing the camera and detector inside — is what
    makes live systems untestable.
    """
    engine = HybridMovementEngine(fusion_config)
    pipeline = AsyncMotionPipeline(
        source=source,
        detector=detector,
        recognizer=recognizer,
        config=pipeline_config or PipelineConfig(),
        on_pose=engine.update_pose,
        on_recognition=engine.update_recognition,
    )
    return LiveStack(pipeline=pipeline, engine=engine)


def load_recognizer(checkpoint: Optional[str], device: str = "cpu"):
    """
    Load a checkpoint if one is given, otherwise run without a recogniser.

    Missing torch or a missing file is reported and downgraded to
    kinematics-only rather than raised: the deterministic branch is a
    genuinely useful fallback, and a demo that refuses to start because a
    model file moved is worse than one that counts reps without naming
    the exercise.
    """
    if not checkpoint:
        return None
    try:
        from motion.recognizer import NeuralExerciseRecognizer

        recognizer = NeuralExerciseRecognizer.load(checkpoint, device=device)
        warmup_ms = recognizer.warmup() * 1000.0
        print(f"loaded {checkpoint} (warmup {warmup_ms:.0f} ms)")
        return recognizer
    except Exception as exc:
        print(f"could not load {checkpoint}: {exc}\n  continuing with kinematics only")
        return None


def build_camera(config: AppConfig):
    from camera.capture import Camera

    source = config.video_file if config.is_video_source else config.camera_id
    return Camera(source=source, width=config.width, height=config.height, fps=config.fps).start()


def build_detector(config: AppConfig):
    from pose.detector import PoseDetector

    return PoseDetector(model_path=config.pose_model_path)


def profile(
    seconds: float = 15.0,
    checkpoint: Optional[str] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    source: Optional[Any] = None,
    detector: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Drive the pipeline and report per-stage latency.

    Defaults to the synthetic camera so this runs anywhere. Pass a real
    source and detector to profile the real thing — the numbers from the
    mock camera are only meaningful for the stages after capture.
    """
    if source is None:
        from camera.sources import MockCameraSource

        source = MockCameraSource(width=640, height=480).start()
        # The mock source has no frame pacing, so cap it at a realistic
        # camera rate — otherwise capture spins at thousands of fps and
        # the drop rate reads as 99% for a pipeline that is fine.
        pipeline_config = pipeline_config or PipelineConfig()
        if pipeline_config.max_capture_fps is None:
            pipeline_config = replace(pipeline_config, max_capture_fps=30.0)
    if detector is None:
        detector = build_detector(AppConfig())

    stack = build_live_stack(
        source, detector, load_recognizer(checkpoint),
        pipeline_config=pipeline_config or PipelineConfig(),
    )

    stack.start()
    try:
        time.sleep(seconds)
        report = stack.pipeline.profile()
    finally:
        stack.stop()
        # MediaPipe holds native resources and complains at interpreter
        # shutdown if they are still open.
        if hasattr(detector, "close"):
            try:
                detector.close()
            except Exception:
                pass

    print(stack.pipeline.latency.report())
    print()
    for key in ("capture_fps", "pose_fps", "inference_fps", "drop_rate", "pose_failure_rate"):
        print(f"{key:<20} {report[key]:.3f}")
    if report["errors"]:
        print("\nrecent errors:")
        for err in report["errors"]:
            print(f"  {err}")
    return report


def run_live(config: AppConfig, checkpoint: Optional[str] = None,
             display: bool = True, max_seconds: Optional[float] = None) -> Dict[str, Any]:
    """
    Run against a camera or video file, drawing the existing overlay.

    The consumer loop deliberately does no work beyond drawing: every
    stage that could stall it is on another thread, which is what keeps
    the displayed frame rate independent of the model's cost.
    """
    import cv2

    from ui.overlay import draw_skeleton, draw_stats_panel

    camera = build_camera(config)
    detector = build_detector(config)
    stack = build_live_stack(camera, detector, load_recognizer(checkpoint))

    started = time.time()
    stack.start()
    try:
        while True:
            pipeline_state = stack.pipeline.state()
            frame = pipeline_state.raw_frame
            state = stack.state()

            if display and frame is not None:
                canvas = frame.copy()
                draw_skeleton(canvas, pipeline_state.pose_frame)
                draw_stats_panel(canvas, state, state.get("pose_fps", 0.0),
                                 stack.pipeline.latency.stats("inference")["p95_ms"])
                cv2.imshow("Fitness CV - motion", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            else:
                time.sleep(1.0 / max(1, config.fps))

            if max_seconds is not None and (time.time() - started) > max_seconds:
                break
    finally:
        stack.stop()
        if display:
            cv2.destroyAllWindows()

    return stack.pipeline.profile()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run or profile the learned motion pipeline")
    p.add_argument("--checkpoint", default=None, help="trained motion checkpoint (.pt)")
    p.add_argument("--video", default=None, help="run against a video file instead of a camera")
    p.add_argument("--camera-id", type=int, default=0, dest="camera_id")
    p.add_argument("--profile", action="store_true", help="synthetic-camera latency profile, no display")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--inference-interval", type=float, default=0.15, dest="inference_interval")
    p.add_argument("--window-frames", type=int, default=48, dest="window_frames")
    p.add_argument("--no-display", action="store_true", dest="no_display")
    p.add_argument("--json", default=None, help="write the profile report to this path")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    pipeline_config = PipelineConfig(
        window_frames=args.window_frames,
        inference_interval=args.inference_interval,
    )

    if args.profile:
        report = profile(args.seconds, args.checkpoint, pipeline_config)
    else:
        config = AppConfig(camera_id=args.camera_id, video_file=args.video)
        report = run_live(config, args.checkpoint, display=not args.no_display,
                          max_seconds=args.seconds if args.no_display else None)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
