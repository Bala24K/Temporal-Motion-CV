"""Benchmark pose estimators on YOUR footage:  python -m ml.benchmark_pose video.mp4 [--gt clip_id --store DIR] mediapipe rtmpose yolo-pose
Reports FPS, p50/p95 latency, dropout rate, jitter and (if ground-truth landmarks exist) error/PCK.
Don't replace MediaPipe until this says another model is better on your cameras and exercises."""
import json
import sys

import cv2

from motion_engine.pose import PoseUnavailable, benchmark, make_estimator


def frames_from_video(path, limit=300):
    cap = cv2.VideoCapture(path)
    n = 0
    while n < limit:
        ok, img = cap.read()
        if not ok:
            break
        yield img, None
        n += 1
    cap.release()


def main(argv):
    video, names = argv[0], argv[1:] or ["mediapipe"]
    out = []
    for n in names:
        try:
            with make_estimator(n) as est:
                out.append(benchmark(est, frames_from_video(video)).to_dict())
        except PoseUnavailable as e:
            out.append({"name": n, "unavailable": str(e)})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
