"""
Example API Client — demonstrates calling the Motion ST-GCN Inference API.

Usage:
    python examples/api_client.py                         # default localhost:8000
    python examples/api_client.py --host 192.168.1.50     # remote server
    python examples/api_client.py --port 8000             # custom port

This script generates a synthetic multi-joint motion sequence (e.g. bicep curl trajectory),
sends it to /predict and /analyze on the Motion ST-GCN API, and prints rich outputs
including calibrated confidence, abstention status, learned phase transitions,
rep segmentation, and spatial joint attention.
"""

import argparse
import json
import sys
import time
import math

try:
    import requests
except ImportError:
    print("This example requires the 'requests' library:")
    print("  pip install requests")
    sys.exit(1)


def make_bicep_curl_sequence(num_frames: int = 45) -> list:
    """Builds a synthetic pose sequence simulating a bicep curl rep.

    The wrists flex upward toward the shoulders over time and return down.
    """
    frames = []
    for t in range(num_frames):
        # Sine wave phase over frames: 0 -> pi (up) -> 2pi (down)
        phase_rad = (2.0 * math.pi * t) / max(1, num_frames - 1)
        curl_factor = (1.0 - math.cos(phase_rad)) / 2.0  # 0.0 at rest, 1.0 at peak flexion

        wrist_y = 0.65 - 0.25 * curl_factor
        elbow_y = 0.48 + 0.02 * curl_factor

        frames.append({
            "landmarks": {
                "nose":            {"x": 0.50, "y": 0.20, "z": 0.0, "visibility": 0.99},
                "left_shoulder":   {"x": 0.40, "y": 0.32, "z": 0.0, "visibility": 0.99},
                "right_shoulder":  {"x": 0.60, "y": 0.32, "z": 0.0, "visibility": 0.99},
                "left_elbow":      {"x": 0.36, "y": elbow_y, "z": 0.0, "visibility": 0.98},
                "right_elbow":     {"x": 0.64, "y": elbow_y, "z": 0.0, "visibility": 0.98},
                "left_wrist":      {"x": 0.34, "y": wrist_y, "z": -0.05 * curl_factor, "visibility": 0.98},
                "right_wrist":     {"x": 0.66, "y": wrist_y, "z": -0.05 * curl_factor, "visibility": 0.98},
                "left_hip":        {"x": 0.45, "y": 0.65, "z": 0.0, "visibility": 0.99},
                "right_hip":       {"x": 0.55, "y": 0.65, "z": 0.0, "visibility": 0.99},
                "left_knee":       {"x": 0.45, "y": 0.82, "z": 0.0, "visibility": 0.99},
                "right_knee":      {"x": 0.55, "y": 0.82, "z": 0.0, "visibility": 0.99},
                "left_ankle":      {"x": 0.45, "y": 0.96, "z": 0.0, "visibility": 0.99},
                "right_ankle":     {"x": 0.55, "y": 0.96, "z": 0.0, "visibility": 0.99},
            },
            "timestamp": float(t) * 0.0333,  # 30 FPS
        })
    return frames


def main():
    parser = argparse.ArgumentParser(description="Motion ST-GCN API Client Example")
    parser.add_argument("--host", default="localhost", help="API server hostname")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Motion ST-GCN API Client — targeting {base_url}\n")

    # ── 1. Health Check ──────────────────────────────────────────────
    print("=" * 60)
    print("  1. GET /health")
    print("=" * 60)
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Body:   {json.dumps(resp.json(), indent=2)}")
    except requests.ConnectionError:
        print(f"  ERROR: Cannot connect to {base_url}")
        print("  Is the server running? Start it with:")
        print("    uvicorn api.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # ── 2. Model Metadata ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  2. GET /model")
    print("=" * 60)
    resp = requests.get(f"{base_url}/model", timeout=5)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Model:        {data['model_name']} ({data['architecture']})")
        print(f"  Device:       {data['device']}")
        print(f"  Classes:      {data['classes']}")
        print(f"  Calibrated:   {data['temperature_calibrated']} (T={data['temperature']:.3f})")
        print(f"  Abstention:   {data['abstention_threshold']}")

    # ── 3. Predict ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  3. POST /predict (ST-GCN Neural Classifier)")
    print("=" * 60)
    sequence = make_bicep_curl_sequence(45)
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/predict",
        json={"sequence": sequence},
        timeout=10,
    )
    roundtrip_ms = (time.perf_counter() - t0) * 1000
    print(f"  Status:       {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Exercise:     {data['class']}")
        print(f"  Confidence:   {data['confidence']:.3f} (calibrated: {data['calibrated']})")
        print(f"  Abstained:    {data['abstained']}")
        print(f"  Frames:       {data['frame_count']}")
        print(f"  Server Time:  {data['latency_ms']:.2f} ms")
        print(f"  Client Time:  {roundtrip_ms:.2f} ms")
        print(f"  Probabilities:")
        for cls, prob in sorted(data["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {cls:<20} {prob:.3f}  {bar}")

    # ── 4. Analyze ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  4. POST /analyze (Hybrid Neural + Kinematic Pipeline)")
    print("=" * 60)
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/analyze",
        json={"sequence": sequence},
        timeout=10,
    )
    roundtrip_ms = (time.perf_counter() - t0) * 1000
    print(f"  Status:       {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Exercise:     {data['exercise']} ({data['exercise_id']})")
        print(f"  Confidence:   {data['confidence']:.3f}")
        print(f"  Abstained:    {data['abstained']}")
        print(f"  Reps Count:   {data['rep_count']}")
        print(f"  Active Phase: {data['phase']}")
        print(f"  Rep Segments: {len(data['rep_segments'])}")
        for idx, seg in enumerate(data['rep_segments']):
            print(f"    Rep #{seg['rep_number']}: frames [{seg['start_frame']} -> {seg['inflection_frame']} -> {seg['end_frame']}] "
                  f"duration={seg['duration_s']:.2f}s (concentric={seg['concentric_duration_s']:.2f}s, eccentric={seg['eccentric_duration_s']:.2f}s)")

        if data.get("joint_attention"):
            print(f"  Top Joint Attention:")
            top_joints = sorted(data["joint_attention"].items(), key=lambda x: -x[1])[:5]
            for jname, att in top_joints:
                print(f"    {jname:<18} {att:.4f}")

        print(f"  Form Feedback:")
        for fb in data.get("form_feedback", []):
            icon = "✓" if fb["status"] == "good" else "⚠"
            print(f"    {icon} {fb['message']}")

    print(f"\n{'=' * 60}")
    print("  Completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
