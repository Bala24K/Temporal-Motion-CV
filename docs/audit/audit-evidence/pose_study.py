"""Real-footage pose study on QEVD clips (read-only). Measures MediaPipe lite latency, landmark visibility,
and the shoulder-width / torso-length ratio in normalized coords (what Kinetic uses) vs pixel coords."""
import json, os, sys, time
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "QEVD-FIT-300k-Part-1") + os.sep
MODEL = os.path.join(ROOT, "Ver-1", "models", "pose_landmarker.task")
IDX = dict(nose=0, ls=11, rs=12, le=13, re=14, lw=15, rw=16, lh=23, rh=24, lk=25, rk=26, la=27, ra=28)
files = sorted(os.listdir(D))[::1900][:40]
rows, lat = [], []
for fi, f in enumerate(files):
    opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    lm = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(D + f)
    W, H = cap.get(3), cap.get(4)
    i = 0
    while i < 90:
        ok, img = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[:, :, ::-1])), int(i * 1000 / 30))
        lat.append((time.perf_counter() - t0) * 1000)
        i += 1
        if not res.pose_landmarks:
            rows.append(dict(f=f, aspect=W / H, det=False)); continue
        p = res.pose_landmarks[0]
        g = {k: (p[v].x, p[v].y, p[v].visibility, p[v].presence) for k, v in IDX.items()}
        def ratio(sx, sy):
            P = {k: np.array([g[k][0] * sx, g[k][1] * sy]) for k in g}
            sc, hc = (P["ls"] + P["rs"]) / 2, (P["lh"] + P["rh"]) / 2
            return float(np.linalg.norm(P["ls"] - P["rs"]) / (np.linalg.norm(sc - hc) + 1e-9))
        vis = [g[k][2] for k in g]
        full = all(v >= 0.5 for v in vis) and all(0.02 <= g[k][0] <= 0.98 and 0.02 <= g[k][1] <= 0.98 for k in g)
        hanging = g["lw"][1] > g["ls"][1] and g["rw"][1] > g["rs"][1]
        rows.append(dict(f=f, aspect=W / H, det=True, vis_min=min(vis), vis_med=float(np.median(vis)),
                         pres_med=float(np.median([g[k][3] for k in g])),
                         sr_norm=ratio(1, 1), sr_px=ratio(W, H), full=full, hanging=hanging))
    cap.release(); lm.close()

lat = np.array(lat[5:])
det = [r for r in rows if r["det"]]
print(f"clips={len(files)} frames={len(rows)} detection_rate={len(det)/len(rows):.3f}")
print(f"lite CPU latency ms: p50={np.percentile(lat,50):.1f} p95={np.percentile(lat,95):.1f} (n={len(lat)})")
print(f"visibility: median-of-min={np.median([r['vis_min'] for r in det]):.2f} median={np.median([r['vis_med'] for r in det]):.2f}; presence median={np.median([r['pres_med'] for r in det]):.2f}")
print(f"frames with all 13 landmarks vis>=0.5 and inside 2% margin: {np.mean([r['full'] for r in det]):.3f}")
for asp in sorted({round(r['aspect'], 2) for r in det}):
    sub = [r for r in det if round(r["aspect"], 2) == asp and r["full"] and r["hanging"]]
    if not sub:
        continue
    n = np.array([r["sr_norm"] for r in sub]); px = np.array([r["sr_px"] for r in sub])
    print(f"aspect {asp}: n={len(sub)} (full-body, wrists below shoulders) shoulder/torso median norm={np.median(n):.3f} px={np.median(px):.3f} | "
          f"pass front gate (>=0.55) norm={np.mean(n>=0.55):.2f} px={np.mean(px>=0.55):.2f}")
