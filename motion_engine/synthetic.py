"""Synthetic pose generator (forward kinematics, 2D image coordinates).

Used for: tests without camera/data, the UI demo mode, and sanity-checking new
ExerciseDefinitions before real footage exists. It is NOT training data and
makes no claim about real-world appearance (e.g. it ignores foreshortening).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

T_, UA, FA, TH, SH, SW, HW = 0.22, 0.13, 0.12, 0.19, 0.19, 0.13, 0.09
CX, GROUND = 0.5, 0.93


def _lerp(a, b, p):
    return a + (b - a) * p


def _pose(elbow_int=170, arm_elev=8, bend=-1, knee_int=172, lean=0.0, scale=1.0,
          cx=CX, sway=0.0) -> Dict[str, Tuple[float, float, float]]:
    s = scale
    th, sh, t, ua, fa, sw, hw = TH * s, SH * s, T_ * s, UA * s, FA * s, SW * s, HW * s
    K = math.radians(knee_int)
    d = math.sqrt(th**2 + sh**2 - 2 * th * sh * math.cos(K))
    beta = math.acos(np.clip((sh**2 + d**2 - th**2) / (2 * sh * d), -1, 1))
    hip_y = GROUND - d
    P: Dict[str, Tuple[float, float, float]] = {}
    for side, sgn in (("left", -1), ("right", 1)):       # sgn: outward x direction
        ax = cx + sgn * hw / 2
        # knee: rotate ankle->hip (straight up) forward (+x) by beta
        kx, ky = ax + sh * math.sin(beta), GROUND - sh * math.cos(beta)
        P[f"{side}_ankle"] = (ax, GROUND, 1.0)
        P[f"{side}_knee"] = (kx, ky, 1.0)
        P[f"{side}_hip"] = (ax, hip_y, 1.0)
    L = math.radians(lean)
    hc = np.array([cx, hip_y])
    sc = hc + t * np.array([math.sin(L), -math.cos(L)])
    P["nose"] = (sc[0] + 0.05 * s * math.sin(L), sc[1] - 0.08 * s, 1.0)
    e, f = math.radians(arm_elev), math.radians(180 - elbow_int)
    for side, sgn in (("left", -1), ("right", 1)):
        sp = sc + np.array([sgn * sw / 2, 0.0])
        ep = sp + ua * np.array([sgn * math.sin(e), math.cos(e)])
        a2 = e + bend * f * 1.0
        wp = ep + fa * np.array([sgn * math.sin(a2), math.cos(a2)])
        P[f"{side}_shoulder"] = (sp[0], sp[1], 1.0)
        P[f"{side}_elbow"] = (ep[0], ep[1], 1.0)
        P[f"{side}_wrist"] = (wp[0], wp[1], 1.0)
    return P


# exercise id -> function(progress 0..1 [rest->peak], fault) -> _pose kwargs
def _curl(p, fault):
    return dict(elbow_int=_lerp(170, 40, p), arm_elev=8 + (30 * p if fault == "drift" else 0),
                bend=-1, lean=(25 * p if fault == "lean" else 3))


def _squat(p, fault):
    return dict(knee_int=_lerp(172, 80, p), lean=8 + 25 * p + (20 * p if fault == "lean" else 0),
                arm_elev=15, elbow_int=165)


EX = {
    "bicep_curl": _curl, "hammer_curl": _curl,
    "squat": _squat,
    "goblet_squat": lambda p, f: {**_squat(p, f), "elbow_int": 45, "arm_elev": 20, "bend": -1},
    "lateral_raise": lambda p, f: dict(arm_elev=_lerp(8, 92, p), elbow_int=(_lerp(168, 115, p) if f == "bent_arms" else 168),
                                       lean=(15 * p if f == "lean" else 2)),
    "front_raise": lambda p, f: dict(arm_elev=_lerp(8, 92, p), elbow_int=168, lean=(15 * p if f == "lean" else 2)),
    "shoulder_press": lambda p, f: dict(elbow_int=_lerp(90, 140 if f == "no_lockout" else 172, p), arm_elev=_lerp(85, 168, p),
                                        bend=+1, lean=(18 * p if f == "lean" else 2)),
    "overhead_triceps_extension": lambda p, f: dict(elbow_int=_lerp(40, 170, p), arm_elev=165, bend=+1,
                                                   lean=(15 * p if f == "lean" else 2)),
    "romanian_deadlift": lambda p, f: dict(knee_int=165, lean=_lerp(3, 80, p), arm_elev=2, elbow_int=175),
}


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def synth_sequence(exercise_id: str, reps: int = 5, fps: float = 30.0, fault: Optional[str] = None,
                   depth: float = 1.0, rep_seconds: float = 2.4, noise: float = 0.002, seed: int = 0,
                   scale: float = 1.0, lead_in: float = 0.6, left_right_skew: float = 0.0
                   ) -> Tuple[List[dict], List[float]]:
    """Returns (frames, timestamps). `fault` in {None,'shallow','fast','drift','lean','asym','no_lockout'}.
    asym: the left arm only travels 65% as far as the right. no_lockout: shoulder press stops at 140 deg elbow."""
    if exercise_id not in EX:
        raise KeyError(f"no synthetic model for {exercise_id}")
    rng = np.random.default_rng(seed)
    if fault == "shallow":
        depth = min(depth, 0.45)
    if fault == "fast":
        rep_seconds = min(rep_seconds, 0.75)
    n_rest = int(lead_in * fps)
    n_rep = int(rep_seconds * fps)
    prog = [0.0] * n_rest
    for _ in range(reps):
        x = np.arange(n_rep) / n_rep
        tri = 1 - np.abs(2 * x - 1)
        prog += list(smoothstep(tri) * depth)
        prog += [0.0] * int(0.3 * fps)
    frames, ts = [], []
    for i, p in enumerate(prog):
        kw = EX[exercise_id](p, fault)
        pose = _pose(scale=scale, **kw)
        if fault == "asym":                                  # left arm lags: take it from a pose at 65% progress
            lag = _pose(scale=scale, **EX[exercise_id](0.65 * p, None))
            pose.update({k: v for k, v in lag.items() if k in ("left_elbow", "left_wrist")})
        frame = {}
        for k, (x, y, v) in pose.items():
            frame[k] = (x + rng.normal(0, noise), y + rng.normal(0, noise), v)
        if left_right_skew:   # make the left arm lag: crude asymmetry for tests
            frame["left_wrist"] = (frame["left_wrist"][0], frame["left_wrist"][1] + left_right_skew * p, 1.0)
        frames.append(frame)
        ts.append(i / fps)
    return frames, ts


def press_with_rack(reps: int = 5, fps: float = 30.0, fault: Optional[str] = None, seed: int = 0,
                    noise: float = 0.002, rep_seconds: float = 2.4) -> Tuple[List[dict], List[float]]:
    """A whole shoulder-press set as a user does it: arms hanging -> dumbbells raised to the shoulders -> presses ->
    dumbbells lowered back to the sides. Hanging and locked-out arms have the same elbow angle (~170 deg), so a
    counter driven by the elbow angle counts the final lowering as an extra rep."""
    rng = np.random.default_rng(seed)
    top_elbow = 140 if fault == "no_lockout" else 172
    down, rack, top = (170.0, 8.0), (90.0, 85.0), (float(top_elbow), 168.0)

    def go(a, b, sec):
        n = int(sec * fps)
        return [(a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w) for w in smoothstep(np.arange(n) / n)]
    path = [down] * int(0.7 * fps) + go(down, rack, 1.0) + [rack] * int(0.4 * fps)
    for _ in range(reps):
        path += go(rack, top, rep_seconds / 2) + go(top, rack, rep_seconds / 2) + [rack] * int(0.3 * fps)
    path += go(rack, down, 1.0) + [down] * int(1.0 * fps)
    frames = []
    for e, v in path:
        pose = _pose(elbow_int=e, arm_elev=v, bend=+1 if v > 45 else -1, lean=2)
        frames.append({k: (x + rng.normal(0, noise), y + rng.normal(0, noise), c) for k, (x, y, c) in pose.items()})
    return frames, [i / fps for i in range(len(frames))]


def synth_standing(n: int = 45, fps: float = 30.0, scale: float = 1.0, noise: float = 0.001, seed: int = 1):
    """Static neutral stance, for body calibration tests."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        pose = _pose(scale=scale, arm_elev=10, elbow_int=175)
        frames.append({k: (x + rng.normal(0, noise), y + rng.normal(0, noise), v) for k, (x, y, v) in pose.items()})
    return frames, [i / fps for i in range(n)]


def yaw_project(frame: dict, theta_deg: float) -> dict:
    """Rotate a frontal-plane pose about the body's vertical axis and project orthographically.
    Exact for frontal-plane motion; sagittal motion would also gain depth (not modelled)."""
    c = math.cos(math.radians(theta_deg))
    cx = (frame["left_hip"][0] + frame["right_hip"][0]) / 2
    return {k: (cx + (v[0] - cx) * c, v[1], v[2]) for k, v in frame.items()}


M_PER_UNIT = 2.0          # synthetic body ~0.8 image-heights tall ~ 1.6 m


def with_world(frame: dict, yaw_deg: float = 0.0, m_per_unit: float = M_PER_UNIT) -> dict:
    """Add MediaPipe-style world landmarks ("w:<joint>": [X, Y, Z] metres, hip-centred, y down) for an isotropic
    frontal pose rotated `yaw_deg` about the vertical axis. Lets tests and the demo exercise the 3D facing path."""
    hx = (frame["left_hip"][0] + frame["right_hip"][0]) / 2
    hy = (frame["left_hip"][1] + frame["right_hip"][1]) / 2
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    out = dict(frame)
    for k, v in frame.items():
        if k.startswith("_") or k.startswith("w:"):
            continue
        X, Y = (v[0] - hx) * m_per_unit, (v[1] - hy) * m_per_unit
        out["w:" + k] = [X * c, Y, X * s]
    return out


def standing_stage(stage: str, n: int = 40, scale: float = 1.0, seed: int = 1, world: bool = True) -> List[dict]:
    """Calibration-stage frames: front / side (85 deg) / arms_out (T) / arms_up, optionally with world landmarks."""
    if stage in ("front", "side"):
        base, _ = synth_standing(n, scale=scale, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        kw = dict(arm_elev=90, elbow_int=176) if stage == "arms_out" else dict(arm_elev=170, elbow_int=175)
        base = [{k: (x + rng.normal(0, 0.001), y + rng.normal(0, 0.001), v) for k, (x, y, v) in _pose(scale=scale, **kw).items()}
                for _ in range(n)]
    yaw = 85.0 if stage == "side" else 0.0
    out = []
    for f in base:
        g = yaw_project(f, yaw) if yaw else dict(f)          # 2D: orthographic projection of the turned body
        if world:                                            # 3D: the unprojected body rotated by the same yaw
            g.update({k: v for k, v in with_world(f, yaw).items() if k.startswith("w:")})
        out.append(g)
    return out
