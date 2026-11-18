"""Calibration v3: guided, window-based body calibration -> BodyProfile (approximate personal proportions).

Protocol (docs/POSE_CALIBRATION_AUDIT.md section 6):
  front     facing the camera, arms relaxed        (required)  -> torso, shoulder/hip width, legs, setup fingerprint
  arms_out  facing the camera, arms out level (T)  (required for full coverage) -> arm segments in the image plane
  side      a quarter turn                          (optional)  -> sagittal-view check; far limbs are NOT required
  arms_up   legacy v2 stage, still accepted in place of arms_out

Each stage is judged on a contiguous window (`assess_window`): per-joint visibility for the joints that stage
needs, in-frame with a direction, distance, centring, facing (3D yaw from world landmarks when the client sends
them, isotropic shoulder ratio otherwise), the pose itself, and stillness. Every failure names what to fix.

Output is *relative body proportions for normalisation*. Centimetres are only derived when the user gives their
height, and carry that assumption. Quality = min(stability, confidence, coverage); plausibility only warns.

Thresholds marked PROVISIONAL were set from synthetic data and the QEVD pose study; tune them on recordings
(docs/TEST_AND_EVALUATION_PLAN.md section 3) before treating them as validated.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .skeleton import (BodyProfile, IMAGE_KEY, J, REAL_JOINTS, SEGMENTS, Frame, frames_to_array,
                       image_aspect, measure_body)

STAGES = ("front", "side", "arms_out", "arms_up")       # order used when reporting captured stages
GUIDED_STAGES = ("front", "arms_out", "side")           # what the wizard asks for, in order
OPTIONAL_STAGES = ("side",)
ARM_STAGES = ("arms_out", "arms_up")
WORLD_PREFIX = "w:"                                     # world landmarks travel as "w:<joint>": [X, Y, Z] (metres)
NOSE_HEIGHT_FRACTION = 0.92
REQUIRED = list(REAL_JOINTS)
# typical limb length / stature (Drillis & Contini style averages); used only as a plausibility *warning*
TYPICAL = {"upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shin": 0.246, "shoulder_width": 0.259, "torso": 0.30}
TOLERANCE = 0.40            # +-40% of typical before we call a measurement implausible
FRONT_MIN_RATIO = 0.50      # isotropic shoulder width / torso when no 3D landmarks (QEVD facing median ~0.69)
SIDE_MAX_RATIO = 0.40
FRONT_MAX_YAW = 30.0        # deg, from 3D shoulder/hip vectors                                   PROVISIONAL
SIDE_MIN_YAW = 55.0         #                                                                      PROVISIONAL
EDGE = 0.02                 # in-frame margin, fraction of the image
FRAME_MIN_VIS = 0.5         # a required joint below this in a frame fails that frame
WINDOW_MIN_VIS = 0.6        # median visibility of each required joint over a window               PROVISIONAL
BODY_HEIGHT_FRAC = (0.35, 0.95)   # nose->ankle span as a fraction of image height                    PROVISIONAL
CENTRE_X = (0.2, 0.8)       # hip centre, fraction of image width
MAX_JITTER = 0.03           # median |2nd difference| per joint, torso lengths per frame          PROVISIONAL
MAX_DRIFT = 0.15            # hip-centre travel across a window, torso lengths                     PROVISIONAL
MIN_WINDOW = 20             # frames (~0.7 s at 30 fps)
OK_FRACTION = 0.8           # share of window frames that must pass the per-frame checks
QUALITY_GOOD, QUALITY_USABLE = 0.8, 0.6
INSTRUCTIONS = {
    "front": "Face the camera, whole body in view, arms relaxed by your sides. Hold still.",
    "arms_out": "Face the camera and hold both arms straight out to the sides at shoulder height.",
    "side": "Turn a quarter turn so your side faces the camera, arms relaxed. Hold still.",
    "arms_up": "Face the camera and raise both arms overhead.",
}
_PRETTY = lambda j: j.replace("_", " ")


def _torso(a):
    return np.linalg.norm(a[:, J["shoulder_center"], :2] - a[:, J["hip_center"], :2], axis=1)


def _shoulder_ratio(a):
    return np.linalg.norm(a[:, J["left_shoulder"], :2] - a[:, J["right_shoulder"], :2], axis=1) / _torso(a)


def _world(frame: Frame, joint: str) -> Optional[np.ndarray]:
    v = frame.get(WORLD_PREFIX + joint)
    if v is None or len(v) < 3:
        return None
    a = np.asarray(v[:3], float)
    return a if np.all(np.isfinite(a)) else None


def facing_yaw_deg(frame: Frame) -> Optional[float]:
    """Body yaw relative to the camera (0 = facing, 90 = side-on) from 3D shoulder and hip vectors, or None
    when the frame carries no world landmarks. Independent of the user's proportions (unlike a width ratio)."""
    yaws = []
    for a, b in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")):
        pa, pb = _world(frame, a), _world(frame, b)
        if pa is not None and pb is not None:
            d = pa - pb
            yaws.append(math.degrees(math.atan2(abs(d[2]), abs(d[0]) + 1e-9)))
    return float(np.mean(yaws)) if yaws else None


def stage_joints(stage: str, frame: Frame) -> List[str]:
    """Joints a stage needs. Side-on, the far limbs are occluded by design, so only the visible side counts."""
    if stage != "side":
        return list(REQUIRED)
    chain = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    vis = {s: np.mean([float(frame.get(f"{s}_{c}", (0, 0, 0))[2]) if len(frame.get(f"{s}_{c}", ())) > 2 else 1.0
                       for c in chain]) for s in ("left", "right")}
    near = max(vis, key=vis.get)
    return ["nose"] + [f"{near}_{c}" for c in chain]


def _check(id_, ok, message=None):
    return {"id": id_, "ok": bool(ok), "message": None if ok else message}


def check_stage_frame(stage: str, frame: Frame) -> Dict:
    """Guidance for one frame: {ok, message (first failing check), checks: [...], metrics}."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    need = stage_joints(stage, frame)
    checks, m = [], {}

    # 1. visibility, naming the weakest joint
    vis = {j: (float(frame[j][2]) if len(frame[j]) > 2 else 1.0) if frame.get(j) is not None else 0.0 for j in need}
    weak = min(vis, key=vis.get)
    m["min_visibility"] = round(vis[weak], 3)
    if vis[weak] < FRAME_MIN_VIS:
        legs = any(k in weak for k in ("knee", "ankle"))
        checks.append(_check("visibility", False, f"We can't see your {_PRETTY(weak)} clearly - "
                                                   + ("step back so your whole body is in view." if legs else
                                                      "improve the lighting or avoid baggy sleeves.")))
    else:
        checks.append(_check("visibility", True))

    # 2. in frame, with a direction (raw [0,1] coordinates: this is about the picture, not geometry)
    out = [(j, frame[j]) for j in need if frame.get(j) is not None and vis[j] >= FRAME_MIN_VIS
           and not (EDGE <= frame[j][0] <= 1 - EDGE and EDGE <= frame[j][1] <= 1 - EDGE)]
    if out:
        j, v = out[0]
        hint = ("step back" if v[1] > 1 - EDGE or v[1] < EDGE else "move toward the centre")
        checks.append(_check("in_frame", False, f"Your {_PRETTY(j)} is out of frame - {hint}."))
    else:
        checks.append(_check("in_frame", True))

    full_ok = all(c["ok"] for c in checks)
    a = frames_to_array([frame])
    if full_ok:
        # 3. distance: body span as a fraction of image height (y is already a fraction of height)
        foot = [frame[k][1] for k in ("left_ankle", "right_ankle") if k in need and frame.get(k) is not None]
        if foot and frame.get("nose") is not None:
            span = float(max(foot) - frame["nose"][1])
            m["body_height_frac"] = round(span, 3)
            if span < BODY_HEIGHT_FRAC[0]:
                checks.append(_check("distance", False, "Come a little closer to the camera."))
            elif span > BODY_HEIGHT_FRAC[1]:
                checks.append(_check("distance", False, "Step back a little."))
            else:
                checks.append(_check("distance", True))
        # 4. centring
        hips = [frame[k][0] for k in ("left_hip", "right_hip") if frame.get(k) is not None]
        if hips:
            cx = float(np.mean(hips))
            checks.append(_check("centered", CENTRE_X[0] <= cx <= CENTRE_X[1], "Move toward the centre of the frame."))
        # 5. facing: 3D yaw when available, isotropic shoulder ratio otherwise
        yaw = facing_yaw_deg(frame)
        sr = float(_shoulder_ratio(a)[0])
        m["shoulder_ratio"] = round(sr, 3)
        if yaw is not None:
            m["yaw_deg"] = round(yaw, 1)
            facing, side_on = yaw <= FRONT_MAX_YAW, yaw >= SIDE_MIN_YAW
        else:
            # far shoulder hidden (ratio unmeasurable) is itself evidence of a side-on pose
            facing, side_on = sr >= FRONT_MIN_RATIO, (not np.isfinite(sr)) or sr <= SIDE_MAX_RATIO
        if stage == "side":
            checks.append(_check("facing", side_on, "Turn further sideways."))
        else:
            checks.append(_check("facing", facing, "Turn to face the camera."))
        # 6. the pose this stage is for
        checks.append(_pose_check(stage, a))

    first = next((c for c in checks if not c["ok"]), None)
    return {"ok": first is None, "message": first["message"] if first else None, "checks": checks, "metrics": m}


def _pose_check(stage: str, a: np.ndarray) -> Dict:
    y = lambda j: a[0, J[j], 1]
    if stage == "front":
        hanging = y("left_wrist") > y("left_shoulder") and y("right_wrist") > y("right_shoulder")
        return _check("pose", hanging, "Let your arms hang by your sides.")
    if stage == "arms_up":
        up = y("left_wrist") < y("left_shoulder") and y("right_wrist") < y("right_shoulder")
        return _check("pose", up, "Raise both arms overhead.")
    if stage == "arms_out":
        t = float(_torso(a)[0])
        ok = True
        for s in ("left", "right"):
            sh, el, wr = (a[0, J[f"{s}_{p}"], :2] for p in ("shoulder", "elbow", "wrist"))
            level = abs(wr[1] - sh[1]) < 0.35 * t and abs(el[1] - sh[1]) < 0.35 * t
            v1, v2 = sh - el, wr - el
            straight = np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))) >= 140
            outward = abs(wr[0] - sh[0]) > abs(el[0] - sh[0]) > 0.15 * t
            ok = ok and level and straight and outward
        return _check("pose", ok, "Hold both arms straight out to the sides, level with your shoulders.")
    return _check("pose", True)


def assess_window(stage: str, frames: Sequence[Frame]) -> Dict:
    """Judge a contiguous window of frames for one stage: per-frame checks plus stillness.
    Returns {ok, message, checks, metrics, frames_ok}."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    frames = list(frames)
    if len(frames) < MIN_WINDOW:
        return {"ok": False, "message": "Hold still for a moment...", "checks": [], "metrics": {"frames": len(frames)},
                "frames_ok": 0}
    per = [check_stage_frame(stage, f) for f in frames]
    ok_idx = [i for i, r in enumerate(per) if r["ok"]]
    frac = len(ok_idx) / len(frames)
    ids = [c["id"] for c in per[0]["checks"]] + [c["id"] for r in per for c in r["checks"] if c["id"] not in
                                                 [x["id"] for x in per[0]["checks"]]]
    agg = []
    for cid in dict.fromkeys(ids):
        res = [c for r in per for c in r["checks"] if c["id"] == cid]
        bad = [c["message"] for c in res if not c["ok"]]
        agg.append(_check(cid, len(bad) <= (1 - OK_FRACTION) * len(frames),
                          Counter(bad).most_common(1)[0][0] if bad else None))
    metrics = {"frames": len(frames), "ok_fraction": round(frac, 3)}
    if frac < OK_FRACTION:
        first = next((c for c in agg if not c["ok"]), None)
        msg = first["message"] if first else Counter(r["message"] for r in per if r["message"]).most_common(1)[0][0]
        return {"ok": False, "message": msg, "checks": agg, "metrics": metrics, "frames_ok": len(ok_idx)}

    good = [frames[i] for i in ok_idx]
    need = stage_joints(stage, good[0])
    arr = frames_to_array(good)
    if stage == "side":                       # midpoints are NaN when the far side is occluded: use the near side
        near = need[1].split("_")[0]
        torso = float(np.nanmedian(np.linalg.norm(arr[:, J[f"{near}_shoulder"], :2] - arr[:, J[f"{near}_hip"], :2], axis=1)))
        hip = arr[:, J[f"{near}_hip"], :2]
    else:
        torso = float(np.nanmedian(_torso(arr)))
        hip = arr[:, J["hip_center"], :2]
    idx = [J[j] for j in need]
    xy = arr[:, idx, :2]
    jit = np.nanmedian(np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=2), axis=0) / torso if len(xy) >= 3 else np.zeros(len(idx))
    # drift = sustained displacement (first third vs last third, medians), so frame-to-frame noise is not "movement"
    k = max(1, len(hip) // 3)
    drift = float(np.linalg.norm(np.nanmedian(hip[-k:], axis=0) - np.nanmedian(hip[:k], axis=0)) / torso) if np.isfinite(hip).any() else 0.0
    vis = {j: float(np.median([float(f[j][2]) if len(f[j]) > 2 else 1.0 for f in good])) for j in need}
    weak = min(vis, key=vis.get)
    metrics.update(jitter=round(float(np.nanmedian(jit)), 4), drift=round(drift, 3), visibility=round(float(np.median(list(vis.values()))), 3),
                   weakest_joint=weak)
    still = float(np.nanmedian(jit)) <= MAX_JITTER and drift <= MAX_DRIFT
    agg.append(_check("still", still, "Hold still for two seconds." if drift <= MAX_DRIFT else
                      "You (or the camera) moved - stay in place and hold still."))
    agg.append(_check("confidence", vis[weak] >= WINDOW_MIN_VIS,
                      f"Your {_PRETTY(weak)} is hard to track - improve the lighting or wear fitted sleeves."))
    first = next((c for c in agg if not c["ok"]), None)
    return {"ok": first is None, "message": first["message"] if first else None, "checks": agg, "metrics": metrics,
            "frames_ok": len(ok_idx)}


def setup_fingerprint(frames: Sequence[Frame]) -> Dict:
    """Camera-setup description from front-stage frames (for invalidation when the camera changes)."""
    img = next((f.get(IMAGE_KEY) for f in frames if f.get(IMAGE_KEY)), None)
    span = [max(f["left_ankle"][1], f["right_ankle"][1]) - f["nose"][1] for f in frames
            if all(f.get(k) is not None for k in ("nose", "left_ankle", "right_ankle"))]
    hips = [((f["left_hip"][0] + f["right_hip"][0]) / 2, (f["left_hip"][1] + f["right_hip"][1]) / 2) for f in frames
            if f.get("left_hip") is not None and f.get("right_hip") is not None]
    rolls = []
    for f in frames:
        la, ra = f.get("left_ankle"), f.get("right_ankle")
        if la is not None and ra is not None:
            k = image_aspect(f)
            rolls.append(math.degrees(math.atan2(la[1] - ra[1], (la[0] - ra[0]) * k + 1e-9)))
    roll = float(np.median(rolls)) if rolls else None
    if roll is not None:                      # left/right order flips the vector; fold to (-90, 90]
        roll = (roll + 90) % 180 - 90
    return {"image": list(img) if img else None, "aspect": round(float(img[0]) / float(img[1]), 4) if img else None,
            "body_height_frac": round(float(np.median(span)), 4) if span else None,
            "hip_center": [round(float(v), 4) for v in np.median(hips, axis=0)] if hips else None,
            "ankle_roll_deg": round(roll, 2) if roll is not None else None, "view": "front"}


def check_setup(profile: Optional[BodyProfile], frames: Sequence[Frame]) -> Dict:
    """Compare a live set's opening frames with the calibration setup. status: ok | uncalibrated | camera_changed |
    distance_changed (informational: angles do not depend on distance, but tracking resolution does)."""
    fp = (profile.setup or {}) if profile else {}
    if not profile or not profile.calibrated or not fp.get("aspect"):
        return {"status": "uncalibrated", "message": "Not calibrated with this camera yet - calibrate in Profile for "
                                                     "personalised analysis."}
    img = next((f.get(IMAGE_KEY) for f in frames if f.get(IMAGE_KEY)), None)
    if img and abs(float(img[0]) / float(img[1]) - fp["aspect"]) / fp["aspect"] > 0.02:
        return {"status": "camera_changed", "message": "Your camera changed since calibration - recalibrate in Profile "
                                                       "for the best accuracy."}
    now = setup_fingerprint([f for f in frames if all(
        f.get(k) is not None and len(f[k]) > 2 and f[k][2] >= FRAME_MIN_VIS for k in ("nose", "left_ankle", "right_ankle"))])
    if now.get("body_height_frac") and fp.get("body_height_frac"):
        rel = abs(now["body_height_frac"] - fp["body_height_frac"]) / fp["body_height_frac"]
        if rel > 0.15:
            return {"status": "distance_changed", "message": "You're at a different distance than when you calibrated - "
                                                             "fine for counting, but keep your whole body in view."}
    return {"status": "ok", "message": None}


@dataclass
class StageResult:
    stage: str
    frames_used: int
    frames_total: int
    ok: bool
    message: str


class CalibrationError(ValueError):
    pass


def _median_ratio(arrs: List[np.ndarray], pairs) -> float:
    vals = []
    for a in arrs:
        t = _torso(a)
        for (p, q) in pairs:
            d = np.linalg.norm(a[:, J[p], :2] - a[:, J[q], :2], axis=1) / t
            vals.append(d[np.isfinite(d)])
    v = np.concatenate(vals) if vals else np.array([])
    return float(np.median(v)) if len(v) else 0.0


def _world_ratios(front: Sequence[Frame], arms: Sequence[Frame]) -> Optional[Dict[str, float]]:
    """The same proportions from MediaPipe's 3D world landmarks (model-estimated depth), for a cross-check."""
    def seg(frames, pairs):
        out = []
        for f in frames:
            for a, b in pairs:
                pa, pb = _world(f, a), _world(f, b)
                if pa is not None and pb is not None:
                    out.append(np.linalg.norm(pa - pb))
        return float(np.median(out)) if out else None

    def torso(frames):
        out = []
        for f in frames:
            p = [_world(f, k) for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
            if all(x is not None for x in p):
                out.append(np.linalg.norm((p[0] + p[1]) / 2 - (p[2] + p[3]) / 2))
        return float(np.median(out)) if out else None

    t = torso(front)
    if not t:
        return None
    src = arms or front
    r = {"upper_arm": seg(src, SEGMENTS["upper_arm"]), "forearm": seg(src, SEGMENTS["forearm"]),
         "thigh": seg(front, SEGMENTS["thigh"]), "shin": seg(front, SEGMENTS["shin"]),
         "shoulder_width": seg(front, [("left_shoulder", "right_shoulder")])}
    return {k: v / t for k, v in r.items() if v} or None


def missing_stages(captured: Sequence[str]) -> List[str]:
    have = set(captured) | ({"arms_out"} if "arms_up" in captured else set())
    return [s for s in GUIDED_STAGES if s not in have]


def calibrate(stages: Dict[str, Sequence[Frame]], height_cm: Optional[float] = None,
              min_frames: int = 8, camera_label: Optional[str] = None) -> tuple[BodyProfile, List[StageResult]]:
    """stages: mapping stage -> frames. 'front' is required; an arm stage completes coverage; 'side' is optional."""
    if "front" not in stages:
        raise CalibrationError("The front stage is required.")
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise CalibrationError(f"Unknown stage(s): {sorted(unknown)}")
    results, good = [], {}
    for st in STAGES:
        if st not in stages:
            continue
        frames = list(stages[st])
        checks = [check_stage_frame(st, f) for f in frames]
        keep = [f for f, c in zip(frames, checks) if c["ok"]]
        ok = len(keep) >= min_frames
        msg = "" if ok else (Counter(c["message"] for c in checks if c["message"]).most_common(1) or [("Not enough clear frames.",)])[0][0]
        results.append(StageResult(st, len(keep), len(frames), ok, msg))
        if ok:
            good[st] = keep
    if "front" not in good:
        raise CalibrationError(next(r.message for r in results if r.stage == "front"))

    base = measure_body(good["front"], height_cm)
    if not base.calibrated:
        raise CalibrationError("Measurement was too unstable - hold still and try again.")
    fa = frames_to_array(good["front"])
    arm_stage = next((s for s in ARM_STAGES if s in good), None)
    arm_arrs = [frames_to_array(good[arm_stage])] if arm_stage else [fa]

    # Arms from the arms-out pose (segments lie in the image plane); legs from the front pose (standing, legs
    # straight). The side view is never used for lengths: its far limbs are guesses.
    ratios = {"upper_arm": _median_ratio(arm_arrs, SEGMENTS["upper_arm"]), "forearm": _median_ratio(arm_arrs, SEGMENTS["forearm"]),
              "thigh": _median_ratio([fa], SEGMENTS["thigh"]), "shin": _median_ratio([fa], SEGMENTS["shin"])}
    torso = base.torso_len
    base.upper_arm, base.forearm = ratios["upper_arm"] * torso, ratios["forearm"] * torso
    base.thigh, base.shin = ratios["thigh"] * torso, ratios["shin"] * torso
    base.shoulder_ratio = float(np.nanmedian(_shoulder_ratio(fa)))
    base.hip_ratio = float(np.nanmedian(np.linalg.norm(fa[:, J["left_hip"], :2] - fa[:, J["right_hip"], :2], axis=1) / _torso(fa)))
    base.shoulder_width, base.hip_width = base.shoulder_ratio * torso, base.hip_ratio * torso

    if height_cm:
        h_units = np.nanmedian((fa[:, [J["left_ankle"], J["right_ankle"]], 1].mean(1)) - fa[:, J["nose"], 1])
        if h_units > 0:
            base.cm_per_torso = float(height_cm * NOSE_HEIGHT_FRACTION / h_units * torso)

    warnings: List[str] = []
    typ = {k: TYPICAL[k] / TYPICAL["torso"] for k in ("upper_arm", "forearm", "thigh", "shin", "shoulder_width")}
    obs = {**ratios, "shoulder_width": base.shoulder_ratio}
    dev = [abs(obs[k] / typ[k] - 1) for k in typ]
    base.plausibility = float(np.clip(1 - np.mean(dev) / TOLERANCE, 0, 1))
    if base.plausibility < 0.6:
        warnings.append("Your measured proportions look unusual - check the camera is level and at about hip height.")

    w3 = _world_ratios(good["front"], good.get(arm_stage, []) if arm_stage else [])
    base.method = "2d+3d" if w3 else "2d"
    if w3:
        common = [k for k in w3 if obs.get(k)]
        agree = 1 - float(np.mean([abs(obs[k] / w3[k] - 1) for k in common])) if common else None
        base.ratio_agreement_3d = round(agree, 3) if agree is not None else None
        if agree is not None and agree < 0.85:
            warnings.append("2D and 3D estimates of your proportions disagree - make sure you face the camera squarely.")

    vis = [float(f[j][2]) if len(f[j]) > 2 else 1.0 for f in good["front"] for j in REQUIRED if f.get(j) is not None]
    base.confidence = float(np.clip((np.median(vis) - WINDOW_MIN_VIS) / (0.95 - WINDOW_MIN_VIS), 0, 1)) if vis else 0.0
    have = set(good)
    base.coverage = 1.0 if arm_stage else (0.8 if "side" in have else 0.6)     # side is optional: bonus only without arms
    if not arm_stage:
        warnings.append("Add the arms-out pose to measure your arms.")
    base.stages = [s for s in STAGES if s in good]
    base.calibration_version = 3
    base.stability = float(base.stability or base.quality)
    base.quality = float(min(base.stability, base.confidence, base.coverage))
    base.status = "good" if base.quality >= QUALITY_GOOD else "usable" if base.quality >= QUALITY_USABLE else "retry"
    if base.status == "retry":
        warnings.append("Calibration quality is low - retake the poses in better light, holding still.")
    base.warnings = warnings
    base.setup = {**setup_fingerprint(good["front"]), **({"label": camera_label} if camera_label else {})}
    base.n_frames = sum(len(v) for v in good.values())
    return base, results

