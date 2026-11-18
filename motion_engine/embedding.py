"""Movement descriptors and few-shot family matching.

A descriptor summarises *which body channels move and how much*, independent of
any ExerciseDefinition. Given a clip of an unsupported exercise, ranking known
exercise families by descriptor similarity answers the developer question:
"can existing primitives describe this, or does it need new work?"

This is hand-built (physically interpretable), not learned. Prototypes come
from the synthetic generator until real labelled clips exist, so similarity
scores are triage hints, not recognition. A learned encoder (contrastive /
VideoMAE-style) can replace `descriptor` later without changing callers.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .primitives import compute_signal, smooth
from .skeleton import fill_gaps, frames_to_array, normalize

CHANNELS: Dict[str, dict] = {
    "elbow_flexion": {"primitive": "joint_angle", "joints": ["shoulder", "elbow", "wrist"], "side": "mean"},
    "knee_flexion": {"primitive": "joint_angle", "joints": ["hip", "knee", "ankle"], "side": "mean"},
    "hip_flexion": {"primitive": "joint_angle", "joints": ["shoulder", "hip", "knee"], "side": "mean"},
    "arm_elevation": {"primitive": "segment_vertical_angle", "joints": ["shoulder", "elbow"], "side": "mean"},
    "trunk_lean": {"primitive": "trunk_lean"},
    "wrist_height": {"primitive": "relative_height", "joints": ["wrist", "shoulder"], "side": "mean"},
    "arm_asymmetry": None,   # computed below: |L-R| elbow angle
}
FULL_SCALE = {"elbow_flexion": 130, "knee_flexion": 90, "hip_flexion": 80, "arm_elevation": 90,
              "trunk_lean": 60, "wrist_height": 0.8, "arm_asymmetry": 60}
NAMES = list(CHANNELS)


def _channel_series(arr: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for n, spec in CHANNELS.items():
        if spec is not None:
            out[n] = smooth(compute_signal(arr, spec), 5)
    L = compute_signal(arr, {**CHANNELS["elbow_flexion"], "side": "left"})
    R = compute_signal(arr, {**CHANNELS["elbow_flexion"], "side": "right"})
    out["arm_asymmetry"] = smooth(np.abs(L - R), 5)
    return out


def descriptor(frames: Sequence, fps: float = 30.0) -> Dict[str, object]:
    """Per-channel activity (range / full-scale, 0..1+) plus dominant tempo in reps/min."""
    arr = normalize(fill_gaps(frames_to_array(frames)))
    ch = _channel_series(arr)
    act = np.array([float(np.nanmax(ch[n]) - np.nanmin(ch[n])) / FULL_SCALE[n] for n in NAMES])
    lead = ch[NAMES[int(np.argmax(act[:6]))]]
    lead = lead - np.nanmean(lead)
    spec = np.abs(np.fft.rfft(lead))
    freqs = np.fft.rfftfreq(len(lead), 1 / fps)
    band = (freqs > 0.1) & (freqs < 2.0)
    tempo = float(freqs[band][int(np.argmax(spec[band]))] * 60) if band.any() and spec[band].max() > 0 else 0.0
    return {"activity": act, "tempo_rpm": tempo, "lead_channel": NAMES[int(np.argmax(act[:6]))]}


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine on log-compressed activity: shape of *which channels move*, not how much."""
    a, b = np.log1p(4 * np.clip(a, 0, 2)), np.log1p(4 * np.clip(b, 0, 2))
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else 0.0


def build_prototypes(exercise_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    from .synthetic import synth_sequence
    protos = {}
    for ex in exercise_ids:
        frames, _ = synth_sequence(ex, reps=3, noise=0.0)
        protos[ex] = descriptor(frames)["activity"]
    return protos


def match_families(frames: Sequence, registry, protos: Dict[str, np.ndarray], top: int = 3) -> Dict:
    d = descriptor(frames)
    if float(np.max(d["activity"])) < 0.25:
        return {"movement_detected": False, "lead_channel": None, "matches": [],
                "verdict": "no clear movement in this clip"}
    ranked = sorted(((similarity(d["activity"], v), k) for k, v in protos.items()), reverse=True)
    by_family: Dict[str, dict] = {}
    for s, k in ranked:
        defn = registry.get(k)
        fam = defn.family if defn else k
        by_family.setdefault(fam, {"family": fam, "similarity": round(s, 3), "example": k})
    matches = list(by_family.values())[:top]
    best = matches[0]["similarity"]
    verdict = ("existing primitives likely sufficient: try a definition in the "
               f"'{matches[0]['family']}' family" if best >= 0.9 else
               "partial overlap: reuse primitives, but check posture/phase rules" if best >= 0.7 else
               "poor match: probably needs a new primitive or representation")
    return {"movement_detected": True, "lead_channel": d["lead_channel"], "tempo_rpm": round(d["tempo_rpm"], 1),
            "activity": {n: round(float(v), 2) for n, v in zip(NAMES, d["activity"])}, "matches": matches,
            "verdict": verdict}
