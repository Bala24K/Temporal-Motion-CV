"""Definition-driven exercise recognition with explicit support status.

No per-exercise model: every ExerciseDefinition is scored against the same
window by how well its rep signal, amplitude, periodicity and posture
constraints fit. Definitions that cannot be told apart from pose alone
(same signal + range + posture, e.g. bicep vs hammer curl) are merged into one
candidate group instead of pretending to discriminate. The system says
"unsure" or "unsupported" rather than forcing a label.

A learned model (see research/) can later replace/augment `score`; the
decision logic and `Recognition` contract stay the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .primitives import compute_signal, smooth
from .repetition import counter_for
from .skeleton import fill_gaps, frames_to_array, normalize, BodyProfile

TEMP = 0.12
MIN_MOVEMENT = 0.30       # progress range below this = no clear movement
MIN_SCORE = 0.45          # best score below this = not a movement we know


@dataclass
class Candidate:
    exercise: str                 # canonical id of the group
    members: List[str]
    score: float
    probability: float
    distinguishable: bool

    def to_dict(self):
        return {"exercise": self.exercise, "members": self.members, "score": round(self.score, 3),
                "probability": round(self.probability, 3), "distinguishable": self.distinguishable}


@dataclass
class Recognition:
    status: str                   # recognized | ambiguous | no_movement | unsupported
    action: str                   # analyze | confirm | wait | fallback
    confidence: float
    top: Optional[Candidate]
    candidates: List[Candidate] = field(default_factory=list)
    message: str = ""

    def to_dict(self):
        return {"status": self.status, "action": self.action, "confidence": round(self.confidence, 3),
                "top": self.top.to_dict() if self.top else None,
                "candidates": [c.to_dict() for c in self.candidates], "message": self.message}


def _progress(sig: np.ndarray, rest: float, peak: float) -> np.ndarray:
    return (smooth(sig, 5) - rest) / (peak - rest)


def _posture_penalty(arr, defn) -> float:
    pen = 1.0
    for p in defn.posture:
        s = compute_signal(arr, p.signal)
        s = s[np.isfinite(s)]
        if not len(s):
            continue
        v = {"median": np.median, "min": np.min, "max": np.max}[p.stat](s)
        viol = max(0.0, (p.min - v) if p.min is not None else 0.0, (v - p.max) if p.max is not None else 0.0)
        pen *= float(np.exp(-(viol / p.scale) ** 2))
    return pen


def score_definition(arr: np.ndarray, defn, fps: float) -> Dict[str, float]:
    sig = compute_signal(arr, defn.rep.signal)
    if not np.isfinite(sig).any():
        return {"score": 0.0, "range": 0.0}
    prog = _progress(sig, defn.rep.rest, defn.rep.peak)
    r = float(np.nanmax(prog) - np.nanmin(prog))
    # amplitude fit: penalise both too-small and far-too-large excursions
    amp = float(np.sqrt(min(r, 1.0))) if r <= 1.0 else float(np.exp(-((r - 1.0) / 0.5) ** 2))
    # the window should start near the rest pose for a clean rep cycle
    rc = counter_for(defn.rep)
    rc.min_rep_seconds = 0.0
    for i, v in enumerate(sig):
        rc.update(i / fps, {"rep": float(v)})
    periodic = 1.0 if rc.count + rc.partials >= 1 else 0.6
    return {"score": amp * periodic * _posture_penalty(arr, defn), "range": r}


def recognize(frames, registry, fps: float = 30.0, profile: Optional[BodyProfile] = None,
              min_frames: int = 30) -> Recognition:
    if len(frames) < min_frames:
        return Recognition("no_movement", "wait", 0.0, None, message="Collecting movement...")
    arr = normalize(fill_gaps(frames_to_array(frames) if not isinstance(frames, np.ndarray) else frames), profile)
    defs = registry.recognisable()
    scored = {d.id: score_definition(arr, d, fps) for d in defs}
    if max((s["range"] for s in scored.values()), default=0) < MIN_MOVEMENT:
        return Recognition("no_movement", "wait", 0.0, None, message="Waiting for you to start moving.")
    groups: Dict[str, List[str]] = {}
    for d in defs:
        groups.setdefault(d.equivalence_key(), []).append(d.id)
    cands = []
    for ids in groups.values():
        cands.append(Candidate(sorted(ids)[0], sorted(ids), max(scored[i]["score"] for i in ids), 0.0, len(ids) == 1))
    z = np.array([c.score for c in cands]) / TEMP
    p = np.exp(z - z.max())
    p /= p.sum()
    for c, pi in zip(cands, p):
        c.probability = float(pi)
    cands.sort(key=lambda c: -c.score)
    top = cands[0]
    if top.score < MIN_SCORE:
        return Recognition("unsupported", "fallback", top.score * top.probability, None, cands[:3],
                           "This movement isn't fully supported yet. You can still track it manually.")
    conf = top.score * top.probability           # low if score is weak OR a rival is close
    if conf < 0.55:
        return Recognition("ambiguous", "confirm", conf, top, cands[:3], "Not sure which exercise this is - please confirm.")
    if not top.distinguishable:
        return Recognition("ambiguous", "confirm", conf, top, cands[:3],
                           "These exercises look identical to the camera - tell us which one you're doing.")
    return Recognition("recognized", "analyze", conf, top, cands[:3])
