"""Camera/view quality: decide between 'analyse', 'reposition' and 'wait'."""
from __future__ import annotations

from typing import Dict, List, Sequence

from .skeleton import Frame

EDGE = 0.02


def _joint_ok(frame: Frame, j: str, min_visibility: float) -> str:
    """'ok' | 'missing' | 'clipped' for one joint (raw [0,1] image coordinates)."""
    v = frame.get(j)
    if v is None or (len(v) > 2 and v[2] < min_visibility):
        return "missing"
    if not (EDGE <= v[0] <= 1 - EDGE and EDGE <= v[1] <= 1 - EDGE):
        return "clipped"
    return "ok"


def check_frame(frame: Frame, required: Sequence[str], min_visibility: float = 0.5,
                laterality: str = "bilateral") -> Dict:
    """laterality='visible_side': for paired joints (left_x/right_x) one clearly seen side is enough - side-on,
    the far limb is occluded by design and must not block analysis."""
    status = {j: _joint_ok(frame, j, min_visibility) for j in required}
    if laterality == "visible_side":
        for j in list(status):
            if j.startswith(("left_", "right_")):
                other = ("right_" + j[5:]) if j.startswith("left_") else ("left_" + j[6:])
                if other in status and status[other] == "ok":
                    status[j] = "ok"
    missing: List[str] = [j for j, s in status.items() if s == "missing"]
    clipped: List[str] = [j for j, s in status.items() if s == "clipped"]
    ok = not missing and not clipped
    msg = None
    if missing:
        lowers = {"knee", "ankle"}
        msg = ("Step back so your whole body is in view." if any(any(l in m for l in lowers) for m in missing)
               else "Make sure your arms and shoulders are visible.")
    elif clipped:
        msg = "You're at the edge of the frame - move toward the centre."
    return {"ok": ok, "missing": missing, "clipped": clipped, "message": msg}


class QualityTracker:
    """Smooths per-frame checks so a single dropped frame doesn't flash a warning."""

    def __init__(self, required: Sequence[str], window: int = 15, bad_ratio: float = 0.4, laterality: str = "bilateral"):
        self.required, self.window, self.bad_ratio, self.laterality = list(required), window, bad_ratio, laterality
        self.hist: List[bool] = []
        self.last: Dict = {"ok": True, "missing": [], "clipped": [], "message": None}

    def update(self, frame: Frame) -> Dict:
        self.last = check_frame(frame, self.required, laterality=self.laterality)
        self.hist = (self.hist + [self.last["ok"]])[-self.window:]
        bad = 1 - sum(self.hist) / len(self.hist)
        return {**self.last, "ok": bad < self.bad_ratio, "bad_ratio": round(bad, 2), "frame_ok": self.last["ok"]}
