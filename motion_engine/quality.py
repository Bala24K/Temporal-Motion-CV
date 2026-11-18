"""Camera/view quality: decide between 'analyse', 'reposition' and 'wait'."""
from __future__ import annotations

from typing import Dict, List, Sequence

from .skeleton import Frame

EDGE = 0.02


def check_frame(frame: Frame, required: Sequence[str], min_visibility: float = 0.5) -> Dict:
    missing: List[str] = []
    clipped: List[str] = []
    for j in required:
        v = frame.get(j)
        if v is None or (len(v) > 2 and v[2] < min_visibility):
            missing.append(j)
        elif not (EDGE <= v[0] <= 1 - EDGE and EDGE <= v[1] <= 1 - EDGE):
            clipped.append(j)
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

    def __init__(self, required: Sequence[str], window: int = 15, bad_ratio: float = 0.4):
        self.required, self.window, self.bad_ratio = list(required), window, bad_ratio
        self.hist: List[bool] = []
        self.last: Dict = {"ok": True, "missing": [], "clipped": [], "message": None}

    def update(self, frame: Frame) -> Dict:
        self.last = check_frame(frame, self.required)
        self.hist = (self.hist + [self.last["ok"]])[-self.window:]
        bad = 1 - sum(self.hist) / len(self.hist)
        return {**self.last, "ok": bad < self.bad_ratio, "bad_ratio": round(bad, 2)}
