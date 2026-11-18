"""Personal movement baseline: running per-exercise statistics of the user's own reps."""
from __future__ import annotations

from typing import Dict, Optional


class Baseline:
    """Exponentially-weighted mean of a user's rep metrics for one exercise.
    Serialises to a plain dict so it can live in a JSON DB column."""

    FIELDS = ("peak_progress", "duration", "form_score")

    def __init__(self, d: Optional[Dict] = None):
        d = d or {}
        self.n = int(d.get("n", 0))
        self.mean = {k: float(d.get("mean", {}).get(k, 0.0)) for k in self.FIELDS}
        self.range: Optional[Dict] = d.get("range")       # personal rest/peak (motion_engine/personal_range.py)

    def update(self, rep: Dict, half_life: int = 30) -> None:
        if not rep.get("counted", True):
            return
        a = max(1.0 / (self.n + 1), 1.0 - 0.5 ** (1.0 / half_life))
        for k in self.FIELDS:
            v = rep.get(k)
            if v is None:
                continue
            self.mean[k] = v if self.n == 0 else (1 - a) * self.mean[k] + a * v
        self.n += 1

    def to_dict(self) -> Dict:
        out = {"n": self.n, "mean": {k: round(v, 4) for k, v in self.mean.items()}}
        if self.range:
            out["range"] = self.range
        return out

    def as_reference(self) -> Dict:
        return {"n": self.n, **self.mean}
