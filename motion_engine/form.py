"""Form analysis: interpretable evidence from rules, never one opaque score.

Each rule yields an `Evidence` (measured value, threshold, pass/fail, message).
Form consistency for a rep = weighted share of rules passed. Personal baselines
add *insights* (comparison to the user's own usual range), kept separate from
pass/fail so objective measurement is never conflated with advice.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from .repetition import RepEvent

SEV_WEIGHT = {"info": 0.5, "warn": 1.0, "major": 2.0}


@dataclass
class Evidence:
    rule_id: str
    type: str
    passed: bool
    severity: str
    measured: Optional[float]
    threshold: Optional[float]
    message: str

    def to_dict(self):
        d = asdict(self)
        for k in ("measured", "threshold"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


def signal_table(defn) -> Dict[str, dict]:
    """Every named signal the definition needs, computed per frame by the session."""
    t = {"rep": defn.rep.signal}
    for r in defn.form_rules:
        if r.type in ("stability", "limit") and r.signal:
            t[f"r:{r.id}"] = r.signal
        elif r.type == "symmetry" and r.signal:
            t[f"r:{r.id}:L"] = {**r.signal, "side": "left"}
            t[f"r:{r.id}:R"] = {**r.signal, "side": "right"}
    return t


def evaluate_rep(defn, rep: RepEvent) -> List[Evidence]:
    out: List[Evidence] = []
    for r in defn.form_rules:
        ok, measured, thr = True, None, None
        if r.type == "rom":
            measured = rep.peak_progress
            lo_ok = r.min is None or measured >= r.min
            hi_ok = r.max is None or measured <= r.max
            ok, thr = lo_ok and hi_ok, (r.min if not lo_ok else r.max)
        elif r.type == "tempo":
            measured, thr = rep.duration, r.min_rep_seconds
            ok = r.min_rep_seconds is None or rep.duration >= r.min_rep_seconds
            if ok and r.min_phase_seconds is not None:
                measured = min(rep.toward_peak_seconds, rep.away_from_peak_seconds)
                thr, ok = r.min_phase_seconds, measured >= r.min_phase_seconds
        elif r.type == "stability":
            s = rep.signals.get(f"r:{r.id}")
            if s is not None and np.isfinite(s).any():
                measured, thr = float(np.nanmax(s) - np.nanmin(s)), r.max
                ok = r.max is None or measured <= r.max
        elif r.type == "limit":
            s = rep.signals.get(f"r:{r.id}")
            if s is not None and np.isfinite(s).any():
                if r.max is not None:
                    measured, thr = float(np.nanmax(s)), r.max
                    ok = measured <= r.max
                if ok and r.min is not None:
                    measured, thr = float(np.nanmin(s)), r.min
                    ok = measured >= r.min
        elif r.type == "symmetry":
            a, b = rep.signals.get(f"r:{r.id}:L"), rep.signals.get(f"r:{r.id}:R")
            if a is not None and b is not None and np.isfinite(a).any() and np.isfinite(b).any():
                measured, thr = float(np.nanmean(np.abs(a - b))), r.max
                ok = r.max is None or measured <= r.max
        out.append(Evidence(r.id, r.type, bool(ok), r.severity, measured, thr, (r.good if ok else r.bad) or r.id))
    return out


def form_score(evidence: List[Evidence]) -> Optional[float]:
    """Weighted share of rules passed, 0..100. None if nothing was evaluable."""
    ev = [e for e in evidence if e.measured is not None]
    if not ev:
        return None
    w = sum(SEV_WEIGHT[e.severity] for e in ev)
    return round(100 * sum(SEV_WEIGHT[e.severity] for e in ev if e.passed) / w, 1)


def personal_insights(rep: RepEvent, baseline: Optional[dict]) -> List[str]:
    """Compare to the user's own history. `baseline` = {'n','peak_progress','duration'}."""
    if not baseline or baseline.get("n", 0) < 5:
        return []
    msgs = []
    bp = baseline.get("peak_progress", 0)
    if bp > 0:
        ratio = rep.peak_progress / bp
        if ratio < 0.88:
            msgs.append("Range of motion is a bit below your usual today.")
        elif ratio > 1.12:
            msgs.append("Range of motion is above your usual - nice.")
    bd = baseline.get("duration", 0)
    if bd > 0 and rep.duration < 0.6 * bd:
        msgs.append("This rep was quicker than your usual tempo.")
    return msgs
