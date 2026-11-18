"""Personal range capture: the user's own rest and turnaround values for an exercise's rep signal.

Calibration v1/v2 never changed counting: rest/peak were textbook constants, so users who don't reach the textbook
angles (or whose 2D projection doesn't, e.g. a curl filmed from the front) were under-counted. Here the user
performs 2-3 slow reps through their comfortable full range; rest/peak are robust extremes of that signal,
clamped so a lazy capture cannot make counting trivially easy (docs/REP_COUNTING_AUDIT.md section 4.2).

Stored per (user, exercise) together with the signal key, so a changed definition invalidates an old range.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional, Sequence

import numpy as np

from .primitives import smooth

MIN_CYCLES = 2
MIN_AMPLITUDE = 0.35       # captured |peak - rest| must be >= this share of the population range
                           # (a front-view curl projects to ~60 of the textbook 125 deg)
CLAMP = 0.25               # rest/peak may move at most this share of the population range (default bounds)


class RangeCaptureError(ValueError):
    pass


def signal_key(defn) -> str:
    """Identifies the rep signal a range was measured with. A new definition version or signal invalidates it."""
    raw = json.dumps([defn.id, defn.version, defn.rep.signal], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def bounds(defn) -> Dict[str, tuple]:
    r = defn.rep
    if r.population_range:
        return {"rest": tuple(r.population_range["rest"]), "peak": tuple(r.population_range["peak"])}
    span = abs(r.peak - r.rest) * CLAMP
    return {"rest": (r.rest - span, r.rest + span), "peak": (r.peak - span, r.peak + span)}


def _cycles(sig: np.ndarray, lo: float, hi: float) -> int:
    """Full excursions between the lower and upper thirds of the captured range (hysteresis, noise-proof)."""
    a, b = lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3
    state, n = None, 0
    for v in sig:
        if v <= a:
            if state == "hi":
                n += 1
            state = "lo"
        elif v >= b:
            if state == "lo":
                n += 1
            state = "hi"
    return n // 2


def capture_range(defn, signal: Sequence[float], timestamps: Optional[Sequence[float]] = None) -> Dict:
    """signal: the definition's rep signal per frame (as LiveSession computes it). Returns
    {rest, peak, cycles, clamped, signal_key, population: {rest, peak}}; raises RangeCaptureError with a
    user-facing message when the capture can't be trusted."""
    x = np.asarray(signal, float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        raise RangeCaptureError("We couldn't see the movement clearly - stay in view and try again.")
    x = smooth(x, 5)
    lo, hi = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    r = defn.rep
    pop_span = abs(r.peak - r.rest)
    if hi - lo < MIN_AMPLITUDE * pop_span:
        raise RangeCaptureError("Move through your full comfortable range - that looked like a small movement.")
    cycles = _cycles(x, lo, hi)
    if cycles < MIN_CYCLES:
        raise RangeCaptureError(f"Do at least {MIN_CYCLES} full, slow reps.")
    rest, peak = (hi, lo) if r.peak < r.rest else (lo, hi)
    b = bounds(defn)
    rest_c = float(np.clip(rest, min(b["rest"]), max(b["rest"])))
    peak_c = float(np.clip(peak, min(b["peak"]), max(b["peak"])))
    return {"rest": round(rest_c, 2), "peak": round(peak_c, 2), "cycles": cycles,
            "clamped": bool(abs(rest_c - rest) > 1e-6 or abs(peak_c - peak) > 1e-6),
            "measured": {"rest": round(rest, 2), "peak": round(peak, 2)},
            "population": {"rest": r.rest, "peak": r.peak}, "signal_key": signal_key(defn)}
