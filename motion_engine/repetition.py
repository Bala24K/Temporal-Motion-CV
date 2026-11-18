"""Generic, causal rep/phase detection driven by an ExerciseDefinition (counter v2).

Progress = (value - rest) / (peak - rest): 0 at the start pose, 1 at the turnaround. `rest`/`peak` start as the
definition's population values and are replaced by the user's captured personal range when one exists
(`set_range`). Direction-agnostic; `toward_peak_phase` says whether moving toward the peak is concentric or
eccentric (curl: concentric; squat descent: eccentric).

Why v2 (docs/REP_COUNTING_AUDIT.md): v1 completed a rep only after a return to within 15% of the textbook rest
pose, so users who don't fully re-extend were silently not counted (curl back to 140 deg: 0/5 reps). v2:
  * time-based smoothing and dwell, so counts don't depend on frame rate;
  * a rep starts after progress > start_progress for start_dwell seconds;
  * it completes on the FIRST of: (a) relaxed return below complete_progress for complete_dwell,
    (b) reversal: dropped >= `reversal` from its peak and the next rep has clearly started,
    (c) settle: dropped >= `settle_drop` from its peak and held still for settle_seconds;
  * tracking states: missing signal -> "degraded" (rep kept open) -> "lost" after lost_after seconds (rep
    abandoned, counted in `abandoned`, smoothing re-anchored on recovery);
  * a rep open longer than max_rep_seconds is abandoned and counted, never silently dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

RISE = 0.10                  # reversal rule: progress must climb this far above the valley to call it a new rep
STILL_RATE = 0.25            # settle rule: |d progress / dt| below this (per second) counts as still


@dataclass
class RepEvent:
    index: int
    counted: bool                 # False => partial rep, not counted
    t_start: float
    t_peak: float
    t_end: float
    peak_progress: float
    duration: float
    toward_peak_seconds: float
    away_from_peak_seconds: float
    end_progress: float = 0.0     # how far back toward rest the rep returned (0 = full return)
    completion: str = "return"    # return | reversal | settle | flush
    signals: Dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def summary(self) -> dict:
        return {
            "index": self.index, "counted": self.counted,
            "peak_progress": round(self.peak_progress, 3), "duration": round(self.duration, 2),
            "toward_peak_seconds": round(self.toward_peak_seconds, 2),
            "away_from_peak_seconds": round(self.away_from_peak_seconds, 2),
            "end_progress": round(self.end_progress, 3), "completion": self.completion,
        }


class RepCounter:
    def __init__(self, rest: float, peak: float, *, toward_peak_phase: str = "concentric",
                 count_progress: float = 0.55, partial_progress: float = 0.30,
                 min_rep_seconds: float = 0.3, max_rep_seconds: float = 20.0,
                 start_progress: float = 0.25, start_dwell: float = 0.10,
                 complete_progress: float = 0.35, complete_dwell: float = 0.15,
                 reversal: float = 0.25, settle_drop: float = 0.45, settle_seconds: float = 0.4,
                 smoothing_tau: float = 0.08, lost_after: float = 3.0):
        self.rest, self.peak = float(rest), float(peak)
        self.toward_peak_phase = toward_peak_phase
        self.count_progress, self.partial_progress = count_progress, partial_progress
        self.min_rep_seconds, self.max_rep_seconds = min_rep_seconds, max_rep_seconds
        self.start_progress, self.start_dwell = start_progress, start_dwell
        self.complete_progress, self.complete_dwell = complete_progress, complete_dwell
        self.reversal, self.settle_drop, self.settle_seconds = reversal, settle_drop, settle_seconds
        self.tau, self.lost_after = smoothing_tau, lost_after
        self.personal = False
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.count = 0
        self.partials = 0
        self.abandoned = 0
        self.progress = 0.0
        self.phase = "rest"
        self.tracking = "ok"
        self._ema: Optional[float] = None
        self._t_prev: Optional[float] = None       # last valid sample time
        self._p_prev = 0.0
        self._active = False
        self._rows: List[tuple] = []              # (t, p, values) of the open rep
        self._pre: List[tuple] = []               # candidate rows while waiting for start_dwell
        self._start_cand: Optional[float] = None
        self._comp_cand: Optional[float] = None
        self._still_since: Optional[float] = None
        self._last_event: Optional[str] = None

    def set_range(self, rest: float, peak: float, personal: bool = True):
        """Retarget the expected range (personal range capture). Takes effect immediately; an open rep is dropped
        because its progress was measured against the old range."""
        if float(rest) == float(peak):
            raise ValueError("rest and peak must differ")
        self.rest, self.peak, self.personal = float(rest), float(peak), personal
        self._active, self._rows, self._pre, self._start_cand = False, [], [], None
        self._ema = None

    def _prog(self, v: float) -> float:
        return (v - self.rest) / (self.peak - self.rest)

    # ------------------------------------------------------------------ input
    def missing(self, t: float) -> None:
        """No usable signal this frame (tracking loss / low visibility)."""
        if self._t_prev is None:
            return
        gap = t - self._t_prev
        if gap >= self.lost_after:
            if self.tracking != "lost" and self._active:
                self.abandoned += 1
                self._last_event = "abandoned_tracking"
                self._active, self._rows = False, []
            self.tracking = "lost"
            self._ema = None                       # re-anchor on recovery, never complete a rep off stale values
        else:
            self.tracking = "degraded"

    def update(self, t: float, values: Dict[str, float]) -> Optional[RepEvent]:
        """`values['rep']` drives counting; other entries are recorded for form rules."""
        v = values.get("rep")
        if v is None or not math.isfinite(v):
            self.missing(t)
            return None
        self.tracking = "ok"
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        a = 1.0 if self._ema is None else 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        self._ema = v if self._ema is None else a * v + (1 - a) * self._ema
        p = self._prog(self._ema)
        rate = (p - self._p_prev) / dt if dt > 0 else 0.0
        self.progress = float(np.clip(p, -0.5, 1.5))
        self._t_prev = t
        event = None

        if not self._active:
            self.phase = "rest"
            if p > self.start_progress:
                if self._start_cand is None:
                    self._start_cand, self._pre = t, []
                self._pre.append((t, p, dict(values)))
                if t - self._start_cand >= self.start_dwell:
                    self._active, self._rows, self._pre = True, self._pre, []
                    self._comp_cand = self._still_since = None
            else:
                self._start_cand, self._pre = None, []
        else:
            self._rows.append((t, p, dict(values)))
            event = self._check_completion(t, p, rate)

        if self._active:
            toward = "concentric" if self.toward_peak_phase == "concentric" else "eccentric"
            away = "eccentric" if toward == "concentric" else "concentric"
            if abs(p - self._p_prev) > 0.005:          # small deadband: no phase flicker at the turnaround
                self.phase = toward if p > self._p_prev else away
        self._p_prev = p
        return event

    def _check_completion(self, t: float, p: float, rate: float) -> Optional[RepEvent]:
        rows = self._rows
        ps = np.array([r[1] for r in rows])
        k = int(np.argmax(ps))
        peak_p = float(ps[k])
        # (a) relaxed return toward rest
        if p < self.complete_progress:
            if self._comp_cand is None:
                self._comp_cand = t
            if t - self._comp_cand >= self.complete_dwell:
                return self._close(len(rows), "return")
        else:
            self._comp_cand = None
        if peak_p >= self.partial_progress and k < len(rows) - 1:
            after = ps[k:]
            v_rel = int(np.argmin(after))
            valley = float(after[v_rel])
            # (b) reversal: went down by `reversal`, and is clearly on the way up into the next rep
            if valley <= peak_p - self.reversal and p >= valley + RISE and v_rel < len(after) - 1:
                return self._close(k + v_rel + 1, "reversal", carry=True)
            # (c) settle: well down from the peak and holding still (paused between reps, short of full return)
            if p <= peak_p - self.settle_drop and abs(rate) < STILL_RATE:
                if self._still_since is None:
                    self._still_since = t
                if t - self._still_since >= self.settle_seconds:
                    return self._close(len(rows), "settle")
            else:
                self._still_since = None
        if rows[-1][0] - rows[0][0] > self.max_rep_seconds:
            self.abandoned += 1
            self._last_event = "abandoned_timeout"
            self._active, self._rows = False, []
        return None

    def _close(self, n: int, how: str, carry: bool = False) -> Optional[RepEvent]:
        rows, rest = self._rows[:n], self._rows[n - 1:] if carry else []
        ev = self._finish(rows, how)
        self._comp_cand = self._still_since = None
        if carry:                                   # the valley starts the next rep
            self._rows = rest
        else:
            self._active, self._rows = False, []
            self.phase = "rest"
        return ev

    def flush(self, t: Optional[float] = None) -> Optional[RepEvent]:
        """End of set: close an open rep that reached its peak and came back down part of the way."""
        if not self._active or len(self._rows) < 2:
            return None
        ps = np.array([r[1] for r in self._rows])
        k = int(np.argmax(ps))
        if k < len(ps) - 1 and ps[-1] <= ps[k] - self.reversal:
            return self._close(len(self._rows), "flush")
        return None

    def _finish(self, rows, how: str) -> Optional[RepEvent]:
        ts = np.array([r[0] for r in rows])
        ps = np.array([r[1] for r in rows])
        k = int(np.argmax(ps))
        peak_p = float(ps[k])
        dur = float(ts[-1] - ts[0])
        if peak_p < self.partial_progress or dur < self.min_rep_seconds * 0.5:
            return None  # twitch / noise
        counted = peak_p >= self.count_progress and dur >= self.min_rep_seconds
        if counted:
            self.count += 1
        else:
            self.partials += 1
        keys = rows[0][2].keys()
        sig = {k2: np.array([r[2].get(k2, np.nan) for r in rows], float) for k2 in keys}
        self._last_event = "rep" if counted else "partial"
        return RepEvent(
            index=self.count if counted else -self.partials, counted=counted,
            t_start=float(ts[0]), t_peak=float(ts[k]), t_end=float(ts[-1]),
            peak_progress=peak_p, duration=dur,
            toward_peak_seconds=float(ts[k] - ts[0]), away_from_peak_seconds=float(ts[-1] - ts[k]),
            end_progress=float(ps[-1]), completion=how, signals=sig,
        )

    def pop_event(self) -> Optional[str]:
        """Last notable counter event (rep / partial / abandoned_*), cleared on read."""
        e, self._last_event = self._last_event, None
        return e


def counter_for(rep_def) -> RepCounter:
    g = lambda name, default: getattr(rep_def, name, default)
    return RepCounter(rep_def.rest, rep_def.peak, toward_peak_phase=rep_def.toward_peak_phase,
                      count_progress=rep_def.count_progress, partial_progress=rep_def.partial_progress,
                      min_rep_seconds=rep_def.min_rep_seconds, max_rep_seconds=g("max_rep_seconds", 20.0),
                      start_progress=g("start_progress", 0.25), start_dwell=g("start_dwell", 0.10),
                      complete_progress=g("complete_progress", 0.35), complete_dwell=g("complete_dwell", 0.15),
                      reversal=g("reversal", 0.25), settle_drop=g("settle_drop", 0.45),
                      settle_seconds=g("settle_seconds", 0.4), smoothing_tau=g("smoothing_tau", 0.08))
