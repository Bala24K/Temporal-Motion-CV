"""Generic, causal rep/phase detection driven by an ExerciseDefinition.

Progress = (value - rest) / (peak - rest): 0 at the start pose, 1 at the
turnaround. Direction-agnostic (works for curls, squats, presses alike) and
physiological: `toward_peak_phase` says whether moving toward the peak is the
concentric or eccentric half (curl: concentric; squat descent: eccentric).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ENTER, EXIT = 0.20, 0.15      # hysteresis on progress to start / finish a rep


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
    signals: Dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def summary(self) -> dict:
        return {
            "index": self.index, "counted": self.counted,
            "peak_progress": round(self.peak_progress, 3), "duration": round(self.duration, 2),
            "toward_peak_seconds": round(self.toward_peak_seconds, 2),
            "away_from_peak_seconds": round(self.away_from_peak_seconds, 2),
        }


class RepCounter:
    def __init__(self, rest: float, peak: float, *, toward_peak_phase: str = "concentric",
                 count_progress: float = 0.55, partial_progress: float = 0.30,
                 min_rep_seconds: float = 0.3, max_rep_seconds: float = 12.0, alpha: float = 0.5):
        self.rest, self.peak = float(rest), float(peak)
        self.toward_peak_phase = toward_peak_phase
        self.count_progress, self.partial_progress = count_progress, partial_progress
        self.min_rep_seconds, self.max_rep_seconds = min_rep_seconds, max_rep_seconds
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.count = 0
        self.partials = 0
        self._ema: Optional[float] = None
        self._active = False
        self._rows: List[tuple] = []
        self.progress = 0.0
        self.phase = "rest"
        self._prev_p = 0.0

    def set_range(self, rest: float, peak: float):
        """Personal baseline can retarget the expected range between reps."""
        if not self._active:
            self.rest, self.peak = float(rest), float(peak)

    def _prog(self, v: float) -> float:
        return (v - self.rest) / (self.peak - self.rest)

    def update(self, t: float, values: Dict[str, float]) -> Optional[RepEvent]:
        """`values['rep']` drives counting; other entries are recorded for form rules."""
        v = values.get("rep")
        if v is None or np.isnan(v):
            return None
        self._ema = v if self._ema is None else self.alpha * v + (1 - self.alpha) * self._ema
        p = self._prog(self._ema)
        self.progress = float(np.clip(p, -0.5, 1.5))
        event = None
        if not self._active:
            self.phase = "rest"
            if p > ENTER:
                self._active, self._rows = True, []
        if self._active:
            self._rows.append((t, p, dict(values)))
            rising = p >= self._prev_p
            toward = "concentric" if self.toward_peak_phase == "concentric" else "eccentric"
            away = "eccentric" if toward == "concentric" else "concentric"
            self.phase = toward if rising else away
            if p < EXIT:
                event = self._finish()
                self._active = False
                self.phase = "rest"
            elif self._rows[-1][0] - self._rows[0][0] > self.max_rep_seconds:
                self._active, self._rows = False, []   # stalled: abandon silently
        self._prev_p = p
        return event

    def _finish(self) -> Optional[RepEvent]:
        rows = self._rows
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
        return RepEvent(
            index=self.count if counted else -self.partials, counted=counted,
            t_start=float(ts[0]), t_peak=float(ts[k]), t_end=float(ts[-1]),
            peak_progress=peak_p, duration=dur,
            toward_peak_seconds=float(ts[k] - ts[0]), away_from_peak_seconds=float(ts[-1] - ts[k]),
            signals=sig,
        )


def counter_for(rep_def) -> RepCounter:
    return RepCounter(rep_def.rest, rep_def.peak, toward_peak_phase=rep_def.toward_peak_phase,
                      count_progress=rep_def.count_progress, partial_progress=rep_def.partial_progress,
                      min_rep_seconds=rep_def.min_rep_seconds, max_rep_seconds=rep_def.max_rep_seconds)
