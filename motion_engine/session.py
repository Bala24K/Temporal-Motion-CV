"""LiveSession: streaming processor for one exercise set.

push(frame, t) -> UI payload. Holds per-frame signals, runs the causal rep
counter, evaluates form per completed rep, and adds personal insights.
Everything the UI needs is in the payload; no ML internals leak out.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from .form import evaluate_rep, form_score, personal_insights, signal_table
from .personalization import Baseline
from .primitives import compute_signal
from .quality import QualityTracker
from .recognition import recognize
from .repetition import RepEvent, counter_for
from .skeleton import BodyProfile, frame_to_array, normalize, Frame


class LiveSession:
    def __init__(self, definition, profile: Optional[BodyProfile] = None, baseline: Optional[Baseline] = None,
                 target_reps: Optional[int] = None, buffer_seconds: float = 4.0, fps: float = 30.0):
        self.defn = definition
        self.profile = profile
        self.baseline = baseline or Baseline()
        self.target_reps = target_reps
        self.counter = counter_for(definition.rep)
        self.signals = signal_table(definition)
        self.quality = QualityTracker(definition.required_joints)
        self.fps = fps
        self.buf: Deque[Dict] = deque(maxlen=int(buffer_seconds * fps))
        self.reps: List[Dict] = []
        self.feedback: List[Dict] = []

    def _frame_values(self, frame: Frame) -> Dict[str, float]:
        arr = normalize(frame_to_array(frame)[None], self.profile, compensate_yaw=self.defn.motion_plane == "frontal")
        out = {}
        for name, spec in self.signals.items():
            v = compute_signal(arr, spec)[0]
            out[name] = float(v) if np.isfinite(v) else np.nan
        return out

    def push(self, frame: Frame, t: float) -> Dict:
        q = self.quality.update(frame)
        self.buf.append(frame)
        event: Optional[RepEvent] = None
        if q["ok"]:
            event = self.counter.update(t, self._frame_values(frame))
        if event is not None:
            self._on_rep(event)
        return self.payload(q)

    def _on_rep(self, ev: RepEvent):
        ev_list = evaluate_rep(self.defn, ev) if ev.counted else []
        score = form_score(ev_list)
        insights = personal_insights(ev, self.baseline.as_reference()) if ev.counted else []
        rec = {**ev.summary(), "form_score": score, "evidence": [e.to_dict() for e in ev_list], "insights": insights}
        if not ev.counted:
            rec["evidence"] = [{"rule_id": "partial", "type": "rom", "passed": False, "severity": "warn",
                                "measured": round(ev.peak_progress, 3), "threshold": self.defn.rep.count_progress,
                                "message": "Partial rep - not counted. Move through the full range."}]
        self.reps.append(rec)
        if ev.counted:
            self.baseline.update({"counted": True, "peak_progress": ev.peak_progress, "duration": ev.duration,
                                  "form_score": score})
        fails = [e for e in rec["evidence"] if not e["passed"]]
        goods = [e for e in rec["evidence"] if e["passed"]]
        fails.sort(key=lambda e: {"major": 0, "warn": 1, "info": 2}[e["severity"]])
        self.feedback = ([{"status": "warning", "message": e["message"], "rule": e["rule_id"]} for e in fails[:2]]
                         or [{"status": "good", "message": e["message"], "rule": e["rule_id"]} for e in goods[:2]])
        self.feedback += [{"status": "info", "message": m, "rule": "personal"} for m in insights[:1]]

    def payload(self, quality: Optional[Dict] = None) -> Dict:
        q = quality or self.quality.last
        return {
            "exercise": self.defn.id, "reps": self.counter.count, "target_reps": self.target_reps,
            "partial_reps": self.counter.partials, "phase": self.counter.phase,
            "progress": round(float(self.counter.progress), 3),
            "quality": {"ok": q.get("ok", True), "message": q.get("message")},
            "feedback": self.feedback, "last_rep": self.reps[-1] if self.reps else None,
            "set_complete": bool(self.target_reps and self.counter.count >= self.target_reps),
        }

    def recognition(self):
        """Used by auto-detect mode; requires a registry on the caller side."""
        raise NotImplementedError

    def summary(self) -> Dict:
        counted = [r for r in self.reps if r["counted"]]
        scores = [r["form_score"] for r in counted if r["form_score"] is not None]
        return {
            "exercise": self.defn.id, "reps": len(counted), "partial_reps": len(self.reps) - len(counted),
            "avg_form_score": round(float(np.mean(scores)), 1) if scores else None,
            "avg_rom": round(float(np.mean([r["peak_progress"] for r in counted])), 3) if counted else None,
            "avg_rep_seconds": round(float(np.mean([r["duration"] for r in counted])), 2) if counted else None,
            "reps_detail": self.reps,
        }


def analyze_sequence(definition, frames, timestamps, profile=None, baseline=None) -> Dict:
    """Offline helper: run a whole clip through LiveSession."""
    s = LiveSession(definition, profile=profile, baseline=baseline)
    for f, t in zip(frames, timestamps):
        s.push(f, t)
    return s.summary()
