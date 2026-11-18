"""LiveSession: streaming processor for one exercise set.

push(frame, t) -> UI payload. Computes the definition's signals per frame, runs the causal rep counter (with the
user's personal range when one was captured), tracks tracking loss, estimates the camera view, evaluates form per
completed rep and adds personal insights. Everything the UI needs is in the payload; no ML internals leak out.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from .calibration import facing_yaw_deg
from .form import evaluate_rep, form_score, personal_insights, signal_name, signal_table
from .personal_range import RangeCaptureError, capture_range, signal_key
from .personalization import Baseline
from .primitives import compute_signal
from .quality import QualityTracker
from .repetition import RepEvent, counter_for
from .skeleton import BodyProfile, J, frame_to_array, normalize, Frame

VIEW_WINDOW = 30
VIEW_HINT = {"side": "turn side-on to the camera", "45deg": "turn about 45 degrees to the camera", "front": "face the camera"}


def frame_view(frame: Frame, arr: np.ndarray) -> Optional[str]:
    """front | 45deg | side for one frame: 3D yaw when world landmarks are present, else isotropic shoulder ratio
    (a hidden far shoulder means side-on)."""
    yaw = facing_yaw_deg(frame)
    if yaw is not None:
        return "front" if yaw < 30 else "side" if yaw > 60 else "45deg"
    ls, rs, sc, hc = (arr[J[k], :2] for k in ("left_shoulder", "right_shoulder", "shoulder_center", "hip_center"))
    torso = float(np.linalg.norm(sc - hc))
    if not np.isfinite(torso) or torso <= 0:
        vl, vr = arr[J["left_shoulder"], 2], arr[J["right_shoulder"], 2]
        return "side" if np.isfinite(vl) or np.isfinite(vr) else None
    r = float(np.linalg.norm(ls - rs)) / torso
    return "front" if r >= 0.5 else "side" if r <= 0.25 else "45deg"


class LiveSession:
    def __init__(self, definition, profile: Optional[BodyProfile] = None, baseline: Optional[Baseline] = None,
                 target_reps: Optional[int] = None, buffer_seconds: float = 4.0, fps: float = 30.0):
        self.defn = definition
        self.profile = profile
        self.baseline = baseline or Baseline()
        self.target_reps = target_reps
        self.counter = counter_for(definition.rep)
        self.signals = signal_table(definition)
        self.quality = QualityTracker(definition.required_joints, laterality=getattr(definition, "laterality", "bilateral"))
        self.fps = fps
        self.buf: Deque[Dict] = deque(maxlen=int(buffer_seconds * fps))
        self.reps: List[Dict] = []
        self.feedback: List[Dict] = []
        self.views: Deque[str] = deque(maxlen=VIEW_WINDOW)
        self.capture: Optional[List[float]] = None          # rep-signal values while capturing a personal range
        self.range_info: Optional[Dict] = None
        self._t = 0.0
        rng = self.baseline.range
        if rng and rng.get("signal_key") == signal_key(definition):
            self.apply_range(rng)

    # ---------------------------------------------------------------- personal range
    def apply_range(self, rng: Dict) -> None:
        self.counter.set_range(rng["rest"], rng["peak"])
        self.range_info = {"rest": rng["rest"], "peak": rng["peak"]}

    def start_range_capture(self) -> None:
        """Counting pauses; the next frames are the user's 2-3 slow full-range reps."""
        self.capture = []

    def cancel_range_capture(self) -> None:
        self.capture = None

    def finish_range_capture(self) -> Dict:
        """Returns the captured range (also applied and stored on the baseline) or raises RangeCaptureError."""
        values, self.capture = self.capture or [], None
        rng = capture_range(self.defn, values)
        self.baseline.range = rng
        self.apply_range(rng)
        return rng

    # ---------------------------------------------------------------- per frame
    def _frame_values(self, frame: Frame, arr1: np.ndarray) -> Dict[str, float]:
        arr = normalize(arr1[None], self.profile, compensate_yaw=self.defn.motion_plane == "frontal")
        out = {}
        for name, spec in self.signals.items():
            v = compute_signal(arr, spec)[0]
            out[name] = float(v) if np.isfinite(v) else np.nan
        return out

    @property
    def view(self) -> Optional[str]:
        if not self.views:
            return None
        return max(set(self.views), key=list(self.views).count)

    def push(self, frame: Frame, t: float) -> Dict:
        self._t = t
        q = self.quality.update(frame)
        self.buf.append(frame)
        arr1 = frame_to_array(frame)
        v = frame_view(frame, arr1) if q["frame_ok"] else None
        if v:
            self.views.append(v)
        event: Optional[RepEvent] = None
        values = self._frame_values(frame, arr1) if q["ok"] else None
        if self.capture is not None:
            if values and np.isfinite(values["rep"]):
                self.capture.append(values["rep"])
        elif values is not None:
            event = self.counter.update(t, values)
        else:
            self.counter.missing(t)
        if event is not None:
            self._on_rep(event)
        return self.payload(q)

    def finish(self) -> Optional[Dict]:
        """End of set: close a rep that reached its peak but never fully returned (the last rep of a set)."""
        ev = self.counter.flush(self._t)
        if ev is not None:
            self._on_rep(ev)
            return self.reps[-1]
        return None

    def _on_rep(self, ev: RepEvent):
        ev_list = evaluate_rep(self.defn, ev, self.view) if ev.counted else []
        score = form_score(ev_list)
        insights = personal_insights(ev, self.baseline.as_reference()) if ev.counted else []
        rec = {**ev.summary(), "form_score": score, "evidence": [e.to_dict() for e in ev_list], "insights": insights}
        if not ev.counted:
            rec["evidence"] = [{"rule_id": "partial", "type": "rom", "passed": False, "severity": "warn", "evaluated": True,
                                "measured": round(ev.peak_progress, 3), "threshold": self.defn.rep.count_progress,
                                "message": "Partial rep - not counted. Move through the full range.",
                                "signal": signal_name(self.defn.rep.signal) + " progress", "view": self.view,
                                "window": [round(ev.t_start, 3), round(ev.t_end, 3)], "t_worst": round(ev.t_peak, 3),
                                "status": "counter"}]
        self.reps.append(rec)
        if ev.counted:
            self.baseline.update({"counted": True, "peak_progress": ev.peak_progress, "duration": ev.duration,
                                  "form_score": score})
        judged = [e for e in rec["evidence"] if e.get("evaluated", True)]
        fails = [e for e in judged if not e["passed"]]
        goods = [e for e in judged if e["passed"]]
        fails.sort(key=lambda e: {"major": 0, "warn": 1, "info": 2}[e["severity"]])
        self.feedback = ([{"status": "info" if e["severity"] == "info" else "warning", "message": e["message"], "rule": e["rule_id"]}
                          for e in fails[:2]]
                         or [{"status": "good", "message": e["message"], "rule": e["rule_id"]} for e in goods[:2]])
        self.feedback += [{"status": "info", "message": m, "rule": "personal"} for m in insights[:1]]

    def payload(self, quality: Optional[Dict] = None) -> Dict:
        q = quality or self.quality.last
        view = self.view
        supported = list(self.defn.camera.supported_views)
        view_ok = view is None or len(self.views) < 10 or view in supported
        msg = q.get("message")
        if self.counter.tracking == "lost" and not msg:
            msg = "We lost sight of you - step back into view."
        return {
            "exercise": self.defn.id, "reps": self.counter.count, "target_reps": self.target_reps,
            "partial_reps": self.counter.partials, "abandoned_reps": self.counter.abandoned,
            "phase": "capturing_range" if self.capture is not None else self.counter.phase,
            "progress": round(float(self.counter.progress), 3),
            "quality": {"ok": q.get("ok", True), "message": msg},
            "tracking": self.counter.tracking,
            "view": view, "view_ok": view_ok,
            "view_hint": None if view_ok else f"For accurate {self.defn.name.lower()} analysis, {VIEW_HINT[self.defn.camera.preferred if self.defn.camera.preferred in VIEW_HINT else supported[0]]}.",
            "personal_range": self.range_info, "range_capture": self.capture is not None,
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
            "abandoned_reps": self.counter.abandoned, "personal_range": self.range_info, "view": self.view,
            "avg_form_score": round(float(np.mean(scores)), 1) if scores else None,
            "avg_rom": round(float(np.mean([r["peak_progress"] for r in counted])), 3) if counted else None,
            "avg_rep_seconds": round(float(np.mean([r["duration"] for r in counted])), 2) if counted else None,
            "reps_detail": self.reps,
        }


def rep_signal(definition, frames, profile=None) -> List[float]:
    """The definition's rep signal per frame, computed exactly as LiveSession does (for range capture offline)."""
    s = LiveSession(definition, profile=profile)
    out = []
    for f in frames:
        v = s._frame_values(f, frame_to_array(f))["rep"]
        out.append(v)
    return out


def analyze_sequence(definition, frames, timestamps, profile=None, baseline=None) -> Dict:
    """Offline helper: run a whole clip through LiveSession."""
    s = LiveSession(definition, profile=profile, baseline=baseline)
    for f, t in zip(frames, timestamps):
        s.push(f, t)
    s.finish()
    return s.summary()
