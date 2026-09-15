"""
Overhead shoulder press analyzer. Rule-based, same RepStateMachine as the
other exercises.

Angle convention: elbow_angle = angle_3pt(shoulder, elbow, wrist).
  ~70-90deg   = rack/start position (elbows bent, hands at shoulder height)
  ~160-180deg = lockout overhead (arm extended)

IMPORTANT — this is the FLIPPED convention relative to bicep curl and
squat: "rest" (the start/end position between reps) is the CONTRACTED
(low-angle) zone here, not EXTENDED. RepStateMachine itself doesn't care
which zone is "rest" — it just completes a rep on EXTENDED-zone(confirmed)
->CONTRACTED-zone(confirmed)->EXTENDED-zone(confirmed) again... actually
more precisely: a rep completes whenever EXTENDED gets reconfirmed while
`armed` (i.e. CONTRACTED was visited since the last completion). For a
press that starts at the rack (CONTRACTED), that means the very first
press to lockout already completes rep 1 (correctly — pressing up to
lockout IS one rep), then returning to the rack re-arms for rep 2. See
exercises/base.py's RepStateMachine docstring for the general mechanism;
this module is what exercises it in the "flipped" direction.

Baseline-reset logic below therefore keys off CONTRACTED (not EXTENDED,
unlike bicep_curl.py) as the rest point.
"""
from typing import Optional
from dataclasses import dataclass

from biomechanics.angles import angle_3pt, normalized_displacement
from exercises.base import RepStateMachine, ExerciseAnalyzer, ok, warn
from pose.landmarks import PoseFrame


@dataclass
class PressThresholds:
    extended_angle: float = 160.0    # lockout overhead
    contracted_angle: float = 90.0   # rack position
    min_dwell_seconds: float = 0.08
    partial_lockout_threshold: float = 55.0    # ROM below this on a completed rep -> flag (natural full ROM ~70deg given the thresholds above)
    shoulder_drift_threshold: float = 0.12
    min_visibility: float = 0.5


class ShoulderPressAnalyzer(ExerciseAnalyzer):
    name = "shoulder_press"
    EXTENDED_LABEL = "LOCKOUT"
    CONTRACTED_LABEL = "RACK"
    INCREASING_ANGLE_LABEL = "PRESSING"   # angle increasing = pressing overhead
    DECREASING_ANGLE_LABEL = "LOWERING"   # angle decreasing = returning to rack

    def __init__(self, side: str = "auto", thresholds: Optional[PressThresholds] = None):
        assert side in ("left", "right", "auto")
        self.side = side
        self.t = thresholds or PressThresholds()
        self.sm = RepStateMachine(
            extended_threshold=self.t.extended_angle,
            contracted_threshold=self.t.contracted_angle,
            min_dwell_seconds=self.t.min_dwell_seconds,
        )
        self._active_side: Optional[str] = None
        self._shoulder_baseline = None
        self._prev_angle: Optional[float] = None

    def _resolve_side(self, frame: PoseFrame) -> Optional[str]:
        if self.side != "auto":
            return self.side
        if self._active_side is not None:
            return self._active_side
        joints = ["shoulder", "elbow", "wrist"]
        left_ok = frame.all_visible([f"left_{j}" for j in joints], self.t.min_visibility)
        right_ok = frame.all_visible([f"right_{j}" for j in joints], self.t.min_visibility)
        if left_ok and not right_ok:
            self._active_side = "left"
        elif right_ok and not left_ok:
            self._active_side = "right"
        elif left_ok and right_ok:
            self._active_side = "left"
        else:
            return None
        return self._active_side

    def process(self, frame: PoseFrame, now: Optional[float] = None) -> dict:
        side = self._resolve_side(frame)
        if side is None:
            return self._output(confidence=0.0, phase="NO PERSON DETECTED",
                                 form_flags=[warn("Move into frame - can't see your arm")])

        shoulder = frame.get(f"{side}_shoulder")
        elbow = frame.get(f"{side}_elbow")
        wrist = frame.get(f"{side}_wrist")

        joints_visible = frame.all_visible(
            [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"], self.t.min_visibility)

        angle = None
        if joints_visible:
            angle = angle_3pt(shoulder.xy(), elbow.xy(), wrist.xy())

        prev_angle_for_display = self._prev_angle
        rep_result = self.sm.update(angle, now=now)

        form_flags = []
        confidence = 1.0 if joints_visible else 0.0

        if not joints_visible:
            form_flags.append(warn("Can't see your shoulder/elbow/wrist clearly"))
        elif self.sm.degraded:
            form_flags.append(warn("Tracking lost mid-rep - hold still a moment"))
        else:
            scale = frame.torso_scale()
            if self._shoulder_baseline is None:
                self._shoulder_baseline = shoulder.xy()
            drift = normalized_displacement(shoulder.xy(), self._shoulder_baseline, scale)
            if drift is not None and drift > self.t.shoulder_drift_threshold:
                form_flags.append(warn("Excessive shoulder/torso movement"))

            if not form_flags:
                form_flags.append(ok("Controlled movement"))

        # Reset the shoulder baseline at the RACK position (CONTRACTED,
        # not armed) — that's rest for a press, unlike curl/squat where
        # rest is EXTENDED. See module docstring.
        if self.sm.confirmed_zone is not None and self.sm.confirmed_zone.value == "contracted" and not self.sm._armed:
            self._shoulder_baseline = shoulder.xy() if joints_visible else self._shoulder_baseline

        self._prev_angle = angle

        if rep_result is not None:
            if rep_result.rom < self.t.partial_lockout_threshold:
                form_flags.append(warn("Partial lockout - press higher overhead"))
            else:
                form_flags.append(ok("Full lockout"))

        return self._output(
            confidence=confidence,
            phase=self._display_phase(self.sm, prev_angle_for_display, angle),
            form_flags=form_flags, angle=angle, side=side, rep_result=rep_result,
        )

    def _output(self, confidence, phase, form_flags, angle=None, side=None, rep_result=None) -> dict:
        last_rep = self.sm.completed_reps[-1] if self.sm.completed_reps else None
        return {
            "exercise": "SHOULDER PRESS",
            "side": side,
            "rep_count": self.sm.rep_count,
            "phase": phase,
            "current_angle": angle,
            "rom": round(last_rep.rom, 1) if last_rep else None,
            "tempo": round(last_rep.duration, 2) if last_rep else None,
            "concentric_s": round(last_rep.concentric_duration, 2) if last_rep else None,
            "eccentric_s": round(last_rep.eccentric_duration, 2) if last_rep else None,
            "form_flags": form_flags,
            "confidence": confidence,
            "just_completed_rep": rep_result,
        }

    def reset(self):
        self.sm = RepStateMachine(
            extended_threshold=self.t.extended_angle,
            contracted_threshold=self.t.contracted_angle,
            min_dwell_seconds=self.t.min_dwell_seconds,
        )
        self._shoulder_baseline = None
        self._prev_angle = None
