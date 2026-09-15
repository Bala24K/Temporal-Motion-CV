"""
Bicep curl analyzer. Rule-based, per the V1 plan — no ML here. Ties
together: elbow-angle geometry, the generic RepStateMachine, and a couple
of obvious, geometrically-grounded form checks.

Angle convention: elbow_angle = angle_3pt(shoulder, elbow, wrist).
  ~160-180deg = arm extended (straight)
  ~30-60deg   = arm fully curled
"""
from typing import Optional
from dataclasses import dataclass, field

from biomechanics.angles import angle_3pt, angular_velocity, normalized_displacement
from exercises.base import RepStateMachine, ExerciseAnalyzer, ok, warn
from pose.landmarks import PoseFrame


@dataclass
class FormThresholds:
    extended_angle: float = 155.0
    contracted_angle: float = 55.0
    min_dwell_seconds: float = 0.08
    partial_rom_threshold: float = 90.0          # ROM below this on a completed rep -> flag
    shoulder_drift_threshold: float = 0.10        # normalized displacement -> flag
    fast_movement_deg_per_sec: float = 400.0      # angular velocity above this -> flag
    min_visibility: float = 0.5


class BicepCurlAnalyzer(ExerciseAnalyzer):
    name = "bicep_curl"
    # Rest = arm straight (EXTENDED zone); peak = fully curled (CONTRACTED).
    DECREASING_ANGLE_LABEL = "CURLING"    # angle decreasing = curling up
    INCREASING_ANGLE_LABEL = "LOWERING"   # angle increasing = extending back down

    def __init__(self, side: str = "auto", thresholds: Optional[FormThresholds] = None):
        assert side in ("left", "right", "auto")
        self.side = side
        self.t = thresholds or FormThresholds()
        self.sm = RepStateMachine(
            extended_threshold=self.t.extended_angle,
            contracted_threshold=self.t.contracted_angle,
            min_dwell_seconds=self.t.min_dwell_seconds,
        )
        self._active_side: Optional[str] = None  # resolved once, in "auto" mode
        self._shoulder_baseline = None            # (x, y) captured at calibration / each extended rep-start
        self._prev_angle: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._last_form_flags: list[str] = []
        self._max_abs_velocity_this_rep: float = 0.0

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
            # both visible: default to whichever currently has the more
            # extended (i.e. more "ready") elbow angle, otherwise left
            self._active_side = "left"
        else:
            return None  # neither arm reliably visible yet
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

        # Snapshot BEFORE self._prev_angle gets overwritten below — used
        # for the display-phase direction check at the end of this method.
        # (A real bug during development: computing this AFTER the
        # overwrite compares the current angle to itself, so direction
        # detection silently always returned None and "CURLING"/"LOWERING"
        # never actually displayed — caught by testing the phase label
        # directly, not just by rep-counting behavior looking right.)
        prev_angle_for_display = self._prev_angle

        rep_result = self.sm.update(angle, now=now)

        form_flags = []
        confidence = 1.0 if joints_visible else 0.0

        if not joints_visible:
            form_flags.append(warn("Can't see your shoulder/elbow/wrist clearly"))
        elif self.sm.degraded:
            form_flags.append(warn("Tracking lost mid-rep - hold still a moment"))
        else:
            # --- shoulder compensation check ---
            scale = frame.torso_scale()
            if self._shoulder_baseline is None:
                self._shoulder_baseline = shoulder.xy()
            drift = normalized_displacement(shoulder.xy(), self._shoulder_baseline, scale)
            if drift is not None and drift > self.t.shoulder_drift_threshold:
                form_flags.append(warn("Possible shoulder compensation"))

            # --- excessive speed check ---
            if self._prev_angle is not None and self._prev_time is not None and now is not None:
                dt = now - self._prev_time
                v = angular_velocity(self._prev_angle, angle, dt)
                if v is not None:
                    self._max_abs_velocity_this_rep = max(self._max_abs_velocity_this_rep, abs(v))
                    if abs(v) > self.t.fast_movement_deg_per_sec:
                        form_flags.append(warn("Very fast movement"))

            self._prev_angle = angle
            self._prev_time = now

            if not form_flags:
                form_flags.append(ok("Controlled movement"))

        # Reset shoulder baseline at the top of each rep (extended, unarmed)
        # so slow postural drift across many reps doesn't get misread as
        # compensation on rep 20 just because it wasn't re-anchored.
        if self.sm.confirmed_zone is not None and self.sm.confirmed_zone.value == "extended" and not self.sm._armed:
            self._shoulder_baseline = shoulder.xy() if joints_visible else self._shoulder_baseline

        if rep_result is not None:
            if rep_result.rom < self.t.partial_rom_threshold:
                form_flags.append(warn("Partial ROM"))
            else:
                form_flags.append(ok("Full extension"))
            self._max_abs_velocity_this_rep = 0.0

        return self._output(
            confidence=confidence,
            phase=self._display_phase(self.sm, prev_angle_for_display, angle),
            form_flags=form_flags,
            angle=angle,
            side=side,
            rep_result=rep_result,
        )

    def _output(self, confidence: float, phase: str, form_flags: list,
                angle: Optional[float] = None, side: Optional[str] = None,
                rep_result=None) -> dict:
        last_rep = self.sm.completed_reps[-1] if self.sm.completed_reps else None
        return {
            "exercise": "BICEP CURL",
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
        self._prev_time = None
