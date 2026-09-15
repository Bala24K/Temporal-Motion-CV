"""
Lateral raise analyzer. Rule-based, same RepStateMachine as the other
exercises.

Angle convention: arm_elevation_angle = angle_3pt(hip, shoulder, wrist) —
the angle at the shoulder between the torso line (shoulder->hip) and the
upper arm (shoulder->wrist). Arm hanging at the side points in roughly the
same direction as the torso line, so this is near 0 degrees at rest and
rises toward ~90 degrees as the arm lifts to shoulder height.

Like shoulder_press.py, this is the FLIPPED convention: rest (arm down at
the side) is the CONTRACTED (low-angle) zone; the peak (arm raised to
shoulder height) is EXTENDED. See shoulder_press.py's docstring for how
RepStateMachine handles this without any special-casing.
"""
from typing import Optional
from dataclasses import dataclass

from biomechanics.angles import angle_3pt
from exercises.base import RepStateMachine, ExerciseAnalyzer, ok, warn
from pose.landmarks import PoseFrame


@dataclass
class LateralRaiseThresholds:
    extended_angle: float = 75.0     # arm raised to roughly shoulder height
    contracted_angle: float = 15.0   # arm hanging at the side
    min_dwell_seconds: float = 0.08
    partial_rom_threshold: float = 40.0
    shoulder_shrug_threshold: float = 0.08   # normalized upward shoulder displacement -> flag
    min_visibility: float = 0.5


class LateralRaiseAnalyzer(ExerciseAnalyzer):
    name = "lateral_raise"
    EXTENDED_LABEL = "RAISED"
    CONTRACTED_LABEL = "AT SIDE"
    INCREASING_ANGLE_LABEL = "RAISING"   # angle increasing = lifting the arm
    DECREASING_ANGLE_LABEL = "LOWERING"  # angle decreasing = returning to the side

    def __init__(self, side: str = "auto", thresholds: Optional[LateralRaiseThresholds] = None):
        assert side in ("left", "right", "auto")
        self.side = side
        self.t = thresholds or LateralRaiseThresholds()
        self.sm = RepStateMachine(
            extended_threshold=self.t.extended_angle,
            contracted_threshold=self.t.contracted_angle,
            min_dwell_seconds=self.t.min_dwell_seconds,
        )
        self._active_side: Optional[str] = None
        self._shoulder_y_baseline: Optional[float] = None
        self._prev_angle: Optional[float] = None

    def _resolve_side(self, frame: PoseFrame) -> Optional[str]:
        if self.side != "auto":
            return self.side
        if self._active_side is not None:
            return self._active_side
        joints = ["hip", "shoulder", "wrist"]
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

        hip = frame.get(f"{side}_hip")
        shoulder = frame.get(f"{side}_shoulder")
        wrist = frame.get(f"{side}_wrist")

        joints_visible = frame.all_visible(
            [f"{side}_hip", f"{side}_shoulder", f"{side}_wrist"], self.t.min_visibility)

        angle = None
        if joints_visible:
            angle = angle_3pt(hip.xy(), shoulder.xy(), wrist.xy())

        prev_angle_for_display = self._prev_angle
        rep_result = self.sm.update(angle, now=now)

        form_flags = []
        confidence = 1.0 if joints_visible else 0.0

        if not joints_visible:
            form_flags.append(warn("Can't see your hip/shoulder/wrist clearly"))
        elif self.sm.degraded:
            form_flags.append(warn("Tracking lost mid-rep - hold still a moment"))
        else:
            scale = frame.torso_scale()
            if self._shoulder_y_baseline is None:
                self._shoulder_y_baseline = shoulder.y
            if scale:
                # Image y increases DOWNWARD, so a shrug (shoulder rising)
                # means shoulder.y DECREASING relative to baseline.
                upward_shift = (self._shoulder_y_baseline - shoulder.y) / scale
                if upward_shift > self.t.shoulder_shrug_threshold:
                    form_flags.append(warn("Possible shoulder shrug - relax your traps"))

            if not form_flags:
                form_flags.append(ok("Controlled movement"))

        # Reset the shoulder-height baseline at rest (arm at side,
        # CONTRACTED zone, not mid-rep) — same "flipped rest" convention
        # as shoulder_press.py.
        if self.sm.confirmed_zone is not None and self.sm.confirmed_zone.value == "contracted" and not self.sm._armed:
            self._shoulder_y_baseline = shoulder.y if joints_visible else self._shoulder_y_baseline

        self._prev_angle = angle

        if rep_result is not None:
            if rep_result.rom < self.t.partial_rom_threshold:
                form_flags.append(warn("Partial range - raise higher"))
            else:
                form_flags.append(ok("Good range of motion"))

        return self._output(
            confidence=confidence,
            phase=self._display_phase(self.sm, prev_angle_for_display, angle),
            form_flags=form_flags, angle=angle, side=side, rep_result=rep_result,
        )

    def _output(self, confidence, phase, form_flags, angle=None, side=None, rep_result=None) -> dict:
        last_rep = self.sm.completed_reps[-1] if self.sm.completed_reps else None
        return {
            "exercise": "LATERAL RAISE",
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
        self._shoulder_y_baseline = None
        self._prev_angle = None
