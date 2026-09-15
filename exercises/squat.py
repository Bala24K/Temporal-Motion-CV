"""
Bodyweight squat analyzer. Same pattern as bicep_curl.py: rule-based,
built on the generic RepStateMachine, no ML.

Angle convention: knee_angle = angle_3pt(hip, knee, ankle).
  ~165-180deg = standing (extended)
  ~70-100deg  = squatting down (contracted), depending on depth

"Rest" position (standing) maps to the EXTENDED zone here — same
convention as bicep curl (contrast with shoulder_press.py and
lateral_raise.py, where rest is the CONTRACTED zone instead).
"""
from typing import Optional
from dataclasses import dataclass

from biomechanics.angles import angle_3pt, distance
from exercises.base import RepStateMachine, ExerciseAnalyzer, ok, warn
from pose.landmarks import PoseFrame


@dataclass
class SquatThresholds:
    extended_angle: float = 165.0
    contracted_angle: float = 100.0    # a moderate-depth squat; adjust per person/goal
    min_dwell_seconds: float = 0.08
    partial_rom_threshold: float = 45.0
    torso_lean_threshold_deg: float = 35.0     # forward/backward lean beyond this -> flag
    knee_valgus_ratio_threshold: float = 0.75  # inter-knee/inter-hip width ratio below this -> flag
    min_visibility: float = 0.5


class SquatAnalyzer(ExerciseAnalyzer):
    name = "squat"
    EXTENDED_LABEL = "STANDING"
    CONTRACTED_LABEL = "BOTTOM"
    DECREASING_ANGLE_LABEL = "DESCENDING"   # angle decreasing = squatting down
    INCREASING_ANGLE_LABEL = "RISING"       # angle increasing = standing back up

    def __init__(self, side: str = "auto", thresholds: Optional[SquatThresholds] = None):
        assert side in ("left", "right", "auto")
        self.side = side
        self.t = thresholds or SquatThresholds()
        self.sm = RepStateMachine(
            extended_threshold=self.t.extended_angle,
            contracted_threshold=self.t.contracted_angle,
            min_dwell_seconds=self.t.min_dwell_seconds,
        )
        self._active_side: Optional[str] = None
        self._prev_angle: Optional[float] = None

    def _resolve_side(self, frame: PoseFrame) -> Optional[str]:
        if self.side != "auto":
            return self.side
        if self._active_side is not None:
            return self._active_side
        joints = ["shoulder", "hip", "knee", "ankle"]
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

    def _torso_lean_deg(self, frame: PoseFrame, side: str) -> Optional[float]:
        hip = frame.get(f"{side}_hip")
        shoulder = frame.get(f"{side}_shoulder")
        if hip is None or shoulder is None:
            return None
        vertical_reference = (hip.x, hip.y - 1.0)  # a point straight "up" from the hip
        return angle_3pt(vertical_reference, hip.xy(), shoulder.xy())

    def _knee_valgus_ratio(self, frame: PoseFrame) -> Optional[float]:
        """
        Ratio of inter-knee distance to inter-hip distance, comparing the
        TWO KNEES TO EACH OTHER rather than one knee against its own
        hip-ankle line. That first approach was tried and rejected during
        development: any single 2D camera view shows *some* apparent
        horizontal knee shift as a knee bends forward-and-down (normal,
        expected squat motion), and a single-leg metric can't tell that
        apart from genuine medial (inward) collapse — it flagged a
        perfectly normal bend as "valgus" purely from the projection
        geometry. Comparing both knees' separation to both hips'
        separation sidesteps that: symmetric forward-and-down travel
        doesn't change the ratio much, while knees caving inward
        specifically shrinks it. Requires both knees and hips visible.
        """
        lh, rh = frame.get("left_hip"), frame.get("right_hip")
        lk, rk = frame.get("left_knee"), frame.get("right_knee")
        if not all([lh, rh, lk, rk]):
            return None
        if min(lh.visibility, rh.visibility, lk.visibility, rk.visibility) < self.t.min_visibility:
            return None
        hip_width = distance(lh.xy(), rh.xy())
        if hip_width < 1e-6:
            return None
        knee_width = distance(lk.xy(), rk.xy())
        return knee_width / hip_width

    def process(self, frame: PoseFrame, now: Optional[float] = None) -> dict:
        side = self._resolve_side(frame)
        if side is None:
            return self._output(confidence=0.0, phase="NO PERSON DETECTED",
                                 form_flags=[warn("Move into frame - can't see your leg")])

        hip = frame.get(f"{side}_hip")
        knee = frame.get(f"{side}_knee")
        ankle = frame.get(f"{side}_ankle")

        joints_visible = frame.all_visible(
            [f"{side}_hip", f"{side}_knee", f"{side}_ankle"], self.t.min_visibility)

        angle = None
        if joints_visible:
            angle = angle_3pt(hip.xy(), knee.xy(), ankle.xy())

        prev_angle_for_display = self._prev_angle
        rep_result = self.sm.update(angle, now=now)

        form_flags = []
        confidence = 1.0 if joints_visible else 0.0

        if not joints_visible:
            form_flags.append(warn("Can't see your hip/knee/ankle clearly"))
        elif self.sm.degraded:
            form_flags.append(warn("Tracking lost mid-rep - hold still a moment"))
        else:
            lean = self._torso_lean_deg(frame, side)
            if lean is not None and lean > self.t.torso_lean_threshold_deg:
                form_flags.append(warn("Excessive forward/backward lean"))

            # Only check valgus once actually descending/squatting — a
            # standing leg is trivially aligned and would just add noise.
            if angle is not None and angle < self.t.extended_angle - 15:
                ratio = self._knee_valgus_ratio(frame)
                if ratio is not None and ratio < self.t.knee_valgus_ratio_threshold:
                    form_flags.append(warn("Possible knee valgus (knees caving in)"))

            if not form_flags:
                form_flags.append(ok("Controlled movement"))

        self._prev_angle = angle

        if rep_result is not None:
            if rep_result.rom < self.t.partial_rom_threshold:
                form_flags.append(warn("Partial squat depth"))
            else:
                form_flags.append(ok("Good depth"))

        return self._output(
            confidence=confidence,
            phase=self._display_phase(self.sm, prev_angle_for_display, angle),
            form_flags=form_flags, angle=angle, side=side, rep_result=rep_result,
        )

    def _output(self, confidence, phase, form_flags, angle=None, side=None, rep_result=None) -> dict:
        last_rep = self.sm.completed_reps[-1] if self.sm.completed_reps else None
        return {
            "exercise": "SQUAT",
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
        self._prev_angle = None
