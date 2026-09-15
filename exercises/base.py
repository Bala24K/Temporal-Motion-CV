"""
Generic rep-counting engine, driven by a single scalar "primary angle" per
frame (elbow angle for a curl, knee angle for a squat, etc).

Design notes:

- This is a Schmitt-trigger-style two-threshold counter, not a naive
  `if angle < X: rep += 1`. A rep only counts on the full cycle
  EXTENDED -> (leave extended) -> CONTRACTED (confirmed) -> (leave
  contracted) -> EXTENDED (confirmed again). You cannot double-count by
  jittering near one threshold, because a second EXTENDED confirmation
  only completes a rep if a CONTRACTED confirmation happened in between.

- "Confirmed" means the reading has held inside a zone for at least
  MIN_DWELL_SECONDS, not just touched it for one frame. This is a
  time-based debounce (not a frame-count one) so it behaves the same at
  15 FPS and at 30 FPS.

- A stuck state (e.g. someone stops mid-rep and walks away) times out
  and resets rather than hanging forever waiting to "complete" a rep that
  isn't coming.

- Missing/None angle readings (occlusion, low confidence) freeze
  everything rather than being treated as a real angle. If tracking stays
  lost past DEGRADED_TIMEOUT_SECONDS, the machine flags `degraded` and,
  once tracking recovers, re-anchors cleanly instead of completing a rep
  off stale extremes.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import time


class Zone(Enum):
    EXTENDED = "extended"
    MIDDLE = "middle"
    CONTRACTED = "contracted"


@dataclass
class RepResult:
    rep_number: int
    start_time: float          # when the rep's concentric phase began
    end_time: float            # when the rep completed (back to extended)
    min_angle: float
    max_angle: float
    concentric_duration: float
    eccentric_duration: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def rom(self) -> float:
        return self.max_angle - self.min_angle


class RepStateMachine:
    MIN_DWELL_SECONDS = 0.08        # ~2-3 frames at 30fps; filters single-frame glitches
    STUCK_TIMEOUT_SECONDS = 8.0     # give up on a hanging rep attempt after this long
    DEGRADED_TIMEOUT_SECONDS = 2.0  # how long we tolerate missing angle readings

    def __init__(self, extended_threshold: float, contracted_threshold: float,
                 min_dwell_seconds: Optional[float] = None, alpha: float = 1.0):
        assert extended_threshold > contracted_threshold, \
            "extended_threshold must be greater than contracted_threshold"
        self.extended_threshold = extended_threshold
        self.contracted_threshold = contracted_threshold
        self.min_dwell_seconds = min_dwell_seconds or self.MIN_DWELL_SECONDS
        self.alpha = alpha

        self.confirmed_zone: Optional[Zone] = None
        self._candidate_zone: Optional[Zone] = None
        self._candidate_since: Optional[float] = None
        self._zone_entered_at: Optional[float] = None
        self._last_update_at: Optional[float] = None
        self._prev_raw_zone: Optional[Zone] = None

        self.smoothed_angle: Optional[float] = None
        self._armed = False  # True once CONTRACTED has been confirmed after EXTENDED start
        self._rep_min_angle: Optional[float] = None
        self._rep_max_angle: Optional[float] = None
        self._extended_left_at: Optional[float] = None
        self._contracted_confirmed_at: Optional[float] = None
        self._suppress_confirm_zone: Optional[Zone] = None

        self.rep_count = 0
        self.completed_reps: List[RepResult] = []
        self.degraded = False

    def raw_zone(self, angle: float) -> Zone:
        if angle >= self.extended_threshold:
            return Zone.EXTENDED
        if angle <= self.contracted_threshold:
            return Zone.CONTRACTED
        return Zone.MIDDLE

    @property
    def _has_confirmed_start(self) -> bool:
        return self.confirmed_zone is not None

    @property
    def phase_label(self) -> str:
        if self.degraded:
            return "TRACKING_LOST"
        if self.confirmed_zone == Zone.EXTENDED and not self._armed:
            return "AT_EXTENDED"
        if self._armed and self.confirmed_zone != Zone.CONTRACTED:
            return "MOVING"
        if self.confirmed_zone == Zone.CONTRACTED:
            return "AT_CONTRACTED"
        return "READY"

    def update(self, angle: Optional[float], now: Optional[float] = None) -> Optional[RepResult]:
        """Feed one frame's primary-angle reading. Returns a RepResult if a rep completed on this call, otherwise None."""
        now = now if now is not None else time.time()

        if angle is None:
            if self._last_update_at is not None and now - self._last_update_at > self.DEGRADED_TIMEOUT_SECONDS:
                self.degraded = True
            return None

        if self.degraded:
            self._reanchor()

        # Apply EMA smoothing to reduce MediaPipe joint jitter
        if self.smoothed_angle is None:
            self.smoothed_angle = float(angle)
        else:
            self.smoothed_angle = float(self.alpha * angle + (1.0 - self.alpha) * self.smoothed_angle)

        eval_angle = self.smoothed_angle

        self._last_update_at = now
        self._rep_min_angle = eval_angle if self._rep_min_angle is None else min(self._rep_min_angle, eval_angle)
        self._rep_max_angle = eval_angle if self._rep_max_angle is None else max(self._rep_max_angle, eval_angle)

        zone = self.raw_zone(eval_angle)

        if self._suppress_confirm_zone is not None and zone != self._suppress_confirm_zone:
            self._suppress_confirm_zone = None

        if self._prev_raw_zone == Zone.EXTENDED and zone != Zone.EXTENDED and self._extended_left_at is None:
            self._extended_left_at = now
        self._prev_raw_zone = zone

        if zone != self._candidate_zone:
            self._candidate_zone = zone
            self._candidate_since = now
        dwell_ok = self._candidate_since is not None and \
            (now - self._candidate_since) >= self.min_dwell_seconds

        result = None

        if zone == Zone.EXTENDED and dwell_ok and self.confirmed_zone != Zone.EXTENDED \
                and self._suppress_confirm_zone != Zone.EXTENDED:
            if self._armed:
                result = self._complete_rep(now)
            self.confirmed_zone = Zone.EXTENDED
            self._zone_entered_at = now
            self._extended_left_at = None
            self._rep_min_angle = eval_angle
            self._rep_max_angle = eval_angle

        elif zone == Zone.CONTRACTED and dwell_ok and self.confirmed_zone != Zone.CONTRACTED \
                and self._suppress_confirm_zone != Zone.CONTRACTED:
            self.confirmed_zone = Zone.CONTRACTED
            self._zone_entered_at = now
            self._contracted_confirmed_at = now
            self._armed = True

        if self._armed and self._zone_entered_at is not None and \
           (now - self._zone_entered_at) > self.STUCK_TIMEOUT_SECONDS:
            self._armed = False
            self._suppress_confirm_zone = self.confirmed_zone
            self.confirmed_zone = None
            self._zone_entered_at = None
            self._extended_left_at = None
            self._contracted_confirmed_at = None
            self._rep_min_angle = None
            self._rep_max_angle = None

        return result

    def _complete_rep(self, now: float) -> Optional[RepResult]:
        start = self._extended_left_at or self._contracted_confirmed_at or self._zone_entered_at or now
        concentric_end = self._contracted_confirmed_at or now
        dur = max(0.0, now - start)
        min_a = self._rep_min_angle if self._rep_min_angle is not None else 0.0
        max_a = self._rep_max_angle if self._rep_max_angle is not None else 0.0
        rom_val = max_a - min_a

        # Validate minimum rep duration and ROM excursion to reject noise / partial movements
        if dur < 0.05 or rom_val < 15.0:
            self._armed = False
            self._contracted_confirmed_at = None
            return None

        self.rep_count += 1
        rep = RepResult(
            rep_number=self.rep_count,
            start_time=start,
            end_time=now,
            min_angle=min_a,
            max_angle=max_a,
            concentric_duration=max(0.0, concentric_end - start),
            eccentric_duration=max(0.0, now - concentric_end),
        )
        self.completed_reps.append(rep)
        self._armed = False
        self._contracted_confirmed_at = None
        return rep

    def _reanchor(self):
        self.degraded = False
        self.smoothed_angle = None
        self._armed = False
        self.confirmed_zone = None
        self._candidate_zone = None
        self._candidate_since = None
        self._rep_min_angle = None
        self._rep_max_angle = None
        self._extended_left_at = None
        self._contracted_confirmed_at = None
        self._prev_raw_zone = None
        self._suppress_confirm_zone = None


def direction_from_angles(prev_angle: Optional[float], curr_angle: Optional[float]) -> Optional[str]:
    """'increasing' | 'decreasing' | None (no prior reading, or no real change)."""
    if prev_angle is None or curr_angle is None:
        return None
    if curr_angle > prev_angle + 1e-6:
        return "increasing"
    if curr_angle < prev_angle - 1e-6:
        return "decreasing"
    return None


def ok(message: str) -> dict:
    """A positive/neutral form flag. Kept as a {level, message} dict rather
    than a pre-formatted string with a glyph prefix (e.g. a checkmark) —
    cv2.putText only supports Hershey fonts, which can't render checkmarks,
    warning triangles, or most Unicode symbols; they'd silently show up as
    '???' in the OpenCV UI. Let the UI layer decide how to represent
    "good"/"warn" visually (color, icon shape, etc.)."""
    return {"level": "ok", "message": message}


def warn(message: str) -> dict:
    return {"level": "warn", "message": message}


class ExerciseAnalyzer:
    """Common interface every exercise module implements."""
    name = "base"

    # Subclasses override these four to get exercise-appropriate display
    # text out of _display_phase() below, without needing to reimplement
    # the direction logic each time.
    EXTENDED_LABEL = "EXTENDED"
    CONTRACTED_LABEL = "CONTRACTED"
    INCREASING_ANGLE_LABEL = "MOVING"   # shown while angle is increasing
    DECREASING_ANGLE_LABEL = "MOVING"   # shown while angle is decreasing

    def _display_phase(self, sm, prev_angle: Optional[float], curr_angle: Optional[float]) -> str:
        """
        Real-time phase display, deliberately DECOUPLED from the
        confirmed_zone/armed bookkeeping used for robust rep counting.

        This look simpler because it has to be: an earlier version tried
        to derive the live display from confirmed_zone/armed (the same
        state the counter uses), and hit two bugs in a row from doing so.
        First, a literal dead branch (CURLING vs LOWERING could never both
        be reached, since the two state variables that distinguished them
        are always set/cleared together). Then, after "fixing" that, a
        deeper issue: confirmed_zone only updates on a NEW confirmation,
        so it stays pinned to its last-confirmed value for the entire
        transition back the other way — the display would show
        "AT_CONTRACTED" for the whole ascent back to extended, then jump
        straight to "AT_EXTENDED" the instant the rep completes, never
        passing through anything resembling "moving." That lag is exactly
        the point for rep-counting robustness (don't let one noisy frame
        flip the count) but it's the wrong tradeoff for a live phase
        label, where a little jitter in the displayed word doesn't matter
        and multi-second staleness does. So: use the RAW (undebounced)
        zone for display, and `degraded` (which IS meant to be immediate).
        """
        if sm.degraded:
            return "TRACKING LOST"
        if curr_angle is None:
            return "READY"
        zone = sm.raw_zone(curr_angle)
        if zone == Zone.EXTENDED:
            return self.EXTENDED_LABEL
        if zone == Zone.CONTRACTED:
            return self.CONTRACTED_LABEL
        direction = direction_from_angles(prev_angle, curr_angle)
        if direction == "increasing":
            return self.INCREASING_ANGLE_LABEL
        if direction == "decreasing":
            return self.DECREASING_ANGLE_LABEL
        return "MOVING"

    def process(self, pose_frame, now: Optional[float] = None) -> dict:
        """Returns a dict with at least:
        exercise, rep_count, phase, rom, tempo, form_messages, confidence
        Subclasses implement this."""
        raise NotImplementedError
