"""
Exponential moving average smoothing over PoseFrames.

Raw pose estimates jitter frame to frame even when the body is still.
This smooths each landmark independently, per axis. Low-confidence points
are passed through without updating the running average, so one bad
detection doesn't drag the smoothed position toward garbage.
"""
from .landmarks import PoseFrame, Landmark


class EMASmoother:
    def __init__(self, alpha: float = 0.4, min_visibility: float = 0.3):
        """alpha closer to 1.0 = less smoothing (more responsive, more jitter).
        alpha closer to 0.0 = more smoothing (less jitter, more lag)."""
        self.alpha = alpha
        self.min_visibility = min_visibility
        self._state = {}

    def smooth(self, frame: PoseFrame) -> PoseFrame:
        smoothed = PoseFrame(timestamp=frame.timestamp)
        for name, lm in frame.landmarks.items():
            if lm.visibility < self.min_visibility:
                # Don't let a low-confidence point corrupt the running average.
                # Fall back to the last good smoothed value if we have one,
                # otherwise pass the raw (unreliable) point through.
                smoothed.landmarks[name] = self._state.get(name, lm)
                continue

            prev = self._state.get(name)
            if prev is None:
                new = lm
            else:
                a = self.alpha
                new = Landmark(
                    x=a * lm.x + (1 - a) * prev.x,
                    y=a * lm.y + (1 - a) * prev.y,
                    z=a * lm.z + (1 - a) * prev.z,
                    visibility=lm.visibility,
                )
            self._state[name] = new
            smoothed.landmarks[name] = new
        return smoothed

    def reset(self):
        """Call this after tracking loss / recalibration so old positions
        don't bleed into a new session."""
        self._state = {}
