"""
Session State Machine & Metrics Tracker.

Manages application states:
  - SETUP: Calibration mode showing full-body camera positioning guide.
  - RUNNING: Active workout session, tracking reps, phase, confidence, and form.
  - PAUSED: Frozen timer and rep state.
  - SUMMARY: Post-session summary view displaying stats & local JSON export option.
"""

from enum import Enum
import json
import os
import time
from typing import Any, Dict, List, Optional


class SessionState(Enum):
    SETUP = "SETUP"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    SUMMARY = "SUMMARY"


class SessionTracker:
    """Tracks session timing, metrics, form feedback, rep history, and summary export."""

    def __init__(self, exercise_mode: str = "AUTO"):
        self.state = SessionState.SETUP
        self.exercise_mode = exercise_mode

        self.start_time: Optional[float] = None
        self.pause_start: Optional[float] = None
        self.total_paused_duration: float = 0.0

        self.confidence_history: List[float] = []
        self.fps_history: List[float] = []
        self.latency_history: List[float] = []

        self.rep_events: List[Dict[str, Any]] = []
        self.last_rep_count: int = 0

        self.good_form_count: int = 0
        self.warning_form_count: int = 0
        self.form_messages_seen: Dict[str, str] = {}  # msg -> level

        self.splash_banner_text: Optional[str] = None
        self.splash_banner_until: float = 0.0

        self.active_exercise_name: str = "Detecting..."

    def start_session(self) -> None:
        """Transitions from SETUP to RUNNING state."""
        self.state = SessionState.RUNNING
        self.start_time = time.time()
        self.total_paused_duration = 0.0
        self.pause_start = None

    def pause_session(self) -> None:
        """Toggles between RUNNING and PAUSED states."""
        if self.state == SessionState.RUNNING:
            self.state = SessionState.PAUSED
            self.pause_start = time.time()
        elif self.state == SessionState.PAUSED:
            self.state = SessionState.RUNNING
            if self.pause_start is not None:
                self.total_paused_duration += time.time() - self.pause_start
                self.pause_start = None

    def reset_session(self) -> None:
        """Resets rep counts, timers, and metric histories."""
        self.start_time = time.time() if self.state == SessionState.RUNNING else None
        self.pause_start = None
        self.total_paused_duration = 0.0
        self.confidence_history.clear()
        self.fps_history.clear()
        self.latency_history.clear()
        self.rep_events.clear()
        self.last_rep_count = 0
        self.good_form_count = 0
        self.warning_form_count = 0
        self.form_messages_seen.clear()
        self.splash_banner_text = None
        self.splash_banner_until = 0.0

    def end_session(self) -> None:
        """Transitions state to SUMMARY."""
        if self.state == SessionState.PAUSED and self.pause_start is not None:
            self.total_paused_duration += time.time() - self.pause_start
            self.pause_start = None
        self.state = SessionState.SUMMARY

    def update_frame_metrics(self, engine_res: Dict[str, Any], fps: float, latency_ms: float, now: float) -> None:
        """Updates frame metrics when in RUNNING state."""
        if self.state != SessionState.RUNNING:
            return

        if engine_res.get("valid_pose", False):
            conf = engine_res.get("confidence", 0.0)
            self.confidence_history.append(conf)
            self.active_exercise_name = engine_res.get("exercise", "Detecting...")

        if fps > 0:
            self.fps_history.append(fps)
        if latency_ms > 0:
            self.latency_history.append(latency_ms)

        # Rep completion splash check
        curr_reps = engine_res.get("rep_count", 0)
        if curr_reps > self.last_rep_count:
            self.last_rep_count = curr_reps
            self.splash_banner_text = f"REP {curr_reps} \u2713"
            self.splash_banner_until = now + 1.5
            self.rep_events.append({
                "rep_number": curr_reps,
                "timestamp": now,
                "exercise": self.active_exercise_name,
                "duration": engine_res.get("tempo"),
                "rom": engine_res.get("rom"),
            })

        # Form feedback tracking
        for fb in engine_res.get("form_feedback", []):
            msg = fb.get("message", "")
            lvl = fb.get("status", "good")
            if msg:
                self.form_messages_seen[msg] = lvl
                if lvl in ("good", "ok"):
                    self.good_form_count += 1
                else:
                    self.warning_form_count += 1

    def get_elapsed_seconds(self) -> float:
        """Calculates active session elapsed duration in seconds."""
        if self.start_time is None:
            return 0.0
        if self.state == SessionState.PAUSED and self.pause_start is not None:
            end_t = self.pause_start
        else:
            end_t = time.time()
        return max(0.0, end_t - self.start_time - self.total_paused_duration)

    def get_elapsed_formatted(self) -> str:
        """Returns MM:SS formatted duration string."""
        total_sec = int(self.get_elapsed_seconds())
        mins = total_sec // 60
        secs = total_sec % 60
        return f"{mins:02d}:{secs:02d}"

    def get_summary(self) -> Dict[str, Any]:
        """Calculates post-session summary metrics."""
        avg_conf = (sum(self.confidence_history) / len(self.confidence_history) * 100.0) if self.confidence_history else 0.0
        avg_fps = (sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0.0
        avg_lat = (sum(self.latency_history) / len(self.latency_history)) if self.latency_history else 0.0

        return {
            "exercise_mode": self.exercise_mode,
            "detected_exercise": self.active_exercise_name,
            "total_reps": self.last_rep_count,
            "duration_formatted": self.get_elapsed_formatted(),
            "duration_seconds": round(self.get_elapsed_seconds(), 2),
            "avg_confidence_pct": round(avg_conf, 1),
            "avg_fps": round(avg_fps, 1),
            "avg_latency_ms": round(avg_lat, 1),
            "good_form_checks": self.good_form_count,
            "warning_form_checks": self.warning_form_count,
            "rep_details": self.rep_events,
            "unique_form_feedback": list(self.form_messages_seen.items()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save_summary_json(self, output_dir: str = "sessions") -> str:
        """Saves session summary to a local JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        data = self.get_summary()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Session summary saved to '{filepath}'.")
        return filepath
