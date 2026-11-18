"""Live WebSocket protocol: the single source of truth for /ws/live messages.

Client messages are validated on arrival (malformed -> an `error` message, never a crash or a silent ignore).
Server messages are modelled so tests can check every message the server emits, and so the TypeScript types in
web/lib/protocol.ts are GENERATED from here (`python scripts/gen_protocol_ts.py`; a test fails if they drift).

Connection lifecycle (v2):
  connect /ws/live?exercise=..&target_reps=..&session_id=..&set_key=<client uuid>&state_hz=15
  client -> {"type":"auth","token":...}             first message; the token is NOT put in the URL (logs/proxies)
  server -> {"type":"ready","set_key":..,"resumed":false,...}
  ... frames / control ...
  disconnect without end_set -> the set is parked server-side (PARK_SECONDS) and resumed on reconnect with the
  same set_key, or finished over REST: POST /api/live/sets/{set_key}/end
Legacy: `?token=` in the URL is still accepted (scripts/tests) but the web client no longer uses it.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PROTOCOL_VERSION = 2


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------------------- client -> server
class AuthMsg(_Msg):
    type: Literal["auth"]
    token: str = Field(min_length=1, max_length=4096)


class FrameMsg(_Msg):
    type: Literal["frame"]
    t: float
    landmarks: Dict[str, List[float]]


class SelectExerciseMsg(_Msg):
    type: Literal["select_exercise"]
    exercise: str = Field(min_length=1, max_length=120)
    weight_kg: float = Field(default=0, ge=0, le=1000)


class EndSetMsg(_Msg):
    type: Literal["end_set"]
    weight_kg: float = Field(default=0, ge=0, le=1000)


class RangeCaptureMsg(_Msg):
    type: Literal["range_capture"]
    action: Literal["start", "finish", "cancel"]


class ClientStatsMsg(_Msg):
    type: Literal["client_stats"]
    camera_fps: Optional[float] = None
    pose_ms: Optional[float] = None
    e2e_ms: Optional[float] = None


ClientMessage = Annotated[Union[AuthMsg, FrameMsg, SelectExerciseMsg, EndSetMsg, RangeCaptureMsg, ClientStatsMsg],
                          Field(discriminator="type")]
client_adapter: TypeAdapter = TypeAdapter(ClientMessage)


# ------------------------------------------------------------------------------------------- server -> client
class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Feedback(_Out):
    status: Literal["good", "warning", "info"]
    message: str
    rule: str


class Quality(_Out):
    ok: bool
    message: Optional[str] = None


class PersonalRange(_Out):
    rest: float
    peak: float


class ReadyOut(_Out):
    type: Literal["ready"]
    protocol: int
    set_key: Optional[str] = None
    exercise: str
    resumed: bool
    reps: int = 0


class StateOut(_Out):
    type: Literal["state"]
    exercise: str
    reps: int
    target_reps: Optional[int] = None
    partial_reps: int
    abandoned_reps: int
    phase: str
    progress: float
    quality: Quality
    tracking: Literal["ok", "degraded", "lost"]
    view: Optional[Literal["front", "side", "45deg"]] = None
    view_ok: bool
    view_hint: Optional[str] = None
    personal_range: Optional[PersonalRange] = None
    range_capture: bool
    feedback: List[Feedback]
    last_rep: Optional[Dict[str, Any]] = None
    set_complete: bool
    dropped_frames: int


class RecognitionCandidate(_Out):
    exercise: str
    members: List[str]
    score: float
    probability: float
    distinguishable: bool


class RecognitionOut(_Out):
    type: Literal["recognition"]
    status: Literal["recognized", "ambiguous", "no_movement", "unsupported"]
    action: Literal["analyze", "confirm", "wait", "fallback"]
    confidence: float
    top: Optional[RecognitionCandidate] = None
    candidates: List[RecognitionCandidate]
    message: str


class ExerciseSelectedOut(_Out):
    type: Literal["exercise_selected"]
    exercise: str
    exercise_name: str


class TargetReachedOut(_Out):
    type: Literal["target_reached"]


class SetSavedOut(_Out):
    type: Literal["set_saved"]
    saved: bool
    reason: Literal["end_set", "switch", "rest_end"]
    empty: bool = False
    set_id: Optional[int] = None
    summary: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ErrorOut(_Out):
    type: Literal["error"]
    message: str
    code: Optional[str] = None


class UnsupportedOut(_Out):
    type: Literal["unsupported"]
    message: str


class SetupOut(_Out):
    type: Literal["setup"]
    status: Literal["ok", "uncalibrated", "camera_changed", "distance_changed"]
    message: Optional[str] = None


class RangeCaptureOut(_Out):
    type: Literal["range_capture"]
    status: Literal["started", "cancelled"]
    instruction: Optional[str] = None


class RangeCapturedOut(_Out):
    type: Literal["range_captured"]
    ok: bool
    message: Optional[str] = None
    rest: Optional[float] = None
    peak: Optional[float] = None
    cycles: Optional[int] = None
    clamped: Optional[bool] = None
    measured: Optional[Dict[str, float]] = None
    population: Optional[Dict[str, float]] = None
    signal_key: Optional[str] = None


ServerMessage = Annotated[Union[ReadyOut, StateOut, RecognitionOut, ExerciseSelectedOut, TargetReachedOut, SetSavedOut,
                                ErrorOut, UnsupportedOut, SetupOut, RangeCaptureOut, RangeCapturedOut],
                          Field(discriminator="type")]
server_adapter: TypeAdapter = TypeAdapter(ServerMessage)

CLIENT_MODELS = [AuthMsg, FrameMsg, SelectExerciseMsg, EndSetMsg, RangeCaptureMsg, ClientStatsMsg]
SERVER_MODELS = [ReadyOut, StateOut, RecognitionOut, ExerciseSelectedOut, TargetReachedOut, SetSavedOut, ErrorOut,
                 UnsupportedOut, SetupOut, RangeCaptureOut, RangeCapturedOut]
