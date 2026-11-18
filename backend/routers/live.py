"""Live workout WebSocket + demo clips. Message schemas and lifecycle: backend/protocol.py.

Frames pass through a latest-wins slot (a slow consumer never works on stale data); control messages through an
ordered, lossless queue. A set belongs to a `LiveSet`, not to the socket: if the connection drops before end_set
the set is parked and resumed by a reconnect with the same `set_key` (or finished over REST), so a flaky
network no longer loses counted reps.
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from motion_engine.calibration import check_setup
from motion_engine.contract import frame_error
from motion_engine.personal_range import RangeCaptureError
from motion_engine.runtime import LatestSlot
from motion_engine.session import LiveSession
from motion_engine.skeleton import IMAGE_KEY, BodyProfile
from motion_engine.synthetic import EX, standing_stage, synth_sequence

from .. import telemetry
from ..db import BodyProfileRow, User, WorkoutSession
from ..deps import current_user, user_from_token
from ..protocol import PROTOCOL_VERSION, client_adapter
from ..state import state
from .sessions import RepIn, SetIn, load_context, save_set, store_baseline

router = APIRouter(tags=["live"])
PARK_SECONDS = 120.0           # how long a disconnected set waits for its client to come back
AUTH_TIMEOUT = 10.0
RANGE_INSTRUCTION = "Do 2-3 slow reps through your full comfortable range, then press Done."


# ============================================================================ demo clips
@router.get("/api/demo/sequence")
def demo_sequence(exercise: str = Query(...), reps: int = Query(5, ge=1, le=20), fault: Optional[str] = None,
                  fps: int = Query(30, ge=10, le=60)):
    """Synthetic clip so the product can be exercised without a camera."""
    if exercise not in EX:
        raise HTTPException(404, f"No demo for {exercise}; available: {sorted(EX)}")
    if fault not in (None, "", "shallow", "fast", "drift", "lean"):
        raise HTTPException(422, "fault must be shallow|fast|drift|lean")
    frames, ts = synth_sequence(exercise, reps=reps, fps=fps, fault=fault or None)
    return {"exercise": exercise, "fps": fps, "timestamps": ts, "frames": [_demo_frame(f) for f in frames]}


def _demo_frame(f):
    """Synthetic frames are isotropic by construction: declare a square source image so they satisfy the contract."""
    return {**{k: ([round(c, 4) for c in v] if k.startswith("w:") else [round(v[0], 4), round(v[1], 4), v[2]])
               for k, v in f.items()}, IMAGE_KEY: [1, 1]}


@router.get("/api/demo/standing")
def demo_standing(n: int = Query(40, ge=10, le=120), stage: str = Query("front", pattern="^(front|side|arms_out|arms_up)$")):
    """Synthetic calibration pose (with 3D world landmarks) for the wizard when no camera is available."""
    return {"frames": [_demo_frame(f) for f in standing_stage(stage, n)]}


# ============================================================================ a set, independent of its socket
class LiveSet:
    def __init__(self, uid: int, session_id: Optional[int], target_reps: int, defn, set_key: Optional[str]):
        self.uid, self.session_id, self.target_reps, self.set_key = uid, session_id, target_reps, set_key
        self.defn, self.sess = defn, None
        self.history: list = []
        self.last_rec = 0.0
        self.owner: Optional[str] = None             # id of the connection currently driving this set
        with state.SessionLocal() as db:
            row = db.get(BodyProfileRow, uid)
            self.profile = BodyProfile.from_dict(row.data) if row else None
        self.ctx: Dict = {"setup_frames": []}
        if defn:
            self.start(defn)

    def start(self, d):
        with state.SessionLocal() as db:
            prof, base = load_context(db, self.uid, d.id)
        self.defn = d
        self.sess = LiveSession(d, profile=prof, baseline=base, target_reps=self.target_reps)

    @property
    def exercise(self) -> str:
        return self.defn.id if self.defn and self.sess else "auto"


_active: Dict[Tuple[int, str], LiveSet] = {}
_parked: Dict[Tuple[int, str], Tuple[LiveSet, float]] = {}


def _expire_parked():
    now = time.monotonic()
    for k in [k for k, (_, exp) in _parked.items() if exp < now]:
        _parked.pop(k, None)
        state.metrics.inc("live_sets_expired_total")


def _claim(uid: int, set_key: str) -> Optional[LiveSet]:
    """A reconnect takes over its set: parked (old socket gone) or still active (old socket half-open)."""
    _expire_parked()
    key = (uid, set_key)
    if key in _parked:
        return _parked.pop(key)[0]
    return _active.get(key)


def finish_set(ls: LiveSet, weight: float, reason: str = "end_set") -> Dict:
    """Close the set: last open rep, summary, save (if it belongs to a workout session) and baseline."""
    if ls.sess is None:
        return {"type": "set_saved", "saved": False, "reason": reason, "message": "No exercise was detected."}
    ls.sess.finish()                    # close a last rep that came down but never fully returned
    summary = ls.sess.summary()
    saved, empty = None, not (summary["reps"] or summary["partial_reps"])
    if ls.session_id is not None and not empty:
        with state.SessionLocal() as db:
            u = type("U", (), {"id": ls.uid})()
            s = db.get(WorkoutSession, ls.session_id)
            n = sum(1 for x in s.sets if x.exercise_id == ls.sess.defn.id) + 1
            rec = save_set(db, u, s, SetIn(
                exercise=ls.sess.defn.id, reps=summary["reps"], partial_reps=summary["partial_reps"], weight_kg=weight,
                set_number=n, avg_form_score=summary["avg_form_score"], avg_rom=summary["avg_rom"],
                avg_rep_seconds=summary["avg_rep_seconds"], source="ai",
                reps_detail=[RepIn(index=r["index"], counted=r["counted"], peak_progress=r["peak_progress"],
                                   duration=r["duration"], form_score=r["form_score"], evidence=r["evidence"])
                             for r in summary["reps_detail"]]))
            store_baseline(db, ls.uid, ls.sess.defn.id, ls.sess.baseline)
            saved = rec.id
    out = {"type": "set_saved", "saved": saved is not None, "set_id": saved, "summary": summary, "reason": reason,
           "empty": empty}
    if empty:
        out["message"] = "No reps were detected in this set."
    return out


class EndBody(BaseModel):
    weight_kg: float = Field(default=0, ge=0, le=1000)


@router.post("/api/live/sets/{set_key}/end")
def end_parked_set(set_key: str, body: EndBody, u: User = Depends(current_user)):
    """Finish a set whose connection dropped (client could not reconnect): saves everything counted so far."""
    ls = _claim(u.id, set_key)
    if not ls:
        raise HTTPException(404, "No unfinished set with that key (it may have expired)")
    _active.pop((u.id, set_key), None)
    ls.owner = "rest"
    return finish_set(ls, body.weight_kg, "end_set")


# ============================================================================ the socket
async def _authenticate(ws: WebSocket, token: Optional[str]) -> Tuple[Optional[int], bool]:
    """(user id, used_first_message). Prefers the first-message token; ?token= / cookie are legacy fallbacks."""
    first = False
    if token is None and not ws.cookies.get("session"):
        first = True
        try:
            raw = await asyncio.wait_for(ws.receive_text(), AUTH_TIMEOUT)
            msg = client_adapter.validate_python(json.loads(raw))
            token = msg.token if msg.type == "auth" else None
        except (asyncio.TimeoutError, ValidationError, json.JSONDecodeError, WebSocketDisconnect, RuntimeError):
            token = None
    with state.SessionLocal() as db:
        user = user_from_token(db, token or ws.cookies.get("session"))
        return (user.id if user else None), first


@router.websocket("/ws/live")
async def live(ws: WebSocket, exercise: str = "auto", target_reps: int = 10, session_id: Optional[int] = None,
               token: Optional[str] = None, set_key: Optional[str] = Query(None, max_length=64),
               state_hz: float = Query(0, ge=0, le=60)):
    await ws.accept()
    uid, first_msg_auth = await _authenticate(ws, token)
    if not uid:
        await ws.send_json({"type": "error", "code": "auth", "message": "Not authenticated"})
        await ws.close(code=4401)
        return
    if session_id is not None:
        with state.SessionLocal() as db:
            s = db.get(WorkoutSession, session_id)
            if not s or s.user_id != uid:
                await ws.send_json({"type": "error", "message": "Unknown session"})
                await ws.close(code=4404)
                return

    ls = _claim(uid, set_key) if set_key else None
    resumed = ls is not None
    if not ls:
        defn = None if exercise == "auto" else state.registry.get(exercise)
        if exercise != "auto" and (not defn or defn.status == "draft"):
            await ws.send_json({"type": "unsupported",
                                "message": "This exercise isn't fully supported yet. You can still track it manually."})
            await ws.close()
            return
        ls = LiveSet(uid, session_id, target_reps, defn, set_key)
    me = uuid.uuid4().hex
    ls.owner = me
    if set_key:
        _active[(uid, set_key)] = ls
    if resumed:
        state.metrics.inc("live_sets_resumed_total")
    if token is None or set_key:                        # protocol v2 connection (no token in the URL)
        await ws.send_json({"type": "ready", "protocol": PROTOCOL_VERSION, "set_key": set_key, "exercise": ls.exercise,
                            "resumed": resumed, "reps": ls.sess.counter.count if ls.sess else 0})

    slot = LatestSlot()
    ended = False

    async def reader():
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = client_adapter.validate_python(json.loads(raw))
                except (ValidationError, json.JSONDecodeError) as e:
                    state.metrics.inc("ws_bad_messages_total")
                    slot.put_control({"type": "_error", "message": f"Malformed message: {str(e).splitlines()[0][:200]}"})
                    continue
                if msg.type == "client_stats":          # best-effort telemetry: never goes through the frame slot
                    for key, metric in (("camera_fps", "client_camera_fps"), ("pose_ms", "client_pose_ms"), ("e2e_ms", "client_e2e_ms")):
                        v = getattr(msg, key)
                        if v is not None and 0 <= v < 10_000:
                            state.metrics.observe(metric, float(v))
                elif msg.type == "auth":
                    continue
                elif msg.type == "frame":
                    slot.put(msg.model_dump())          # latest-wins: stale frames are dropped and counted
                else:
                    slot.put_control(msg.model_dump())  # ordered and lossless
                    if msg.type == "end_set":
                        return
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            slot.close()

    throttle = {"last": 0.0, "key": None}

    async def send_state(out: Dict):
        """Every frame by default; with ?state_hz=N at most N/s, except that events (rep, phase, tracking, ...)
        are always sent immediately."""
        key = (out["reps"], out["partial_reps"], out["abandoned_reps"], out["phase"], out["tracking"],
               out["set_complete"], out["range_capture"], out["quality"]["ok"], out["view_ok"])
        now = time.monotonic()
        if state_hz <= 0 or key != throttle["key"] or now - throttle["last"] >= 1.0 / state_hz:
            throttle["last"], throttle["key"] = now, key
            await ws.send_json({"type": "state", **out, "dropped_frames": slot.dropped})
        else:
            state.metrics.inc("ws_state_throttled_total")

    task = asyncio.create_task(reader())
    try:
        while True:
            msg = await slot.get()
            if msg is None:
                break
            if ls.owner != me:                          # a reconnect took this set over
                break
            kind = msg["type"]
            if kind == "_error":
                await ws.send_json({"type": "error", "code": "bad_message", "message": msg["message"]})
                continue
            if kind == "end_set":
                ended = True
                _active.pop((uid, set_key), None)
                await ws.send_json(finish_set(ls, msg["weight_kg"]))
                break
            if kind == "select_exercise":
                await _select_exercise(ws, ls, msg)
                continue
            if kind == "range_capture":
                await _range_capture(ws, ls, msg["action"])
                continue
            await _on_frame(ws, ls, msg, send_state, slot)
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        if not ended and set_key and ls.owner == me:
            _active.pop((uid, set_key), None)
            _parked[(uid, set_key)] = (ls, time.monotonic() + PARK_SECONDS)     # resume on reconnect
            state.metrics.inc("live_sets_parked_total")
        try:
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


async def _select_exercise(ws, ls: LiveSet, msg):
    target_ex = msg["exercise"]
    new_defn = state.registry.get(target_ex) if target_ex != "auto" else None
    if target_ex != "auto" and (not new_defn or new_defn.status == "draft"):
        await ws.send_json({"type": "error", "message": f"Unknown or unsupported exercise: {target_ex!r}"})
        return
    # never throw counted work away: the set in progress is saved before switching
    if ls.sess is not None and (ls.sess.counter.count or ls.sess.counter.partials):
        await ws.send_json(finish_set(ls, msg["weight_kg"], reason="switch"))
    ls.ctx.pop("announced", None)
    ls.history.clear()
    if new_defn:
        ls.start(new_defn)
        await ws.send_json({"type": "exercise_selected", "exercise": new_defn.id, "exercise_name": new_defn.name})
    else:
        ls.defn, ls.sess = None, None
        await ws.send_json({"type": "exercise_selected", "exercise": "auto", "exercise_name": "Auto-detect"})


async def _on_frame(ws, ls: LiveSet, msg, send_state, slot):
    t0 = time.perf_counter()
    frame, t = msg["landmarks"], float(msg["t"])
    err = frame_error(frame)
    if err:
        state.metrics.inc("frames_rejected_total")
        if not ls.ctx.get("contract_error_sent"):          # tell the client once, not 30 times a second
            ls.ctx["contract_error_sent"] = True
            await ws.send_json({"type": "error", "code": "frame_contract", "message": err})
        return
    if not ls.ctx.get("setup_checked"):                   # once per set: is this the calibrated camera setup?
        ls.ctx["setup_frames"].append(frame)
        if len(ls.ctx["setup_frames"]) >= 30:
            ls.ctx["setup_checked"] = True
            res = check_setup(ls.profile, ls.ctx.pop("setup_frames"))
            state.metrics.inc(f"setup_{res['status']}_total")
            if res["status"] != "ok":
                await ws.send_json({"type": "setup", **res})
    state.metrics.inc("frames_total")
    if ls.sess is None:                                   # auto-detect: buffer, then lock on
        ls.history.append((t, frame))
        ls.history[:] = ls.history[-180:]
        if len(ls.history) >= 60 and t - ls.last_rec >= 1.0:
            ls.last_rec = t
            frames = [f for _, f in ls.history]
            with telemetry.tracer.start_as_current_span("recognition") as sp:
                rec = await asyncio.to_thread(state.recognizers.recognize, frames)    # ~14 ms: off the event loop
                sp.set_attribute("recognition.status", rec.status)
                sp.set_attribute("recognition.action", rec.action)
                sp.set_attribute("recognition.confidence", float(rec.confidence))
            await ws.send_json({"type": "recognition", **rec.to_dict()})
            if rec.action == "analyze":
                ls.start(state.registry.get(rec.top.exercise))
                for ht, hf in ls.history:                 # replay so early reps still count
                    ls.sess.push(hf, ht)
                state.metrics.inc("recognition_locked_total")
            elif rec.action == "fallback":
                state.metrics.inc("recognition_unsupported_total")
        return
    out = ls.sess.push(frame, t)
    state.metrics.observe("frame_process", (time.perf_counter() - t0) * 1000)
    await send_state(out)
    if out["set_complete"] and not ls.ctx.get("announced"):
        ls.ctx["announced"] = True
        await ws.send_json({"type": "target_reached"})


async def _range_capture(ws, ls: LiveSet, action):
    """Personal range capture inside a live set: start -> (frames) -> finish | cancel. The range is applied to the
    running counter immediately and stored for future sets of this exercise."""
    sess = ls.sess
    if sess is None:
        await ws.send_json({"type": "range_captured", "ok": False, "message": "Choose an exercise first."})
        return
    if action == "start":
        sess.start_range_capture()
        await ws.send_json({"type": "range_capture", "status": "started", "instruction": RANGE_INSTRUCTION})
    elif action == "cancel":
        sess.cancel_range_capture()
        await ws.send_json({"type": "range_capture", "status": "cancelled"})
    elif action == "finish":
        try:
            rng = sess.finish_range_capture()
        except RangeCaptureError as e:
            await ws.send_json({"type": "range_captured", "ok": False, "message": str(e)})
            return
        with state.SessionLocal() as db:
            store_baseline(db, ls.uid, sess.defn.id, sess.baseline)
        state.metrics.inc("range_captured_total")
        await ws.send_json({"type": "range_captured", "ok": True, **rng})
