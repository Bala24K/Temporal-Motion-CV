"""Live workout WebSocket + demo clips.

Client -> server JSON:
    {"type":"frame","t":1.23,"landmarks":{"left_elbow":[x,y,vis],...}}
    {"type":"end_set","weight_kg":20}
Server -> client JSON: {"type":"state",...} | {"type":"recognition",...} | {"type":"set_saved",...}

Frames pass through a latest-wins slot, so a slow consumer never works on stale data.
"""
import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from motion_engine.contract import frame_error
from motion_engine.runtime import LatestSlot
from motion_engine.session import LiveSession
from motion_engine.skeleton import IMAGE_KEY
from motion_engine.calibration import check_setup
from motion_engine.synthetic import EX, standing_stage, synth_sequence

from ..deps import user_from_token
from ..db import BodyProfileRow, WorkoutSession
from motion_engine.skeleton import BodyProfile
from .. import telemetry
from ..state import state
from .sessions import RepIn, SetIn, load_context, save_set, store_baseline

router = APIRouter(tags=["live"])


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


@router.websocket("/ws/live")
async def live(ws: WebSocket, exercise: str = "auto", target_reps: int = 10, session_id: Optional[int] = None,
               token: Optional[str] = None):
    await ws.accept()
    with state.SessionLocal() as db:
        user = user_from_token(db, token or ws.cookies.get("session"))
        if not user:
            await ws.send_json({"type": "error", "message": "Not authenticated"})
            await ws.close(code=4401)
            return
        uid = user.id
        if session_id is not None:
            s = db.get(WorkoutSession, session_id)
            if not s or s.user_id != uid:
                await ws.send_json({"type": "error", "message": "Unknown session"})
                await ws.close(code=4404)
                return

    defn = None if exercise == "auto" else state.registry.get(exercise)
    if exercise != "auto" and (not defn or defn.status == "draft"):
        await ws.send_json({"type": "unsupported",
                            "message": "This exercise isn't fully supported yet. You can still track it manually."})
        await ws.close()
        return

    slot, sess, history, last_rec = LatestSlot(), None, [], 0.0
    with state.SessionLocal() as db:
        row = db.get(BodyProfileRow, uid)
        ctx = {"profile": BodyProfile.from_dict(row.data) if row else None, "setup_frames": []}

    def start(d):
        with state.SessionLocal() as db:
            prof, base = load_context(db, uid, d.id)
        ctx["baseline"] = base
        return LiveSession(d, profile=prof, baseline=base, target_reps=target_reps)

    if defn:
        sess = start(defn)

    async def reader():
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "client_stats":      # best-effort telemetry: never goes through the frame slot
                    for key, metric in (("camera_fps", "client_camera_fps"), ("pose_ms", "client_pose_ms"), ("e2e_ms", "client_e2e_ms")):
                        v = msg.get(key)
                        if isinstance(v, (int, float)) and 0 <= v < 10_000:
                            state.metrics.observe(metric, float(v))
                    continue
                if msg.get("type") == "frame":
                    slot.put(msg)                       # latest-wins: stale frames are dropped and counted
                else:
                    slot.put_control(msg)               # ordered and lossless (select_exercise, end_set, ...)
                    if msg.get("type") == "end_set":
                        return
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            slot.close()

    task = asyncio.create_task(reader())
    try:
        while True:
            msg = await slot.get()
            if msg is None:
                break
            kind = msg.get("type")
            if kind == "end_set":
                await _finish(ws, sess, uid, session_id, float(msg.get("weight_kg", 0) or 0), ctx)
                break
            if kind == "select_exercise":
                target_ex = msg.get("exercise")
                new_defn = state.registry.get(target_ex) if target_ex and target_ex != "auto" else None
                if target_ex != "auto" and (not new_defn or new_defn.status == "draft"):
                    await ws.send_json({"type": "error", "message": f"Unknown or unsupported exercise: {target_ex!r}"})
                    continue
                # never throw counted work away: the set in progress is saved before switching
                if sess is not None and (sess.counter.count or sess.counter.partials):
                    await _finish(ws, sess, uid, session_id, float(msg.get("weight_kg", 0) or 0), ctx, reason="switch")
                ctx.pop("announced", None)
                history.clear()
                if new_defn:
                    defn, sess = new_defn, start(new_defn)
                    await ws.send_json({"type": "exercise_selected", "exercise": new_defn.id, "exercise_name": new_defn.name})
                else:
                    defn, sess = None, None
                    await ws.send_json({"type": "exercise_selected", "exercise": "auto", "exercise_name": "Auto-detect"})
                continue
            if kind != "frame":
                continue
            t0 = time.perf_counter()
            frame, t = msg.get("landmarks", {}), float(msg.get("t", time.time()))
            err = frame_error(frame)
            if err:
                state.metrics.inc("frames_rejected_total")
                if not ctx.get("contract_error_sent"):          # tell the client once, not 30 times a second
                    ctx["contract_error_sent"] = True
                    await ws.send_json({"type": "error", "code": "frame_contract", "message": err})
                continue
            if not ctx.get("setup_checked"):                   # once per set: is this the calibrated camera setup?
                ctx["setup_frames"].append(frame)
                if len(ctx["setup_frames"]) >= 30:
                    ctx["setup_checked"] = True
                    res = check_setup(ctx["profile"], ctx.pop("setup_frames"))
                    state.metrics.inc(f"setup_{res['status']}_total")
                    if res["status"] != "ok":
                        await ws.send_json({"type": "setup", **res})
            state.metrics.inc("frames_total")
            if sess is None:                                   # auto-detect: buffer, then lock on
                history.append((t, frame))
                history[:] = history[-180:]
                if len(history) >= 60 and t - last_rec >= 1.0:
                    last_rec = t
                    with telemetry.tracer.start_as_current_span("recognition") as sp:
                        rec = state.recognizers.recognize([f for _, f in history])
                        sp.set_attribute("recognition.status", rec.status)
                        sp.set_attribute("recognition.action", rec.action)
                        sp.set_attribute("recognition.confidence", float(rec.confidence))
                    await ws.send_json({"type": "recognition", **rec.to_dict()})
                    if rec.action == "analyze":
                        sess = start(state.registry.get(rec.top.exercise))
                        for ht, hf in history:                 # replay so early reps still count
                            sess.push(hf, ht)
                        state.metrics.inc("recognition_locked_total")
                    elif rec.action == "fallback":
                        state.metrics.inc("recognition_unsupported_total")
                continue
            out = sess.push(frame, t)
            state.metrics.observe("frame_process", (time.perf_counter() - t0) * 1000)
            await ws.send_json({"type": "state", **out, "dropped_frames": slot.dropped})
            if out["set_complete"] and not ctx.get("announced"):
                ctx["announced"] = True
                await ws.send_json({"type": "target_reached"})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        try:
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


async def _finish(ws, sess, uid, session_id, weight, ctx, reason="end_set"):
    if sess is None:
        await ws.send_json({"type": "set_saved", "saved": False, "reason": reason, "message": "No exercise was detected."})
        return
    summary = sess.summary()
    saved = None
    if session_id is not None and (summary["reps"] or summary["partial_reps"]):
        with state.SessionLocal() as db:
            u = type("U", (), {"id": uid})()
            s = db.get(WorkoutSession, session_id)
            n = sum(1 for x in s.sets if x.exercise_id == sess.defn.id) + 1
            rec = save_set(db, u, s, SetIn(
                exercise=sess.defn.id, reps=summary["reps"], partial_reps=summary["partial_reps"], weight_kg=weight,
                set_number=n, avg_form_score=summary["avg_form_score"], avg_rom=summary["avg_rom"],
                avg_rep_seconds=summary["avg_rep_seconds"], source="ai",
                reps_detail=[RepIn(index=r["index"], counted=r["counted"], peak_progress=r["peak_progress"],
                                   duration=r["duration"], form_score=r["form_score"], evidence=r["evidence"])
                             for r in summary["reps_detail"]]))
            store_baseline(db, uid, sess.defn.id, sess.baseline)
            saved = rec.id
    await ws.send_json({"type": "set_saved", "saved": saved is not None, "set_id": saved, "summary": summary, "reason": reason})
