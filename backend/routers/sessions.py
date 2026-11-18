from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, selectinload

from motion_engine.contract import frames_error
from motion_engine.personalization import Baseline
from motion_engine.session import analyze_sequence
from motion_engine.skeleton import BodyProfile

from ..config import settings
from ..db import (BaselineRow, BodyProfileRow, RepRecord, SetRecord, User, Workout, WorkoutItem, WorkoutSession, now)
from ..deps import current_user, get_db
from ..state import state

router = APIRouter(prefix="/api", tags=["workouts"])


# ---------------- workouts (plans) ----------------
class ItemIn(BaseModel):
    exercise: str = Field(min_length=2, max_length=120)      # id or free text
    sets: int = Field(default=3, ge=1, le=20)
    reps: int = Field(default=10, ge=1, le=200)
    weight_kg: float = Field(default=0, ge=0, le=1000)
    rest_seconds: int = Field(default=60, ge=0, le=900)


class WorkoutIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    notes: str = Field(default="", max_length=1000)
    items: List[ItemIn] = Field(min_length=1, max_length=30)


def workout_out(w: Workout) -> dict:
    out = []
    for i in w.items:
        d = state.registry.get(i.exercise_id) if i.exercise_id else None
        out.append({"id": i.id, "exercise_id": i.exercise_id, "exercise_name": i.exercise_name, "sets": i.sets,
                    "reps": i.reps, "weight_kg": i.weight_kg, "rest_seconds": i.rest_seconds,
                    "ai_analysis": bool(d), "target_muscles": d.target_muscles if d else []})
    muscles = sorted({m for i in out for m in i["target_muscles"]})
    return {"id": w.id, "name": w.name, "notes": w.notes, "items": out, "muscles": muscles,
            "unsupported_count": sum(1 for i in out if not i["ai_analysis"])}


def _items(body: WorkoutIn) -> List[WorkoutItem]:
    items = []
    for pos, it in enumerate(body.items):
        d, _ = state.registry.resolve(it.exercise)
        items.append(WorkoutItem(position=pos, exercise_id=d.id if d else None,
                                 exercise_name=d.name if d else it.exercise.strip(),
                                 sets=it.sets, reps=it.reps, weight_kg=it.weight_kg, rest_seconds=it.rest_seconds))
    return items


@router.post("/workouts", status_code=201)
def create_workout(body: WorkoutIn, u: User = Depends(current_user), db: Session = Depends(get_db)):
    w = Workout(user_id=u.id, name=body.name, notes=body.notes, items=_items(body))
    db.add(w)
    db.commit()
    return workout_out(w)


@router.get("/workouts")
def list_workouts(u: User = Depends(current_user), db: Session = Depends(get_db)):
    ws = db.query(Workout).options(selectinload(Workout.items)).filter_by(user_id=u.id).order_by(Workout.id.desc()).all()
    return [workout_out(w) for w in ws]


def _owned_workout(db, u, wid) -> Workout:
    w = db.get(Workout, wid)
    if not w or w.user_id != u.id:
        raise HTTPException(404, "Workout not found")
    return w


@router.put("/workouts/{wid}")
def update_workout(wid: int, body: WorkoutIn, u: User = Depends(current_user), db: Session = Depends(get_db)):
    w = _owned_workout(db, u, wid)
    w.name, w.notes = body.name, body.notes
    w.items.clear()
    db.flush()
    w.items.extend(_items(body))
    db.commit()
    return workout_out(w)


@router.delete("/workouts/{wid}", status_code=204)
def delete_workout(wid: int, u: User = Depends(current_user), db: Session = Depends(get_db)):
    db.delete(_owned_workout(db, u, wid))
    db.commit()


# ---------------- sessions & sets ----------------
class StartSession(BaseModel):
    workout_id: Optional[int] = None
    name: Optional[str] = None


def owned_session(db, u, sid) -> WorkoutSession:
    s = db.get(WorkoutSession, sid)
    if not s or s.user_id != u.id:
        raise HTTPException(404, "Session not found")
    return s


@router.post("/sessions", status_code=201)
def start_session(body: StartSession, u: User = Depends(current_user), db: Session = Depends(get_db)):
    name = body.name or "Workout"
    if body.workout_id:
        name = _owned_workout(db, u, body.workout_id).name
    s = WorkoutSession(user_id=u.id, workout_id=body.workout_id, name=name)
    db.add(s)
    db.commit()
    return {"id": s.id, "name": s.name, "started_at": s.started_at.isoformat()}


class RepIn(BaseModel):
    index: int
    counted: bool = True
    peak_progress: float
    duration: float
    form_score: Optional[float] = None
    evidence: list = []


class SetIn(BaseModel):
    exercise: str = Field(min_length=2, max_length=120)
    reps: int = Field(ge=0, le=500)
    weight_kg: float = Field(default=0, ge=0, le=1000)
    set_number: int = Field(default=1, ge=1, le=50)
    partial_reps: int = Field(default=0, ge=0)
    avg_form_score: Optional[float] = None
    avg_rom: Optional[float] = None
    avg_rep_seconds: Optional[float] = None
    source: str = Field(default="manual", pattern="^(manual|ai)$")
    reps_detail: List[RepIn] = []


def save_set(db: Session, u: User, sess: WorkoutSession, body: SetIn) -> SetRecord:
    d, _ = state.registry.resolve(body.exercise)
    ai = body.source == "ai"
    if ai and not d:
        raise HTTPException(422, "AI-analysed sets require a supported exercise")
    rec = SetRecord(session_id=sess.id, user_id=u.id, exercise_id=d.id if d else None,
                    exercise_name=d.name if d else body.exercise.strip(), set_number=body.set_number, reps=body.reps,
                    partial_reps=body.partial_reps, weight_kg=body.weight_kg,
                    avg_form_score=body.avg_form_score if ai else None, avg_rom=body.avg_rom if ai else None,
                    avg_rep_seconds=body.avg_rep_seconds if ai else None, source=body.source)
    if ai:
        rec.rep_records = [RepRecord(idx=r.index, counted=r.counted, peak_progress=r.peak_progress,
                                     duration=r.duration, form_score=r.form_score, evidence=r.evidence)
                           for r in body.reps_detail]
    db.add(rec)
    db.commit()
    return rec


def set_out(s: SetRecord) -> dict:
    return {"id": s.id, "exercise_id": s.exercise_id, "exercise_name": s.exercise_name, "set_number": s.set_number,
            "reps": s.reps, "partial_reps": s.partial_reps, "weight_kg": s.weight_kg, "avg_form_score": s.avg_form_score,
            "avg_rom": s.avg_rom, "avg_rep_seconds": s.avg_rep_seconds, "source": s.source}


@router.post("/sessions/{sid}/sets", status_code=201)
def add_set(sid: int, body: SetIn, u: User = Depends(current_user), db: Session = Depends(get_db)):
    return set_out(save_set(db, u, owned_session(db, u, sid), body))


@router.post("/sessions/{sid}/finish")
def finish_session(sid: int, u: User = Depends(current_user), db: Session = Depends(get_db)):
    s = owned_session(db, u, sid)
    s.finished_at = s.finished_at or now()
    db.commit()
    return session_out(s)


def session_out(s: WorkoutSession) -> dict:
    sets = [set_out(x) for x in s.sets]
    scores = [x["avg_form_score"] for x in sets if x["avg_form_score"] is not None]
    return {"id": s.id, "name": s.name, "started_at": s.started_at.isoformat(),
            "finished_at": s.finished_at.isoformat() if s.finished_at else None,
            "duration_minutes": round(((s.finished_at or now()) - s.started_at).total_seconds() / 60, 1),
            "sets": sets, "total_reps": sum(x["reps"] for x in sets),
            "volume_kg": round(sum(x["reps"] * x["weight_kg"] for x in sets), 1),
            "avg_form_score": round(sum(scores) / len(scores), 1) if scores else None}


@router.get("/sessions")
def list_sessions(limit: int = 20, u: User = Depends(current_user), db: Session = Depends(get_db)):
    q = (db.query(WorkoutSession).options(selectinload(WorkoutSession.sets)).filter_by(user_id=u.id)
         .order_by(WorkoutSession.id.desc()).limit(min(limit, 100)))
    return [session_out(s) for s in q]


@router.get("/sessions/{sid}")
def get_session(sid: int, u: User = Depends(current_user), db: Session = Depends(get_db)):
    return session_out(owned_session(db, u, sid))


# ---------------- offline clip analysis ----------------
class AnalyzeBody(BaseModel):
    exercise: str
    frames: List[Dict[str, List[float]]]
    timestamps: List[float]
    session_id: Optional[int] = None
    weight_kg: float = 0.0


def load_context(db: Session, user_id: int, exercise_id: str):
    row = db.get(BodyProfileRow, user_id)
    prof = BodyProfile.from_dict(row.data) if row else None
    b = db.query(BaselineRow).filter_by(user_id=user_id, exercise_id=exercise_id).first()
    return prof, Baseline(b.data if b else None)


def store_baseline(db: Session, user_id: int, exercise_id: str, baseline: Baseline):
    row = db.query(BaselineRow).filter_by(user_id=user_id, exercise_id=exercise_id).first()
    if row:
        row.data = baseline.to_dict()
    else:
        db.add(BaselineRow(user_id=user_id, exercise_id=exercise_id, data=baseline.to_dict()))
    db.commit()


@router.post("/analyze")
def analyze(body: AnalyzeBody, u: User = Depends(current_user), db: Session = Depends(get_db)):
    if len(body.frames) != len(body.timestamps):
        raise HTTPException(422, "frames and timestamps must be the same length")
    if len(body.frames) > settings.max_frames_per_request:
        raise HTTPException(413, "Clip too long")
    err = frames_error(body.frames)
    if err:
        raise HTTPException(422, err)
    d, _ = state.registry.resolve(body.exercise)
    if not d:
        raise HTTPException(422, "Exercise isn't supported for AI analysis yet. Track it manually instead.")
    prof, baseline = load_context(db, u.id, d.id)
    summary = analyze_sequence(d, body.frames, body.timestamps, prof, baseline)
    if body.session_id:
        sess = owned_session(db, u, body.session_id)
        n = sum(1 for s in sess.sets if s.exercise_id == d.id) + 1
        save_set(db, u, sess, SetIn(exercise=d.id, reps=summary["reps"], partial_reps=summary["partial_reps"],
                                    weight_kg=body.weight_kg, set_number=n, avg_form_score=summary["avg_form_score"],
                                    avg_rom=summary["avg_rom"], avg_rep_seconds=summary["avg_rep_seconds"],
                                    source="ai", reps_detail=[RepIn(index=r["index"], counted=r["counted"],
                                                                     peak_progress=r["peak_progress"], duration=r["duration"],
                                                                     form_score=r["form_score"], evidence=r["evidence"])
                                                               for r in summary["reps_detail"]]))
        store_baseline(db, u.id, d.id, baseline)
    return summary
