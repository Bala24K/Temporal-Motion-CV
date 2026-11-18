import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import UnsupportedRequest, UnsupportedRequestUser, User, now
from ..deps import current_user, get_db
from ..state import state

router = APIRouter(prefix="/api/exercises", tags=["exercises"])


def public(d, detail: bool = False) -> dict:
    out = {"id": d.id, "name": d.name, "family": d.family, "status": d.status, "difficulty": d.difficulty,
           "target_muscles": d.target_muscles, "secondary_muscles": d.secondary_muscles,
           "equipment": d.equipment, "movement_patterns": d.movement_patterns,
           "camera": d.camera.model_dump(), "body_position": d.body_position, "ai_analysis": True}
    if detail:
        out.update(aliases=d.aliases, primary_joints=d.primary_joints, version=d.version,
                   form_checks=[r.good or r.id for r in d.form_rules if r.good])
    return out


@router.get("")
def list_exercises(q: Optional[str] = None, muscle: Optional[str] = None, equipment: Optional[str] = None,
                   pattern: Optional[str] = None):
    items = state.registry.all()
    if q:
        ql = q.lower()
        items = [d for d in items if ql in d.name.lower() or any(ql in a for a in d.aliases)]
    if muscle:
        items = [d for d in items if muscle.lower() in [m.lower() for m in d.target_muscles + d.secondary_muscles]]
    if equipment:
        items = [d for d in items if equipment.lower() in [e.lower() for e in d.equipment]]
    if pattern:
        items = [d for d in items if pattern.lower() in d.movement_patterns]
    groups = {}
    for d in items:
        groups.setdefault(d.target_muscles[0] if d.target_muscles else "other", []).append(public(d))
    return {"count": len(items), "exercises": [public(d) for d in items], "by_muscle": groups}


@router.get("/{exercise_id}")
def get_exercise(exercise_id: str):
    d = state.registry.get(exercise_id)
    if not d or d.status == "draft":
        raise HTTPException(404, "Unknown exercise")
    return public(d, detail=True)


class ResolveBody(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    note: str = Field(default="", max_length=500)


def log_unsupported(db: Session, user: User, name: str) -> UnsupportedRequest:
    key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()[:120]
    row = db.query(UnsupportedRequest).filter_by(key=key).first()
    pattern, closest = state.registry.suggest_pattern(name), state.registry.closest(name)
    if not row:
        row = UnsupportedRequest(key=key, display_name=name.strip()[:120], request_count=0,
                                 candidate_pattern=pattern, closest=closest)
        db.add(row)
        db.flush()
    row.request_count += 1
    row.last_seen = now()
    if not db.get(UnsupportedRequestUser, (row.id, user.id)):
        db.add(UnsupportedRequestUser(request_id=row.id, user_id=user.id))
    db.commit()
    return row


@router.post("/resolve")
def resolve(body: ResolveBody, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Free text -> supported definition, or an honest 'not supported yet' + a logged request."""
    state.metrics.inc("exercise_resolve_total")
    d, score = state.registry.resolve(body.name)
    if d:
        return {"supported": True, "match_confidence": round(score, 2), "exercise": public(d)}
    state.metrics.inc("exercise_unsupported_total")
    log_unsupported(db, user, body.name)
    return {
        "supported": False,
        "message": ("This exercise isn't fully supported yet. You can still track it manually - "
                    "reliable rep and form analysis isn't available for it right now."),
        "manual_tracking": True, "similar": [public(state.registry.get(i)) for i in state.registry.closest(body.name)],
    }
