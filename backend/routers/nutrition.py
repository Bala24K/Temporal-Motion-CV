from datetime import date as Date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, selectinload

from nutrition.energy import classify_intensity, estimate_session_kcal
from nutrition.foods import FOODS, macros_for, search
from nutrition.targets import ProfileIncomplete, SEX_CONST, ACTIVITY, targets

from ..db import FoodLog, User, WorkoutSession, now
from ..deps import current_user, get_db

router = APIRouter(prefix="/api/nutrition", tags=["nutrition"])


class NutritionProfile(BaseModel):
    sex: Optional[str] = Field(default=None, pattern="^(male|female|unspecified)$")
    birth_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    height_cm: Optional[float] = Field(default=None, ge=100, le=250)
    weight_kg: Optional[float] = Field(default=None, ge=30, le=400)
    activity_level: Optional[str] = None


@router.patch("/profile")
def set_profile(body: NutritionProfile, u: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.activity_level is not None and body.activity_level not in ACTIVITY:
        raise HTTPException(422, f"activity_level must be one of {sorted(ACTIVITY)}")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(u, k, v)
    db.merge(u)
    db.commit()
    return profile_out(u)


def profile_out(u: User) -> dict:
    return {"sex": u.sex, "birth_year": u.birth_year, "height_cm": u.height_cm, "weight_kg": u.weight_kg,
            "activity_level": u.activity_level, "goal": u.goal}


@router.get("/profile")
def get_profile(u: User = Depends(current_user)):
    return profile_out(u)


@router.get("/targets")
def get_targets(u: User = Depends(current_user)):
    try:
        return targets(u.weight_kg, u.height_cm, u.birth_year, u.sex, u.activity_level, u.goal)
    except ProfileIncomplete as e:
        raise HTTPException(422, str(e))


@router.get("/foods")
def foods(q: str = "", limit: int = Query(20, le=50)):
    return search(q, limit)


class LogIn(BaseModel):
    day: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    food_key: Optional[str] = None
    name: Optional[str] = Field(default=None, max_length=120)
    grams: float = Field(gt=0, le=5000)
    kcal: Optional[float] = Field(default=None, ge=0, le=10000)
    protein_g: float = Field(default=0, ge=0, le=1000)
    carbs_g: float = Field(default=0, ge=0, le=2000)
    fat_g: float = Field(default=0, ge=0, le=1000)


def log_out(l: FoodLog) -> dict:
    return {"id": l.id, "day": l.day, "name": l.name, "food_key": l.food_key, "grams": l.grams, "kcal": l.kcal,
            "protein_g": l.protein_g, "carbs_g": l.carbs_g, "fat_g": l.fat_g}


@router.post("/log", status_code=201)
def add_log(body: LogIn, u: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.food_key:
        if body.food_key not in FOODS:
            raise HTTPException(422, "unknown food_key; use /api/nutrition/foods or log custom macros")
        m, name = macros_for(body.food_key, body.grams), FOODS[body.food_key]["name"]
    else:
        if not body.name or body.kcal is None:
            raise HTTPException(422, "custom entries need name and kcal")
        m, name = {"kcal": body.kcal, "protein_g": body.protein_g, "carbs_g": body.carbs_g, "fat_g": body.fat_g}, body.name
    row = FoodLog(user_id=u.id, day=body.day or Date.today().isoformat(), name=name, food_key=body.food_key,
                  grams=body.grams, **m)
    db.add(row)
    db.commit()
    return log_out(row)


@router.delete("/log/{lid}", status_code=204)
def del_log(lid: int, u: User = Depends(current_user), db: Session = Depends(get_db)):
    row = db.get(FoodLog, lid)
    if not row or row.user_id != u.id:
        raise HTTPException(404)
    db.delete(row)
    db.commit()


def expenditure_for_day(db: Session, u: User, day: str) -> dict:
    sess = (db.query(WorkoutSession).options(selectinload(WorkoutSession.sets))
            .filter(WorkoutSession.user_id == u.id, WorkoutSession.finished_at.isnot(None)).all())
    lo = mid = hi = 0
    for s in sess:
        if s.started_at.date().isoformat() != day:
            continue
        mins = (s.finished_at - s.started_at).total_seconds() / 60
        n = max(1, len(s.sets))
        inten = classify_intensity(n, sum(x.reps for x in s.sets) / n, max(1.0, mins * 60 / n - 30))
        e = estimate_session_kcal(mins, u.weight_kg, inten)
        lo, mid, hi = lo + e["kcal_low"], mid + e["kcal_mid"], hi + e["kcal_high"]
    return {"kcal_low": lo, "kcal_mid": mid, "kcal_high": hi, "weight_assumed": u.weight_kg is None}


@router.get("/day")
def day_summary(day: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"), u: User = Depends(current_user),
                db: Session = Depends(get_db)):
    day = day or Date.today().isoformat()
    rows = db.query(FoodLog).filter_by(user_id=u.id, day=day).order_by(FoodLog.id).all()
    tot = {k: round(sum(getattr(r, k) for r in rows), 1) for k in ("kcal", "protein_g", "carbs_g", "fat_g")}
    try:
        tg = targets(u.weight_kg, u.height_cm, u.birth_year, u.sex, u.activity_level, u.goal)
    except ProfileIncomplete:
        tg = None
    return {"day": day, "entries": [log_out(r) for r in rows], "totals": tot, "targets": tg,
            "remaining_kcal": round(tg["target_kcal"] - tot["kcal"]) if tg else None,
            "exercise_expenditure_estimate": expenditure_for_day(db, u, day)}
