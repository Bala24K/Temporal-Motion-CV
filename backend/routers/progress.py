from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
from fastapi import HTTPException

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import SetRecord, User, WorkoutSession, now
from ..deps import current_user, get_db
from ..state import state

router = APIRouter(prefix="/api/progress", tags=["progress"])
DAYS = {"today": 1, "week": 7, "month": 30}


@router.get("")
def progress(window: str = Query("week", alias="range", pattern="^(today|week|month)$"), u: User = Depends(current_user),
             db: Session = Depends(get_db)):
    end = now()
    start = (end.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=DAYS[window] - 1))
    sets = db.query(SetRecord).filter(SetRecord.user_id == u.id, SetRecord.created_at >= start).all()
    sessions = db.query(WorkoutSession).filter(WorkoutSession.user_id == u.id, WorkoutSession.started_at >= start).all()
    finished = [s for s in sessions if s.finished_at]

    by_day: Dict[str, dict] = defaultdict(lambda: {"volume_kg": 0.0, "reps": 0, "scores": []})
    muscles: Dict[str, int] = defaultdict(int)
    best: Dict[str, dict] = {}
    for s in sets:
        day = s.created_at.date().isoformat()
        by_day[day]["volume_kg"] += s.reps * s.weight_kg
        by_day[day]["reps"] += s.reps
        if s.avg_form_score is not None:
            by_day[day]["scores"].append(s.avg_form_score)
        d = state.registry.get(s.exercise_id) if s.exercise_id else None
        for m in (d.target_muscles if d else []):
            muscles[m] += 1
        b = best.setdefault(s.exercise_name, {"max_weight_kg": 0.0, "max_reps": 0})
        b["max_weight_kg"] = max(b["max_weight_kg"], s.weight_kg)
        b["max_reps"] = max(b["max_reps"], s.reps)

    series = []
    for i in range(DAYS[window]):
        day = (start + timedelta(days=i)).date().isoformat()
        x = by_day.get(day)
        series.append({"date": day, "volume_kg": round(x["volume_kg"], 1) if x else 0, "reps": x["reps"] if x else 0,
                       "form": round(sum(x["scores"]) / len(x["scores"]), 1) if x and x["scores"] else None})
    scored = [s.avg_form_score for s in sets if s.avg_form_score is not None]
    roms = [s.avg_rom for s in sets if s.avg_rom is not None]
    days_trained = len({s.started_at.date() for s in sessions})
    return {
        "range": window,
        "workouts": len(sessions), "sets": len(sets), "reps": sum(s.reps for s in sets),
        "volume_kg": round(sum(s.reps * s.weight_kg for s in sets), 1),
        "active_minutes": round(sum((s.finished_at - s.started_at).total_seconds() for s in finished) / 60, 1),
        "avg_form_score": round(sum(scored) / len(scored), 1) if scored else None,
        "avg_rom": round(sum(roms) / len(roms), 3) if roms else None,
        "days_trained": days_trained, "weekly_target": u.weekly_target,
        "consistency": round(min(1.0, days_trained / max(1, u.weekly_target * DAYS[window] / 7)), 2),
        "muscle_sets": dict(sorted(muscles.items(), key=lambda kv: -kv[1])),
        "muscle_share": {m: round(100 * n / max(1, sum(muscles.values())), 1) for m, n in sorted(muscles.items(), key=lambda kv: -kv[1])},
        "personal_bests": best, "series": series,
        "unsupported_sets": sum(1 for s in sets if s.source == "manual" and not s.exercise_id),
    }


def e1rm(weight: float, reps: int) -> Optional[float]:
    """Epley estimate of one-rep max. None for bodyweight/zero-load sets."""
    if weight <= 0 or reps <= 0:
        return None
    return round(weight * (1 + reps / 30.0), 1) if reps > 1 else round(weight, 1)


def trend(values: List[Optional[float]], min_points: int = 4, flat_pct: float = 0.02) -> Dict:
    """Least-squares slope per session as % of the mean; 'flat' inside +-2%/session. Needs >= min_points."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < min_points:
        return {"direction": "insufficient_data", "points": len(pts)}
    x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    slope = float(np.polyfit(x, y, 1)[0])
    rel = slope / (abs(y.mean()) + 1e-9)
    return {"direction": "improving" if rel > flat_pct else "declining" if rel < -flat_pct else "flat",
            "slope_pct_per_session": round(100 * rel, 2), "points": len(pts)}


@router.get("/exercises")
def exercises_with_data(u: User = Depends(current_user), db: Session = Depends(get_db)):
    sets = db.query(SetRecord).filter(SetRecord.user_id == u.id).all()
    agg: Dict[str, dict] = {}
    for s in sets:
        a = agg.setdefault(s.exercise_name, {"exercise_id": s.exercise_id, "name": s.exercise_name, "sets": 0, "best_e1rm": None})
        a["sets"] += 1
        e = e1rm(s.weight_kg, s.reps)
        if e and (a["best_e1rm"] is None or e > a["best_e1rm"]):
            a["best_e1rm"] = e
    return sorted(agg.values(), key=lambda a: -a["sets"])


@router.get("/exercise/{name}")
def exercise_progress(name: str, u: User = Depends(current_user), db: Session = Depends(get_db)):
    q = db.query(SetRecord).filter(SetRecord.user_id == u.id)
    d, _ = state.registry.resolve(name)
    sets = q.filter(SetRecord.exercise_id == d.id).all() if d else q.filter(SetRecord.exercise_name == name).all()
    if not sets:
        raise HTTPException(404, "No sets logged for this exercise")
    by_day: Dict[str, list] = defaultdict(list)
    for s in sorted(sets, key=lambda s: s.created_at):
        by_day[s.created_at.date().isoformat()].append(s)
    series, best, prs = [], 0.0, []
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    for day, ss in by_day.items():
        es = [e for e in (e1rm(s.weight_kg, s.reps) for s in ss) if e]
        top = max(es) if es else None
        if top and top > best:
            best = top
            prs.append({"date": day, "e1rm": top})
        series.append({"date": day, "sets": len(ss), "reps": sum(s.reps for s in ss),
                       "volume_kg": round(sum(s.reps * s.weight_kg for s in ss), 1), "best_e1rm": top,
                       "avg_rom": mean([s.avg_rom for s in ss if s.avg_rom is not None]),
                       "avg_rep_seconds": mean([s.avg_rep_seconds for s in ss if s.avg_rep_seconds is not None]),
                       "avg_form": mean([s.avg_form_score for s in ss if s.avg_form_score is not None])})
    col = lambda k: [x[k] for x in series]
    return {"exercise": d.name if d else name, "ai_analysis": bool(d), "series": series, "personal_records": prs,
            "trends": {"e1rm": trend(col("best_e1rm")), "volume": trend(col("volume_kg")), "rom": trend(col("avg_rom")),
                       "tempo": trend(col("avg_rep_seconds")), "form": trend(col("avg_form"))}}


@router.get("/goal")
def goal_progress(weeks: int = Query(8, ge=2, le=26), u: User = Depends(current_user), db: Session = Depends(get_db)):
    """Weekly consistency, streak, volume trajectory and a goal-specific headline."""
    end = now().date()
    monday = end - timedelta(days=end.weekday())
    start = monday - timedelta(weeks=weeks - 1)
    sessions = db.query(WorkoutSession).filter(WorkoutSession.user_id == u.id, WorkoutSession.started_at >= start).all()
    sets = db.query(SetRecord).filter(SetRecord.user_id == u.id, SetRecord.created_at >= start).all()
    wk = lambda d: (d - start).days // 7
    sess_days = [set() for _ in range(weeks)]
    vol, form = [0.0] * weeks, [[] for _ in range(weeks)]
    for s in sessions:
        i = wk(s.started_at.date())
        if 0 <= i < weeks:
            sess_days[i].add(s.started_at.date())
    for s in sets:
        i = wk(s.created_at.date())
        if 0 <= i < weeks:
            vol[i] += s.reps * s.weight_kg
            if s.avg_form_score is not None:
                form[i].append(s.avg_form_score)
    counts = [len(x) for x in sess_days]
    met = [c >= u.weekly_target for c in counts]
    order = list(reversed(met))
    if not met[-1]:                 # the in-progress week must not break an existing streak
        order = order[1:]
    streak = 0
    for m in order:
        if not m:
            break
        streak += 1
    half = weeks // 2
    prev, cur = sum(vol[:half]), sum(vol[half:])
    change = round(100 * (cur - prev) / prev, 1) if prev > 0 else None
    fs = [round(sum(f) / len(f), 1) if f else None for f in form]
    headline = {"build_muscle": f"Volume {'up' if (change or 0) >= 0 else 'down'} {abs(change)}% vs the previous period" if change is not None else "Log a few weeks of sets to see your volume trend",
                "consistency": f"{streak}-week streak of hitting {u.weekly_target} sessions",
                "movement_quality": "Form consistency is " + trend(fs)["direction"].replace("_", " "),
                "get_stronger": "See per-exercise e1RM trends", "lose_weight": f"{streak}-week consistency streak",
                "improve_fitness": f"{sum(counts)} training day{'' if sum(counts) == 1 else 's'} in {weeks} weeks"}.get(u.goal or "", f"{sum(counts)} training day{'' if sum(counts) == 1 else 's'} in {weeks} weeks")
    return {"goal": u.goal, "weekly_target": u.weekly_target, "weeks": [
        {"week_start": (start + timedelta(weeks=i)).isoformat(), "sessions": counts[i], "target_met": met[i],
         "volume_kg": round(vol[i], 1), "avg_form": fs[i]} for i in range(weeks)],
        "streak_weeks": streak, "volume_change_pct": change, "form_trend": trend(fs), "headline": headline}
