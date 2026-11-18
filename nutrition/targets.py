"""Calorie / macro targets from standard formulas (Mifflin-St Jeor BMR x activity factor). Ranges, with a
safety floor, and no medical claims. Everything here is general guidance, not advice for a medical condition."""
from __future__ import annotations

from datetime import date
from typing import Dict, Optional

ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
SEX_CONST = {"male": 5.0, "female": -161.0, "unspecified": -78.0}      # unspecified = midpoint
GOAL_ADJ = {"lose_weight": 0.85, "build_muscle": 1.10, "get_stronger": 1.05, "improve_fitness": 1.0,
            "consistency": 1.0, "movement_quality": 1.0, None: 1.0}
PROTEIN_G_PER_KG = {"build_muscle": 1.8, "get_stronger": 1.8, "lose_weight": 1.8}
FLOOR_KCAL = {"male": 1500, "female": 1200, "unspecified": 1350}


class ProfileIncomplete(ValueError):
    pass


def bmr(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + SEX_CONST[sex]


def targets(weight_kg: Optional[float], height_cm: Optional[float], birth_year: Optional[int], sex: Optional[str],
            activity: Optional[str], goal: Optional[str], today: Optional[date] = None) -> Dict:
    missing = [n for n, v in (("weight_kg", weight_kg), ("height_cm", height_cm), ("birth_year", birth_year), ("activity_level", activity)) if not v]
    if missing:
        raise ProfileIncomplete("missing: " + ", ".join(missing))
    sex = sex if sex in SEX_CONST else "unspecified"
    age = (today or date.today()).year - birth_year
    if not (14 <= age <= 100):
        raise ProfileIncomplete("targets are only provided for ages 14-100")
    b = bmr(weight_kg, height_cm, age, sex)
    tdee = b * ACTIVITY[activity]
    kcal = max(tdee * GOAL_ADJ.get(goal, 1.0), FLOOR_KCAL[sex])
    protein = PROTEIN_G_PER_KG.get(goal, 1.4) * weight_kg
    fat = 0.25 * kcal / 9
    carbs = max(0.0, (kcal - protein * 4 - fat * 9) / 4)
    return {"bmr": round(b), "maintenance_kcal": round(tdee), "target_kcal": round(kcal),
            "kcal_range": [round(kcal * 0.95), round(kcal * 1.05)],
            "protein_g": round(protein), "fat_g": round(fat), "carbs_g": round(carbs),
            "floor_applied": kcal == FLOOR_KCAL[sex] and tdee * GOAL_ADJ.get(goal, 1.0) < FLOOR_KCAL[sex],
            "note": "General estimates (Mifflin-St Jeor). Not medical advice; talk to a professional for health conditions."}
