"""Exercise energy expenditure *estimates*. Computer vision cannot measure calories, so this returns a range
from MET values (Compendium of Physical Activities, resistance training: light/moderate ~3.5, vigorous ~6.0)
x body mass x time. Never presented as a single precise number."""
from __future__ import annotations

from typing import Dict, Optional

MET = {"light": 3.5, "moderate": 4.5, "vigorous": 6.0}
SPREAD = (0.8, 1.25)            # individual variation: +-20-25%
DEFAULT_WEIGHT_KG = 70.0


def classify_intensity(sets: int, reps_per_set: float, rest_seconds: float, form_weight_ratio: Optional[float] = None) -> str:
    """Crude: short rests + high volume => vigorous. Documented heuristic, easy to replace."""
    density = (sets * reps_per_set) / max(1.0, rest_seconds)
    return "vigorous" if density > 1.5 else "moderate" if density > 0.6 else "light"


def estimate_session_kcal(duration_minutes: float, weight_kg: Optional[float], intensity: str = "moderate") -> Dict:
    if duration_minutes <= 0:
        return {"kcal_low": 0, "kcal_mid": 0, "kcal_high": 0, "assumptions": {}}
    w, met = weight_kg or DEFAULT_WEIGHT_KG, MET[intensity]
    mid = met * w * duration_minutes / 60.0
    return {"kcal_low": round(mid * SPREAD[0]), "kcal_mid": round(mid), "kcal_high": round(mid * SPREAD[1]),
            "assumptions": {"met": met, "intensity": intensity, "weight_kg": w, "weight_assumed": weight_kg is None,
                            "duration_minutes": round(duration_minutes, 1)},
            "note": "Estimate only - based on time and body mass, not measured. Real expenditure varies by ~25%."}
