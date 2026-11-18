"""Exercise registry: load, validate, resolve names, expose support status."""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from motion_engine.primitives import validate_spec
from motion_engine.skeleton import REAL_JOINTS
from .models import ExerciseDefinition

DEFINITIONS_DIR = Path(__file__).parent / "definitions"

# Keyword -> movement pattern, used to triage unsupported requests for developers.
PATTERN_HINTS = {
    "curl": "curl", "raise": "raise", "press": "press", "squat": "squat", "lunge": "lunge",
    "row": "row", "pull": "pull", "push": "push", "deadlift": "hinge", "hinge": "hinge",
    "extension": "extension", "fly": "adduction", "flye": "adduction", "thrust": "hinge",
    "twist": "rotation", "rotation": "rotation", "kick": "extension", "step": "lunge",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def validate_definition(d: ExerciseDefinition) -> List[str]:
    errs = validate_spec(d.rep.signal, "rep.signal")
    if d.rep.rest == d.rep.peak:
        errs.append("rep: rest and peak must differ")
    for r in d.form_rules:
        if r.type in ("stability", "limit", "symmetry"):
            if not r.signal:
                errs.append(f"rule {r.id}: {r.type} needs a signal")
            else:
                errs += validate_spec(r.signal, f"rule {r.id}.signal")
        if r.type == "rom" and r.min is None and r.max is None:
            errs.append(f"rule {r.id}: rom needs min and/or max")
    for p in d.posture:
        errs += validate_spec(p.signal, "posture.signal")
    errs += [f"unknown required joint {j!r}" for j in d.required_joints if j not in REAL_JOINTS]
    if len({r.id for r in d.form_rules}) != len(d.form_rules):
        errs.append("duplicate form rule ids")
    return errs


class ExerciseRegistry:
    def __init__(self, definitions: Iterable[ExerciseDefinition] = ()):
        self._defs: Dict[str, ExerciseDefinition] = {}
        for d in definitions:
            self.register(d)

    @classmethod
    def load(cls, directory: Path = DEFINITIONS_DIR) -> "ExerciseRegistry":
        reg = cls()
        for p in sorted(Path(directory).glob("*.json")):
            reg.register(ExerciseDefinition.model_validate(json.loads(p.read_text())))
        return reg

    def register(self, d: ExerciseDefinition) -> None:
        errs = validate_definition(d)
        if errs:
            raise ValueError(f"invalid exercise {d.id}: " + "; ".join(errs))
        self._defs[d.id] = d

    def get(self, exercise_id: str) -> Optional[ExerciseDefinition]:
        return self._defs.get(exercise_id)

    def all(self, include_draft: bool = False) -> List[ExerciseDefinition]:
        return [d for d in self._defs.values() if include_draft or d.status != "draft"]

    def recognisable(self) -> List[ExerciseDefinition]:
        return [d for d in self._defs.values() if d.status == "supported"]

    def resolve(self, text: str, cutoff: float = 0.82) -> Tuple[Optional[ExerciseDefinition], float]:
        """Map free text to a definition by id/name/alias, then fuzzy match."""
        q = _norm(text)
        if not q:
            return None, 0.0
        names = {}
        for d in self.all():
            for n in [d.id.replace("_", " "), d.name, *d.aliases]:
                names[_norm(n)] = d
        if q in names:
            return names[q], 1.0
        m = difflib.get_close_matches(q, list(names), n=1, cutoff=cutoff)
        if m:
            return names[m[0]], difflib.SequenceMatcher(None, q, m[0]).ratio()
        return None, 0.0

    def suggest_pattern(self, text: str) -> Optional[str]:
        for w in _norm(text).split():
            for k, v in PATTERN_HINTS.items():
                if w.startswith(k):
                    return v
        return None

    def closest(self, text: str, n: int = 3) -> List[str]:
        q = _norm(text)
        pool = {_norm(d.name): d.id for d in self.all()}
        return [pool[m] for m in difflib.get_close_matches(q, list(pool), n=n, cutoff=0.3)]
