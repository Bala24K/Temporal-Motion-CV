"""ExerciseDefinition: exercises as data + composition of movement primitives."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RepDefinition(BaseModel):
    signal: dict                         # primitive spec driving rep counting
    rest: float                          # population default: signal value at the start/end of a rep
    peak: float                          # population default: signal value at the turnaround
    toward_peak_phase: Literal["concentric", "eccentric"] = "concentric"
    count_progress: float = 0.55         # min fraction of rest->peak to count a rep
    partial_progress: float = 0.30       # above this (but < count) => flagged partial rep
    min_rep_seconds: float = 0.3
    max_rep_seconds: float = 20.0        # an open rep older than this is abandoned (and reported)
    # --- counter v2 (time-based, so behaviour does not depend on frame rate) ---
    start_progress: float = 0.25         # progress that starts a rep ...
    start_dwell: float = 0.10            # ... held this long (s)
    complete_progress: float = 0.35      # relaxed return: a rep completes below this ... (was 0.15 = near-full return)
    complete_dwell: float = 0.15         # ... held this long (s)
    reversal: float = 0.25               # or: dropped this far from its peak and started the next rep
    settle_drop: float = 0.45            # or: dropped this far from its peak and stayed still ...
    settle_seconds: float = 0.4          # ... this long (user pauses part-way down between reps)
    smoothing_tau: float = 0.08          # EMA time constant (s)
    # Bounds a personal range capture may move rest/peak to. Default: +-25% of |peak - rest| around each.
    population_range: Optional[Dict[str, List[float]]] = None


class FormRule(BaseModel):
    id: str
    type: Literal["rom", "tempo", "stability", "limit", "symmetry"]
    severity: Literal["info", "warn", "major"] = "warn"
    signal: Optional[dict] = None        # stability / limit / symmetry
    min: Optional[float] = None          # rom: fraction of expected range; limit: value
    max: Optional[float] = None          # rom/limit/stability(range)/symmetry(mean |L-R|)
    min_rep_seconds: Optional[float] = None
    min_phase_seconds: Optional[float] = None   # min duration of each half of the rep
    good: str = ""
    bad: str = ""
    views: Optional[List[Literal["front", "side", "45deg"]]] = None   # evaluate only in these camera views
    min_evaluable: float = 0.8           # share of rep frames with a finite signal required to judge the rule


class Posture(BaseModel):
    """Soft recognition constraint (e.g. 'elbows stay near the torso')."""
    signal: dict
    stat: Literal["median", "min", "max"] = "median"
    min: Optional[float] = None
    max: Optional[float] = None
    scale: float = 30.0


class CameraSpec(BaseModel):
    supported_views: List[Literal["front", "side", "45deg"]] = ["front"]
    preferred: str = "front"
    full_body: bool = False
    notes: str = ""


class ExerciseDefinition(BaseModel):
    id: str
    name: str
    version: int = 1
    status: Literal["supported", "beta", "draft"] = "supported"
    aliases: List[str] = Field(default_factory=list)
    family: str
    movement_patterns: List[str]
    target_muscles: List[str]
    secondary_muscles: List[str] = Field(default_factory=list)
    primary_joints: List[str]
    equipment: List[str] = Field(default_factory=list)
    body_position: Literal["standing", "seated", "lying", "plank", "any"] = "standing"
    difficulty: Literal["beginner", "intermediate", "advanced"] = "beginner"
    camera: CameraSpec = CameraSpec()
    required_joints: List[str]
    rep: RepDefinition
    form_rules: List[FormRule] = Field(default_factory=list)
    posture: List[Posture] = Field(default_factory=list)
    motion_plane: Literal["frontal", "sagittal", "mixed"] = "mixed"   # frontal => yaw compensation applies
    # visible_side: side-on views hide the far limbs, so quality gating and side:"visible" signals use the
    # better-visible side (both sides when both are clearly visible). bilateral: both sides always required.
    laterality: Literal["bilateral", "visible_side"] = "bilateral"
    metric: Literal["reps", "duration"] = "reps"
    confidence_threshold: float = 0.6

    def equivalence_key(self) -> str:
        """Definitions with the same key cannot be told apart from pose alone."""
        import json
        r = self.rep
        return json.dumps([r.signal, r.rest, r.peak, [p.model_dump() for p in self.posture]], sort_keys=True)
