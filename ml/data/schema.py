"""Dataset schemas. Label provenance is a first-class field: the promotion gate only trusts human/expert labels."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

LabelSource = Literal["expert", "human_verified", "mixed_verified", "rule", "model", "user"]
TRUSTED = {"expert", "human_verified", "mixed_verified"}
View = Literal["front", "side", "45deg", "unknown"]
FORM_ISSUES = ["shallow_depth", "too_fast", "elbow_drift", "torso_lean", "knee_cave", "asymmetry",
               "incomplete_lockout", "excess_momentum", "other"]


class ClipMeta(BaseModel):
    clip_id: str
    subject_id: str                                  # the split unit - never share across train/test
    exercise_id: Optional[str] = None                # None => unlabeled / unknown exercise
    view: View = "unknown"
    fps: float = 30.0
    n_frames: int = 0
    source: str = "internal"                         # internal | consented_user | public:<dataset>
    consent: bool = False                            # required for anything originating from users
    pose_model: str = ""                             # estimator name+version that produced the landmarks
    quality: float = 1.0                             # fraction of frames with a usable pose
    label_source: LabelSource = "model"
    sha256: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RepAnnotation(BaseModel):
    start_frame: int
    end_frame: int
    peak_frame: Optional[int] = None
    quality: Literal["good", "partial", "bad"] = "good"
    issues: List[str] = []


class ClipAnnotation(BaseModel):
    clip_id: str
    exercise_id: Optional[str] = None
    view: View = "unknown"
    reps: List[RepAnnotation] = []
    notes: str = ""
    annotator: str = ""
    label_source: LabelSource = "human_verified"
    camera_ok: bool = True
