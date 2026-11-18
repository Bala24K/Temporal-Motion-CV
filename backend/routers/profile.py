from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from motion_engine.contract import frames_error
from motion_engine.calibration import (CalibrationError, GUIDED_STAGES, INSTRUCTIONS, OPTIONAL_STAGES, STAGES, assess_window,
                                       calibrate as run_calibration, check_stage_frame, missing_stages)
from motion_engine.quality import check_frame
from motion_engine.skeleton import BodyProfile, measure_body

from ..config import settings
from ..db import BaselineRow, BodyProfileRow, User
from ..deps import current_user, get_db

router = APIRouter(prefix="/api/profile", tags=["profile"])
CALIB_JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


class CalibrationBody(BaseModel):
    height_cm: Optional[float] = Field(default=None, ge=100, le=250)
    frames: List[Dict[str, List[float]]] = Field(max_length=600)   # landmark -> [x, y, visibility]


@router.get("")
def get_profile(u: User = Depends(current_user), db: Session = Depends(get_db)):
    row = db.get(BodyProfileRow, u.id)
    p = BodyProfile.from_dict(row.data if row else None)
    bases = db.query(BaselineRow).filter_by(user_id=u.id).all()
    return {"calibrated": p.calibrated, "profile": p.to_dict(), "ratios": p.ratios() if p.torso_len else None,
            "segments_cm": segments_cm(p), "stages": p.stages, "instructions": INSTRUCTIONS,
            "calibration": calibration_summary(p) if p.torso_len else None,
            "missing_stages": missing_stages(p.stages) if p.torso_len else list(GUIDED_STAGES),
            "baselines": {b.exercise_id: b.data for b in bases}}


def calibration_summary(p: BodyProfile) -> dict:
    """Quality broken into its parts, so the UI can say *why* a calibration is weak."""
    r = lambda x: round(float(x), 2)
    return {"status": p.status or ("good" if p.quality >= 0.8 else "usable" if p.quality >= 0.6 else "retry"),
            "quality": r(p.quality), "stability": r(p.stability), "confidence": r(p.confidence), "coverage": r(p.coverage),
            "plausibility": r(p.plausibility), "method": p.method, "ratio_agreement_3d": p.ratio_agreement_3d,
            "warnings": list(p.warnings), "setup": p.setup or None}


def segments_cm(p: BodyProfile):
    """Approximate segment lengths in cm (needs height at calibration). None if unavailable."""
    if not p.cm_per_torso or not p.torso_len:
        return None
    k = p.cm_per_torso / p.torso_len
    return {n: round(getattr(p, n) * k, 1) for n in ("shoulder_width", "hip_width", "upper_arm", "forearm", "thigh", "shin")} | {
        "torso": round(p.cm_per_torso, 1)}


def _require_contract(frames):
    err = frames_error(frames)
    if err:
        raise HTTPException(422, err)


class StageCheck(BaseModel):
    stage: str
    frame: Dict[str, List[float]]


@router.post("/calibrate/stage-check")
def stage_check(body: StageCheck):
    """Per-frame coaching for the calibration screen ('turn sideways', 'raise your arms')."""
    if body.stage not in STAGES:
        raise HTTPException(422, f"stage must be one of {STAGES}")
    _require_contract([body.frame])
    return {**check_stage_frame(body.stage, body.frame), "instruction": INSTRUCTIONS[body.stage]}


class AssessBody(BaseModel):
    stage: str
    frames: List[Dict[str, List[float]]] = Field(min_length=1, max_length=120)


@router.post("/calibrate/assess")
def assess(body: AssessBody):
    """Judge a contiguous window (~1.5 s) of one calibration stage: framing, distance, facing, pose, stillness and
    tracking confidence, each with a specific message. One request per window instead of one per frame."""
    if body.stage not in STAGES:
        raise HTTPException(422, f"stage must be one of {STAGES}")
    _require_contract(body.frames)
    return {**assess_window(body.stage, body.frames), "stage": body.stage, "instruction": INSTRUCTIONS[body.stage]}


@router.get("/calibrate/protocol")
def protocol():
    """Stages the wizard should run, in order, with instructions (single source of truth for the client)."""
    return {"stages": [{"id": s, "instruction": INSTRUCTIONS[s], "optional": s in OPTIONAL_STAGES} for s in GUIDED_STAGES]}


class StagedBody(BaseModel):
    height_cm: Optional[float] = Field(default=None, ge=100, le=250)
    stages: Dict[str, List[Dict[str, List[float]]]]
    camera_label: Optional[str] = Field(default=None, max_length=120)


@router.post("/calibrate/staged")
def calibrate_staged(body: StagedBody, u: User = Depends(current_user), db: Session = Depends(get_db)):
    for k, v in body.stages.items():
        if k not in STAGES:
            raise HTTPException(422, f"unknown stage {k!r}")
        if len(v) > 300:
            raise HTTPException(413, "Too many frames in one stage")
        _require_contract(v)
    try:
        p, results = run_calibration(body.stages, body.height_cm, camera_label=body.camera_label)
    except CalibrationError as e:
        raise HTTPException(422, str(e))
    row = db.get(BodyProfileRow, u.id)
    if row:
        row.data = p.to_dict()
    else:
        db.add(BodyProfileRow(user_id=u.id, data=p.to_dict()))
    db.commit()
    # report the state the engine will actually use (it used to say True even when the stored profile was unusable)
    return {"calibrated": p.calibrated, **calibration_summary(p),
            "stages": [r.__dict__ for r in results], "segments_cm": segments_cm(p), "ratios": p.ratios(),
            "missing_stages": missing_stages(p.stages), "optional_stages": list(OPTIONAL_STAGES),
            "note": "Approximate body proportions for normalisation - not an anatomical reconstruction."}


@router.post("/calibrate/check")
def calibrate_check(body: Dict[str, List[float]]):
    """Live framing feedback for the calibration screen (one frame)."""
    _require_contract([body])
    return check_frame(body, CALIB_JOINTS)


@router.post("/calibrate")
def calibrate(body: CalibrationBody, u: User = Depends(current_user), db: Session = Depends(get_db)):
    _require_contract(body.frames)
    good = [f for f in body.frames if check_frame(f, CALIB_JOINTS)["ok"]]
    if len(good) < settings.min_calibration_frames:
        raise HTTPException(422, f"Need at least {settings.min_calibration_frames} clear full-body frames; "
                                 f"got {len(good)}. Stand where your whole body is visible.")
    p = measure_body(good, body.height_cm)
    if not p.calibrated:
        raise HTTPException(422, "Measurement was too unstable - hold still and try again.")
    row = db.get(BodyProfileRow, u.id)
    if row:
        row.data = p.to_dict()
    else:
        db.add(BodyProfileRow(user_id=u.id, data=p.to_dict()))
    db.commit()
    return {"calibrated": p.calibrated, "quality": round(p.quality, 2), "ratios": p.ratios(), "frames_used": len(good),
            "note": "Approximate body proportions used to normalise movement - not an anatomical reconstruction."}
