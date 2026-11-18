"""Admin / ML console: unsupported-exercise demand, exercise definitions, model registry."""
from typing import Any, Dict, List, Optional, Tuple

import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from exercise_system.models import ExerciseDefinition
from ml.data.extract import ingest_video
from ml.data.labelstudio import from_export
from ml.data.report import report as dataset_report
from ml.data.schema import ClipMeta
from ml.data.store import DatasetStore
from exercise_system.registry import validate_definition
from motion_engine.contract import frames_error
from motion_engine.embedding import build_prototypes, match_families
from motion_engine.synthetic import EX as SYNTH_EX

from ..db import (CustomExercise, ModelVersion, RepRecord, SetRecord, UnsupportedRequest, UnsupportedRequestUser,
                  User, WorkoutSession, now)
from ..deps import admin_user, get_db
from ..state import state

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(admin_user)])

REQUEST_STATUSES = {"new", "triaged", "planned", "collecting", "shipped", "declined"}
REQUIRED_METRICS = ("macro_f1", "subject_disjoint_accuracy", "ece", "latency_ms_p95", "ood_auroc",
                    "conformal_coverage", "id_ood_false_flag_rate")
REQUIRED_LINEAGE = ("code_commit", "training_config", "pose_model_version", "feature_version", "evaluation_dataset")
MIN_OOD_AUROC, MIN_CONFORMAL_COVERAGE, MAX_ID_FALSE_FLAG = 0.90, 0.85, 0.10
TRUSTED_LABEL_SOURCES = {"expert", "human_verified", "mixed_verified"}


def promotion_gate(candidate: ModelVersion, current: Optional[ModelVersion]) -> Tuple[bool, List[str]]:
    """Never auto-deploy on 'accuracy went up'. All of these must hold."""
    m, why = candidate.metrics or {}, []
    for k in REQUIRED_METRICS:
        if m.get(k) is None:
            why.append(f"missing required metric: {k}")
    if m.get("label_source") not in TRUSTED_LABEL_SOURCES:
        why.append("labels are not human-verified (label_source=%r): scores measure agreement with a rule, "
                   "not real-world correctness" % m.get("label_source"))
    if not candidate.dataset_version:
        why.append("dataset_version is required")
    for k in REQUIRED_LINEAGE:
        if not (candidate.lineage or {}).get(k):
            why.append(f"missing lineage: {k} (a production model must be reproducible)")
    if m.get("ood_auroc") is not None and m["ood_auroc"] < MIN_OOD_AUROC:
        why.append(f"OOD detection too weak (AUROC {m['ood_auroc']} < {MIN_OOD_AUROC})")
    if m.get("conformal_coverage") is not None and m["conformal_coverage"] < MIN_CONFORMAL_COVERAGE:
        why.append(f"conformal coverage {m['conformal_coverage']} < {MIN_CONFORMAL_COVERAGE}")
    if m.get("id_ood_false_flag_rate") is not None and m["id_ood_false_flag_rate"] > MAX_ID_FALSE_FLAG:
        why.append(f"too many in-distribution inputs rejected ({m['id_ood_false_flag_rate']} > {MAX_ID_FALSE_FLAG})")
    if not candidate.limitations:
        why.append("known limitations must be documented")
    if current and not why:
        c = current.metrics or {}
        if m["macro_f1"] < c.get("macro_f1", 0):
            why.append(f"macro_f1 regressed ({m['macro_f1']} < {c['macro_f1']})")
        if m["subject_disjoint_accuracy"] < c.get("subject_disjoint_accuracy", 0):
            why.append("subject-disjoint accuracy regressed")
        if m["ece"] > c.get("ece", 1) + 0.01:
            why.append("calibration (ECE) got worse")
        if c.get("latency_ms_p95") and m["latency_ms_p95"] > 1.25 * c["latency_ms_p95"]:
            why.append("p95 latency increased by more than 25%")
    return (not why), why


def model_out(v: ModelVersion) -> dict:
    return {"id": v.id, "kind": v.kind, "name": v.name, "version": v.version, "dataset_version": v.dataset_version,
            "status": v.status, "metrics": v.metrics, "supported_exercises": v.supported_exercises,
            "limitations": v.limitations, "artifact_uri": v.artifact_uri, "lineage": v.lineage or {},
            "promotion_record": v.promotion_record or {},
            "promoted_at": v.promoted_at.isoformat() if v.promoted_at else None}


@router.get("/overview")
def overview(db: Session = Depends(get_db)):
    reps = db.query(func.count(RepRecord.id)).scalar()
    return {"users": db.query(func.count(User.id)).scalar(), "sessions": db.query(func.count(WorkoutSession.id)).scalar(),
            "sets": db.query(func.count(SetRecord.id)).scalar(), "reps": reps,
            "ai_sets": db.query(func.count(SetRecord.id)).filter(SetRecord.source == "ai").scalar(),
            "manual_unsupported_sets": db.query(func.count(SetRecord.id)).filter(
                SetRecord.source == "manual", SetRecord.exercise_id.is_(None)).scalar(),
            "supported_exercises": len(state.registry.recognisable()),
            "open_requests": db.query(func.count(UnsupportedRequest.id)).filter(
                UnsupportedRequest.status.in_(["new", "triaged", "planned", "collecting"])).scalar(),
            "metrics": state.metrics.snapshot()}


@router.get("/requests")
def requests(status: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(UnsupportedRequest)
    if status:
        q = q.filter_by(status=status)
    rows = q.order_by(UnsupportedRequest.request_count.desc()).limit(200).all()
    return [{"id": r.id, "name": r.display_name, "requests": r.request_count, "distinct_users": len(r.users),
             "candidate_pattern": r.candidate_pattern, "closest_supported": r.closest, "status": r.status,
             "last_seen": r.last_seen.isoformat()} for r in rows]


class StatusBody(BaseModel):
    status: str


@router.patch("/requests/{rid}")
def set_request_status(rid: int, body: StatusBody, db: Session = Depends(get_db)):
    if body.status not in REQUEST_STATUSES:
        raise HTTPException(422, f"status must be one of {sorted(REQUEST_STATUSES)}")
    r = db.get(UnsupportedRequest, rid)
    if not r:
        raise HTTPException(404)
    r.status = body.status
    db.commit()
    return {"id": r.id, "status": r.status}


class DefinitionBody(BaseModel):
    definition: Dict[str, Any]


def _check(defn: Dict[str, Any]) -> Tuple[Optional[ExerciseDefinition], List[str]]:
    try:
        d = ExerciseDefinition.model_validate(defn)
    except Exception as e:
        return None, [str(e)]
    return d, validate_definition(d)


@router.post("/exercises/validate")
def validate_exercise(body: DefinitionBody):
    d, errs = _check(body.definition)
    clash = []
    if d:
        twins = [x.id for x in state.registry.all(include_draft=True)
                 if x.id != d.id and x.equivalence_key() == d.equivalence_key()]
        if twins:
            clash = [f"indistinguishable from pose alone vs: {', '.join(twins)} (recognition will ask the user to confirm)"]
    return {"valid": not errs, "errors": errs, "notes": clash}


@router.post("/exercises", status_code=201)
def add_exercise(body: DefinitionBody, db: Session = Depends(get_db)):
    """Adding an exercise is configuration: validate, store, hot-register. No deploy."""
    d, errs = _check(body.definition)
    if errs:
        raise HTTPException(422, {"errors": errs})
    if state.registry.get(d.id) and not db.get(CustomExercise, d.id):
        raise HTTPException(409, "Built-in exercise ids are immutable; use a new id/version")
    state.registry.register(d)
    row = db.get(CustomExercise, d.id)
    if row:
        row.data = d.model_dump()
    else:
        db.add(CustomExercise(id=d.id, data=d.model_dump()))
    db.commit()
    return {"id": d.id, "status": d.status}


class ModelIn(BaseModel):
    kind: str = Field(pattern="^(pose|movement|form|recognizer)$")
    name: str
    version: int = Field(ge=1)
    dataset_version: str = ""
    metrics: Dict[str, Any] = {}
    supported_exercises: List[str] = []
    limitations: str = ""
    artifact_uri: str = ""
    lineage: Dict[str, Any] = {}


@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    return [model_out(v) for v in db.query(ModelVersion).order_by(ModelVersion.kind, ModelVersion.name,
                                                                   ModelVersion.version.desc())]


@router.post("/models", status_code=201)
def register_model(body: ModelIn, db: Session = Depends(get_db)):
    if db.query(ModelVersion).filter_by(kind=body.kind, name=body.name, version=body.version).first():
        raise HTTPException(409, "That version already exists; versions are immutable")
    v = ModelVersion(**body.model_dump(), status="candidate")
    db.add(v)
    db.commit()
    return model_out(v)


class PromoteBody(BaseModel):
    acknowledge_risks: bool = False


@router.post("/models/{mid}/promote")
def promote(mid: int, body: PromoteBody, db: Session = Depends(get_db), admin: User = Depends(admin_user)):
    v = db.get(ModelVersion, mid)
    if not v:
        raise HTTPException(404)
    cur = db.query(ModelVersion).filter_by(kind=v.kind, name=v.name, status="production").first()
    ok, reasons = promotion_gate(v, cur)
    if not ok:
        raise HTTPException(409, {"promoted": False, "blocked_by": reasons})
    if cur:
        cur.status = "retired"
    v.status, v.promoted_at = "production", now()
    v.promotion_record = {"promoted_at": v.promoted_at.isoformat(), "promoted_by": admin.id,
                          "replaced_version": cur.version if cur else None,
                          "gate": {"passed": True, "required_metrics": list(REQUIRED_METRICS),
                                   "required_lineage": list(REQUIRED_LINEAGE)},
                          "metrics_at_promotion": dict(v.metrics or {}),
                          "previous_metrics": dict(cur.metrics or {}) if cur else None}
    db.commit()
    return {"promoted": True, "model": model_out(v), "replaced": cur.version if cur else None}


@router.post("/models/{mid}/retire")
def retire(mid: int, db: Session = Depends(get_db)):
    v = db.get(ModelVersion, mid)
    if not v:
        raise HTTPException(404)
    v.status = "retired"
    db.commit()
    return model_out(v)


class ClipBody(BaseModel):
    frames: List[Dict[str, List[float]]] = Field(max_length=1500)


_protos = {}


@router.post("/requests/{rid}/clip")
def triage_clip(rid: int, body: ClipBody, db: Session = Depends(get_db)):
    """Few-shot triage: which known movement families does this clip resemble?
    Answers 'can existing primitives describe it, or is this research work?' (hint, not recognition)."""
    if not db.get(UnsupportedRequest, rid):
        raise HTTPException(404)
    if len(body.frames) < 30:
        raise HTTPException(422, "Need at least 30 frames (about one second)")
    err = frames_error(body.frames)
    if err:
        raise HTTPException(422, err)
    if not _protos:
        _protos.update(build_prototypes(sorted(SYNTH_EX)))
    out = match_families(body.frames, state.registry, _protos)
    out["caveat"] = "Prototypes come from the synthetic generator until real labelled clips exist."
    return out


@router.get("/models/production")
def production_models(db: Session = Depends(get_db)):
    return [model_out(v) for v in db.query(ModelVersion).filter_by(status="production")]


@router.get("/models/{mid}/why")
def why_model(mid: int, db: Session = Depends(get_db)):
    """Answer 'why is this model running?': lineage, dataset, metrics, who promoted it, what it replaced, and
    what the gate would say about it right now."""
    v = db.get(ModelVersion, mid)
    if not v:
        raise HTTPException(404)
    cur = db.query(ModelVersion).filter(ModelVersion.kind == v.kind, ModelVersion.name == v.name,
                                        ModelVersion.status == "production", ModelVersion.id != v.id).first()
    ok, reasons = promotion_gate(v, cur)
    return {"model": model_out(v), "would_pass_gate_now": ok, "blocked_by": reasons,
            "reproducible": all((v.lineage or {}).get(k) for k in REQUIRED_LINEAGE)}


class FromRun(BaseModel):
    kind: str = Field(pattern="^(pose|movement|form|recognizer)$")
    name: str
    version: int = Field(ge=1)
    run: Dict[str, Any]                       # a ml.tracking.Run record
    sidecar: Dict[str, Any] = {}              # exported model .json (classes, dataset_version, ...)
    limitations: str = ""
    pose_model_version: str = ""
    evaluation_dataset: str = ""
    hardware: str = ""


@router.post("/models/from-run", status_code=201)
def register_from_run(body: FromRun, db: Session = Depends(get_db)):
    """Register a candidate straight from a training run so lineage is copied, not retyped."""
    r = body.run
    if db.query(ModelVersion).filter_by(kind=body.kind, name=body.name, version=body.version).first():
        raise HTTPException(409, "That version already exists; versions are immutable")
    v = ModelVersion(kind=body.kind, name=body.name, version=body.version, status="candidate",
                     dataset_version=r.get("dataset_version", ""), metrics=r.get("metrics", {}),
                     supported_exercises=body.sidecar.get("classes", []), limitations=body.limitations,
                     artifact_uri=(r.get("artifacts") or {}).get("onnx", ""),
                     lineage={"code_commit": r.get("code_commit"), "training_config": r.get("params"),
                              "feature_version": body.sidecar.get("feature_version"), "run_id": r.get("id"),
                              "pose_model_version": body.pose_model_version, "evaluation_dataset": body.evaluation_dataset,
                              "hardware": body.hardware, "train_seconds": r.get("seconds")})
    db.add(v)
    db.commit()
    return model_out(v)


MAX_UPLOAD = 100 * 1024 * 1024
VIEWS = {"front", "side", "45deg", "unknown"}


def _ingest_job(path: str, meta: ClipMeta) -> dict:
    with state.pose_factory() as est:
        return ingest_video(DatasetStore(state.dataset_dir), path, meta, est).model_dump()


@router.post("/dataset/ingest", status_code=202)
async def dataset_ingest(file: UploadFile = File(...), subject_id: str = Form(..., max_length=80),
                         exercise_id: Optional[str] = Form(None), view: str = Form("unknown"),
                         consent: bool = Form(False), source: str = Form("internal"), admin: User = Depends(admin_user)):
    """Upload a video; pose is extracted in a background job and the raw video is deleted (landmarks only are kept)."""
    if view not in VIEWS:
        raise HTTPException(422, f"view must be one of {sorted(VIEWS)}")
    if exercise_id and not state.registry.get(exercise_id):
        raise HTTPException(422, "unknown exercise_id (leave empty for unlabeled clips)")
    if source.startswith("consented") and not consent:
        raise HTTPException(422, "consent is required for clips originating from users")
    fd, path = tempfile.mkstemp(suffix=".mp4")
    size = 0
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close(); os.remove(path)
                raise HTTPException(413, "video too large (100 MB max)")
            out.write(chunk)
    meta = ClipMeta(clip_id=uuid.uuid4().hex[:12], subject_id=subject_id, exercise_id=exercise_id or None, view=view,
                    consent=consent, source=source, label_source="user" if exercise_id else "model")
    jid = state.jobs.submit(_ingest_job, path, meta, owner=admin.id, kind="ingest")
    return {"job_id": jid, "clip_id": meta.clip_id}


@router.get("/jobs/{jid}")
def job_status(jid: str):
    j = state.jobs.get(jid)
    if not j:
        raise HTTPException(404)
    return j


@router.get("/dataset/report")
def dataset_report_endpoint():
    st = DatasetStore(state.dataset_dir)
    return {**dataset_report(st.effective()), "version": st.version_id()}


@router.post("/dataset/annotations")
def import_annotations(export: List[Dict[str, Any]]):
    """Import a Label Studio JSON export. Annotated clips become human-verified (new dataset version)."""
    st = DatasetStore(state.dataset_dir)
    anns, known, skipped = from_export(export), st.meta_index(), 0
    for a in anns:
        if a.clip_id in known:
            st.annotate(a)
        else:
            skipped += 1
    return {"imported": len(anns) - skipped, "skipped_unknown_clips": skipped, "dataset_version": st.version_id()}
