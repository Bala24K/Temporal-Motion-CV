"""
FastAPI Inference Service — Motion ST-GCN Pipeline.

Exposes REST endpoints for exercise classification and full pipeline
analysis using the learned spatial-temporal graph network. The API never
opens a camera or requires a display.

Endpoints:
    GET  /health   — service/model availability
    GET  /model    — loaded model metadata (architecture, classes, streams, calibration)
    POST /predict  — exercise classification from pose sequence
    POST /analyze  — full hybrid analysis (learned + kinematic branches)
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Ensure Ver-1 is importable (it contains pose/, exercises/, biomechanics/, body_model/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_motion_dir = os.path.dirname(_script_dir)
_workspace_dir = os.path.dirname(_motion_dir)
_ver1_dir = os.path.join(_workspace_dir, "Ver-1")
for p in [_ver1_dir, _motion_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from api.config import load_config
from api.dependencies import get_inference_service, set_inference_service
from api.inference import InferenceService
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


# ── Logging ──────────────────────────────────────────────────────────

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ── Application Lifespan ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load ST-GCN model. Shutdown: release resources."""
    cfg = load_config()
    _setup_logging(cfg.log_level)
    logger = logging.getLogger("api.main")

    service = InferenceService(model_path=cfg.model_path, device=cfg.device)
    try:
        service.load_model()
    except Exception as exc:
        logger.error(
            "Model load failed — service will start but inference endpoints will be unavailable: %s",
            exc,
        )

    set_inference_service(service)
    logger.info("FastAPI inference service ready on %s:%d", cfg.api_host, cfg.api_port)

    yield  # application runs here

    logger.info("Shutting down inference service.")


# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Motion ST-GCN Inference API",
    description=(
        "REST API for exercise classification and biomechanical analysis, "
        "powered by a spatial-temporal graph convolutional network (ST-GCN) "
        "with temporal transformer and calibrated confidence. "
        "Accepts structured pose-landmark sequences — no camera access required."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Global exception handler ────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.getLogger("api.main").error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health(service: InferenceService = Depends(get_inference_service)):
    """Check service and model availability."""
    info = service.get_model_info() if service.is_loaded else {}
    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_name=info.get("model_name"),
        model_path=info.get("model_path"),
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["status"])
async def model_info(service: InferenceService = Depends(get_inference_service)):
    """Return metadata about the loaded ST-GCN model."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(**service.get_model_info())


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["inference"],
)
async def predict(
    request: PredictRequest,
    service: InferenceService = Depends(get_inference_service),
):
    """Classify an exercise from a pose-landmark sequence.

    Accepts a temporal window of pose frames and returns the predicted
    exercise class, calibrated confidence, per-class probabilities, and
    abstention status. The inference uses the same ST-GCN model and
    preprocessing as the local real-time pipeline.
    """
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cfg = load_config()
    if len(request.sequence) > cfg.max_sequence_length:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence length {len(request.sequence)} exceeds maximum {cfg.max_sequence_length}",
        )

    try:
        result = service.predict(request.sequence)
    except Exception as exc:
        logging.getLogger("api.main").error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictResponse(**result)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["inference"],
)
async def analyze(
    request: AnalyzeRequest,
    service: InferenceService = Depends(get_inference_service),
):
    """Run full hybrid analysis on a pose sequence.

    Returns exercise classification (learned), repetition counts (both
    learned and state-machine), movement phases, form feedback
    (kinematic), rep boundaries, and calibrated confidence — the full
    fusion of neural and kinematic branches.
    """
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cfg = load_config()
    if len(request.sequence) > cfg.max_sequence_length:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence length {len(request.sequence)} exceeds maximum {cfg.max_sequence_length}",
        )

    try:
        result = service.analyze(
            request.sequence,
            exercise_mode=request.exercise_mode,
        )
    except Exception as exc:
        logging.getLogger("api.main").error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    return AnalyzeResponse(**result)
