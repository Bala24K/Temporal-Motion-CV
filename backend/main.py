import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import ModelVersion, make_engine, make_sessionmaker
from .routers import admin, auth, exercises, live, nutrition, profile, progress, sessions
from . import telemetry
from .buildinfo import ui_build_status
from .jobs import JobRunner
from .oidc import OIDCVerifier, fetch_jwks
from .state import state

log = logging.getLogger("fitness")
ROOT = Path(__file__).resolve().parent.parent
WEB_OUT = ROOT / "web" / "out"          # Next.js static export (npm run build in web/)
LEGACY = ROOT / "frontend"             # original dependency-free SPA, kept at /legacy/
RESEARCH_RESULTS = Path(__file__).resolve().parent.parent / "research/motion_upgrade/models/motion_stgcn_results.json"


def seed_models(db):
    """Register what we truthfully know. Nothing here is promoted automatically."""
    if db.query(ModelVersion).count():
        return
    db.add(ModelVersion(
        kind="recognizer", name="definition-rules", version=1, status="production", dataset_version="none (rule-based)",
        supported_exercises=[d.id for d in state.registry.recognisable()],
        metrics={"evaluation": "synthetic clips only (tests/test_engine.py); no real-world validation yet"},
        limitations="Pose-geometry rules. Cannot tell apart exercises that look identical to a camera "
                    "(curl vs hammer curl, lateral vs front raise) - asks the user to confirm. "
                    "No real-world, multi-subject validation yet."))
    if RESEARCH_RESULTS.exists():
        r = json.loads(RESEARCH_RESULTS.read_text())
        db.add(ModelVersion(
            kind="movement", name="stgcn-transformer", version=0, status="candidate",
            dataset_version="QEVD-subset-247clips", supported_exercises=r.get("classes", []),
            metrics={"test_accuracy": r["test"]["accuracy"], "test_macro_f1": r["test"]["macro_f1"],
                     "ece_after_temperature_scaling": r["calibration"]["ece_after"],
                     "label_source": "kinematic_rule", "release_ready": r.get("release_ready", False)},
            limitations=r.get("label_provenance", {}).get("note", "") +
                        ". 247 clips, video-disjoint but not subject-disjoint; test set has 37 clips.",
            artifact_uri="research/motion_upgrade/models/motion_stgcn.pt"))
    db.commit()


def init_app_state(database_url: str = None):
    telemetry.configure()
    state.init_recognizers()
    state.jobs = JobRunner()
    iss, aud = os.getenv("OIDC_ISSUER"), os.getenv("OIDC_AUDIENCE")
    state.oidc = None
    if iss and aud:
        url = os.getenv("OIDC_JWKS_URL") or iss.rstrip("/") + "/.well-known/jwks.json"
        state.oidc = OIDCVerifier(iss, aud, lambda: fetch_jwks(url))
    state.engine = make_engine(database_url or settings.database_url)
    state.SessionLocal = make_sessionmaker(state.engine)
    with state.SessionLocal() as db:
        state.load_custom(db)
        seed_models(db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.SessionLocal is None:
        init_app_state()
    yield


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Movement Intelligence Platform", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o],
                   allow_methods=["*"], allow_headers=["*"], allow_credentials=False)
for r in (auth, exercises, profile, sessions, progress, nutrition, admin, live):
    app.include_router(r.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    with telemetry.tracer.start_as_current_span("http.request") as span:
        response = await call_next(request)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.target", request.url.path)       # path only: never the query string (tokens)
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("request.id", rid)
    ms = (time.perf_counter() - t0) * 1000
    state.metrics.inc("http_requests_total")
    state.metrics.observe("http", ms)
    response.headers["x-request-id"] = rid
    response.headers.update({
        "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
        "Content-Security-Policy": ("default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; "
                                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                                    "font-src https://fonts.gstatic.com; img-src 'self' data:; media-src 'self' blob:; "
                                    "connect-src 'self' ws: wss: https://cdn.jsdelivr.net https://storage.googleapis.com; "
                                    "worker-src 'self' blob:; frame-ancestors 'none'")})
    log.info(json.dumps({"rid": rid, "method": request.method, "path": request.url.path,
                         "status": response.status_code, "ms": round(ms, 1)}))
    return response


@app.get("/api/health")
def health():
    ui = ui_build_status(ROOT / "web")
    if ui.get("stale"):
        log.warning("served UI (web/out) is stale - rebuild with: cd web && npm run build")
    return {"status": "ok", "supported_exercises": len(state.registry.recognisable()), "ui": ui}


@app.get("/api/metrics", response_class=PlainTextResponse)
def prom_metrics():
    return state.metrics.prometheus()


if LEGACY.exists():
    app.mount("/legacy", StaticFiles(directory=LEGACY, html=True), name="legacy")
if WEB_OUT.exists():                   # must be mounted last: it catches every path not matched above
    app.mount("/", StaticFiles(directory=WEB_OUT, html=True), name="web")
elif LEGACY.exists():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(LEGACY / "index.html")
