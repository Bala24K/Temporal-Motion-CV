# ── Motion ST-GCN Inference Service ────────────────────────────────────
# Containerized ML inference server for the learned spatial-temporal graph
# neural network (ST-GCN) and hybrid movement engine.
# No camera, no GUI, no display hardware required.
#
# Build (from workspace root):
#   docker build -f motion-upgrade/Dockerfile -t motion-stgcn-api .
#
# Build (from motion-upgrade directory):
#   docker build -t motion-stgcn-api .
#
# Run:
#   docker run -p 8000:8000 motion-stgcn-api
#
# Health:
#   curl http://localhost:8000/health

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/motion-upgrade:/app/Ver-1:/app"

WORKDIR /app

# ── System dependencies ─────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────
# 1. Install lightweight CPU PyTorch wheel (~150MB instead of ~2.5GB CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Copy requirement manifests
COPY Ver-1/requirements.txt /app/Ver-1/requirements.txt
COPY motion-upgrade/requirements-motion.txt /app/motion-upgrade/requirements-motion.txt
COPY motion-upgrade/requirements-api.txt /app/motion-upgrade/requirements-api.txt

# 3. Install core, motion, and API dependencies
RUN pip install --no-cache-dir -r /app/Ver-1/requirements.txt && \
    pip install --no-cache-dir -r /app/motion-upgrade/requirements-motion.txt && \
    pip install --no-cache-dir -r /app/motion-upgrade/requirements-api.txt && \
    pip install --no-cache-dir opencv-python-headless && \
    pip uninstall -y opencv-python 2>/dev/null || true

# ── Application Code & Model Weights ────────────────────────────────
# Copy Ver-1 (domain logic, pose landmarks, biomechanics)
COPY Ver-1/ /app/Ver-1/

# Copy motion-upgrade (ST-GCN model, tensors, calibration, phase, API)
COPY motion-upgrade/ /app/motion-upgrade/

WORKDIR /app/motion-upgrade

# ── Runtime Configuration ───────────────────────────────────────────
ENV MODEL_PATH=/app/motion-upgrade/models/motion_stgcn_best.pt \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=info

EXPOSE 8000

# ── Health Check ────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Entrypoint ──────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
