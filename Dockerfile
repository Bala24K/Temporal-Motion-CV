# ---- build the Next.js static export
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- API + static UI
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend backend
COPY motion_engine motion_engine
COPY exercise_system exercise_system
COPY nutrition nutrition
COPY ml/__init__.py ml/seq.py ml/tracking.py ml/
COPY ml/data ml/data
COPY frontend frontend
COPY --from=web /web/out web/out
# research/ and torch training code (ml/torch_models.py, ml/train_*.py) are intentionally not shipped: serving uses ONNX only.
RUN useradd -m app && mkdir -p /app/data && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
