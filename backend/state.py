"""Process-wide singletons: exercise registry, metrics, DB handles."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict, deque
from typing import Deque, Dict

import os

from exercise_system.models import ExerciseDefinition
from exercise_system.registry import ExerciseRegistry


class Metrics:
    """Tiny in-process observability (Prometheus-style names, exported as JSON/text)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.counters: Counter = Counter()
        self.latency: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=500))

    def inc(self, name: str, n: int = 1):
        with self.lock:
            self.counters[name] += n

    def observe(self, name: str, value_ms: float):
        with self.lock:
            self.latency[name].append(value_ms)

    def snapshot(self) -> dict:
        with self.lock:
            out = {"counters": dict(self.counters), "latency_ms": {}}
            for k, v in self.latency.items():
                s = sorted(v)
                if s:
                    out["latency_ms"][k] = {"p50": round(s[len(s) // 2], 3), "p95": round(s[int(len(s) * 0.95)], 3),
                                            "n": len(s)}
            req, unsup = self.counters.get("exercise_resolve_total", 0), self.counters.get("exercise_unsupported_total", 0)
            out["unsupported_exercise_rate"] = round(unsup / req, 3) if req else 0.0
            return out

    def prometheus(self) -> str:
        snap, lines = self.snapshot(), []
        for k, v in snap["counters"].items():
            lines.append(f"fitness_{k} {v}")
        for k, v in snap["latency_ms"].items():
            lines.append(f'fitness_latency_ms{{op="{k}",q="p50"}} {v["p50"]}')
            lines.append(f'fitness_latency_ms{{op="{k}",q="p95"}} {v["p95"]}')
        return "\n".join(lines) + "\n"


class AppState:
    def __init__(self):
        self.registry = ExerciseRegistry.load()
        self.metrics = Metrics()
        self.SessionLocal = None
        self.engine = None
        self.recognizers = None
        self.oidc = None
        self.jobs = None
        self.pose_factory = lambda: __import__('motion_engine.pose', fromlist=['make_estimator']).make_estimator(os.getenv('POSE_ESTIMATOR', 'mediapipe'))
        self.dataset_dir = os.getenv('DATASET_DIR', 'data/dataset')

    def init_recognizers(self, **kw):
        from .recognizers import RecognizerService
        self.recognizers = RecognizerService(self.registry, self.metrics, **kw)
        return self.recognizers

    def load_custom(self, db):
        from .db import CustomExercise
        for row in db.query(CustomExercise).all():
            try:
                self.registry.register(ExerciseDefinition.model_validate(row.data))
            except Exception:  # a bad stored definition must never take the service down
                self.metrics.inc("custom_exercise_load_errors_total")


state = AppState()
