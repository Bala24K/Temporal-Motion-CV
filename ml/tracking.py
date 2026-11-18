"""Experiment tracking. Writes a local JSONL run log always (so runs are reproducible without a server) and
mirrors to MLflow when MLFLOW_TRACKING_URI is set and mlflow is installed."""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Optional


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class Run:
    def __init__(self, name: str, params: Dict, root: str = "runs", dataset_version: str = ""):
        self.id, self.name, self.t0 = uuid.uuid4().hex[:10], name, time.time()
        self.path = Path(root) / f"{self.id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rec = {"id": self.id, "name": name, "params": params, "dataset_version": dataset_version,
                    "code_commit": git_commit(), "metrics": {}, "history": [], "artifacts": {}}
        self._ml = None
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            try:                                     # pragma: no cover - needs mlflow
                import mlflow
                mlflow.set_tracking_uri(uri)
                self._ml = mlflow
                mlflow.start_run(run_name=name)
                mlflow.log_params({**params, "dataset_version": dataset_version, "code_commit": self.rec["code_commit"]})
            except Exception:
                self._ml = None

    def log(self, step: int, **metrics: float) -> None:
        self.rec["history"].append({"step": step, **metrics})
        if self._ml:                                  # pragma: no cover
            self._ml.log_metrics(metrics, step=step)

    def finish(self, metrics: Dict, artifacts: Optional[Dict[str, str]] = None) -> Dict:
        self.rec["metrics"], self.rec["artifacts"] = metrics, artifacts or {}
        self.rec["seconds"] = round(time.time() - self.t0, 2)
        self.path.write_text(json.dumps(self.rec, indent=2, default=float))
        if self._ml:                                  # pragma: no cover
            self._ml.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            self._ml.end_run()
        return self.rec
