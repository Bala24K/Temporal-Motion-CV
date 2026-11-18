"""Minimal in-process job runner (thread pool) with the interface a Celery adapter would implement.
Adopt Celery+Redis when you run >1 API replica or jobs must survive restarts (docs/STACK.md)."""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict


class JobRunner:
    def __init__(self, workers: int = 2):
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="job")
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def submit(self, fn: Callable, *a, owner: int = 0, kind: str = "job", **kw) -> str:
        jid = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[jid] = {"id": jid, "kind": kind, "status": "queued", "result": None, "error": None, "owner": owner}

        def run():
            self._set(jid, status="running")
            try:
                self._set(jid, status="done", result=fn(*a, **kw))
            except Exception as e:
                self._set(jid, status="failed", error=f"{type(e).__name__}: {e}")
        self.pool.submit(run)
        return jid

    def _set(self, jid, **kw):
        with self.lock:
            self.jobs[jid].update(kw)

    def get(self, jid: str):
        with self.lock:
            j = self.jobs.get(jid)
            return dict(j) if j else None

    def wait(self, jid: str, timeout: float = 30.0):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.get(jid)
            if j and j["status"] in ("done", "failed"):
                return j
            time.sleep(0.02)
        raise TimeoutError(jid)
