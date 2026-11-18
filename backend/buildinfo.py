"""Is the served UI (web/out) built from the current UI sources?

`web/next.config.mjs` writes `public/build-info.json` (copied to `out/`) with a hash of the sources at build time.
Here the same hash is recomputed from `web/` so /api/health can say `stale: true` instead of silently serving an
old UI. Keep the algorithm identical to `sourceHash` in next.config.mjs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

SOURCE_DIRS = ("app", "components", "hooks", "lib")


def source_hash(web: Path) -> Optional[str]:
    files = [p for d in SOURCE_DIRS if (web / d).is_dir() for p in (web / d).rglob("*") if p.is_file()]
    if not files:
        return None                       # sources not shipped (e.g. the Docker image): nothing to compare
    files.append(web / "package.json")
    h = hashlib.sha256()
    for rel, p in sorted((p.relative_to(web).as_posix(), p) for p in files):
        h.update(rel.encode() + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def ui_build_status(web: Path) -> dict:
    out = web / "out"
    if not out.is_dir():
        return {"served": False}
    info_path = out / "build-info.json"
    built = json.loads(info_path.read_text()).get("source_hash") if info_path.exists() else None
    current = source_hash(web)
    stale = None if current is None else built != current
    return {"served": True, "built_hash": built, "source_hash": current, "stale": stale}
