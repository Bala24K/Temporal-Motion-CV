"""Content-addressed dataset store: npz clips + JSONL manifest/annotations + immutable frozen versions.

Raw video is never stored (privacy): only landmarks. Versions are hashes of clip bytes + annotations, so
`dataset_version` in the model registry means exactly one dataset state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from motion_engine.skeleton import Frame, REAL_JOINTS
from .schema import ClipAnnotation, ClipMeta


def frames_to_landmarks(frames: Sequence[Frame]) -> np.ndarray:
    """(T, 13, 3) float32; NaN where a joint is missing."""
    out = np.full((len(frames), len(REAL_JOINTS), 3), np.nan, np.float32)
    for t, f in enumerate(frames):
        for j, n in enumerate(REAL_JOINTS):
            if n in f:
                out[t, j, :len(f[n])] = f[n][:3]
    return out


def landmarks_to_frames(arr: np.ndarray) -> List[Frame]:
    return [{n: [float(v) for v in arr[t, j]] for j, n in enumerate(REAL_JOINTS) if np.isfinite(arr[t, j, :2]).all()}
            for t in range(len(arr))]


class DatasetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "clips").mkdir(parents=True, exist_ok=True)
        (self.root / "versions").mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        self.ann_path = self.root / "annotations.jsonl"

    # ---- clips
    def add_clip(self, frames: Sequence[Frame], timestamps: Sequence[float], meta: ClipMeta) -> ClipMeta:
        if meta.source.startswith("consented") and not meta.consent:
            raise ValueError("clips from users require explicit consent")
        if meta.clip_id in self.meta_index():
            raise ValueError(f"clip {meta.clip_id} already exists (clips are immutable)")
        lm = frames_to_landmarks(frames)
        path = self.root / "clips" / f"{meta.clip_id}.npz"
        np.savez_compressed(path, landmarks=lm, t=np.asarray(timestamps, np.float32))
        meta = meta.model_copy(update={"n_frames": len(frames), "sha256": hashlib.sha256(lm.tobytes()).hexdigest(),
                                       "quality": float(np.isfinite(lm[:, :, 0]).all(1).mean()) if len(lm) else 0.0})
        with self.manifest_path.open("a") as f:
            f.write(meta.model_dump_json() + "\n")
        return meta

    def meta_index(self) -> Dict[str, ClipMeta]:
        if not self.manifest_path.exists():
            return {}
        return {m.clip_id: m for m in (ClipMeta.model_validate_json(l) for l in self.manifest_path.read_text().splitlines() if l)}

    def load_clip(self, clip_id: str) -> Tuple[List[Frame], np.ndarray]:
        z = np.load(self.root / "clips" / f"{clip_id}.npz")
        return landmarks_to_frames(z["landmarks"]), z["t"]

    # ---- annotations (latest annotation per clip wins; history is kept in the file)
    def annotate(self, ann: ClipAnnotation) -> None:
        if ann.clip_id not in self.meta_index():
            raise KeyError(f"unknown clip {ann.clip_id}")
        with self.ann_path.open("a") as f:
            f.write(ann.model_dump_json() + "\n")

    def annotations(self) -> Dict[str, ClipAnnotation]:
        if not self.ann_path.exists():
            return {}
        out = {}
        for l in self.ann_path.read_text().splitlines():
            if l:
                a = ClipAnnotation.model_validate_json(l)
                out[a.clip_id] = a
        return out

    def effective(self) -> List[ClipMeta]:
        """Meta with annotations applied: an annotation upgrades label_source/exercise/view for that clip."""
        anns, out = self.annotations(), []
        for m in self.meta_index().values():
            a = anns.get(m.clip_id)
            if a:
                m = m.model_copy(update={"exercise_id": a.exercise_id or m.exercise_id,
                                         "view": a.view if a.view != "unknown" else m.view,
                                         "label_source": a.label_source})
            out.append(m)
        return out

    # ---- versions
    def version_id(self) -> str:
        h = hashlib.sha256()
        for m in sorted(self.effective(), key=lambda m: m.clip_id):
            h.update(f"{m.clip_id}|{m.sha256}|{m.exercise_id}|{m.view}|{m.label_source}|{m.subject_id}".encode())
        anns = self.annotations()
        for cid in sorted(anns):
            h.update(anns[cid].model_dump_json().encode())
        return "ds-" + h.hexdigest()[:12]

    def freeze(self, name: str) -> dict:
        p = self.root / "versions" / f"{name}.json"
        if p.exists():
            raise FileExistsError(f"version {name!r} is frozen and immutable")
        rec = {"name": name, "id": self.version_id(), "clips": sorted(m.clip_id for m in self.effective())}
        p.write_text(json.dumps(rec, indent=2))
        return rec
