"""Dataset health report: the numbers you need before trusting any model trained on it."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from .schema import ClipMeta, TRUSTED

MIN_CLIPS_PER_CLASS = 30
MIN_SUBJECTS_PER_CLASS = 8


def report(metas: Sequence[ClipMeta]) -> Dict:
    labeled = [m for m in metas if m.exercise_id]
    per_ex, subj_per_ex = Counter(m.exercise_id for m in labeled), defaultdict(set)
    for m in labeled:
        subj_per_ex[m.exercise_id].add(m.subject_id)
    views = defaultdict(Counter)
    for m in labeled:
        views[m.exercise_id][m.view] += 1
    trusted = [m for m in labeled if m.label_source in TRUSTED]
    warnings: List[str] = []
    for ex, n in per_ex.items():
        if n < MIN_CLIPS_PER_CLASS:
            warnings.append(f"{ex}: only {n} clips (< {MIN_CLIPS_PER_CLASS})")
        if len(subj_per_ex[ex]) < MIN_SUBJECTS_PER_CLASS:
            warnings.append(f"{ex}: only {len(subj_per_ex[ex])} subjects (< {MIN_SUBJECTS_PER_CLASS}) - generalisation untestable")
        if len([v for v in views[ex] if v != "unknown"]) < 2:
            warnings.append(f"{ex}: fewer than 2 camera views")
    if labeled and len(trusted) < len(labeled):
        warnings.append(f"{len(labeled) - len(trusted)}/{len(labeled)} labels are not human-verified (rule/model/user)")
    if any(not m.consent for m in metas if m.source.startswith("consented")):
        warnings.append("clips from users without consent present - must be removed")
    low_q = [m.clip_id for m in metas if m.quality < 0.8]
    if low_q:
        warnings.append(f"{len(low_q)} clips have <80% usable pose frames")
    counts = list(per_ex.values())
    return {"clips": len(metas), "labeled": len(labeled), "subjects": len({m.subject_id for m in metas}),
            "per_exercise": dict(per_ex), "subjects_per_exercise": {k: len(v) for k, v in subj_per_ex.items()},
            "views": {k: dict(v) for k, v in views.items()}, "label_sources": dict(Counter(m.label_source for m in labeled)),
            "trusted_fraction": round(len(trusted) / len(labeled), 3) if labeled else 0.0,
            "imbalance_ratio": round(max(counts) / min(counts), 2) if counts else None,
            "pose_models": sorted({m.pose_model for m in metas if m.pose_model}),
            "unlabeled": len(metas) - len(labeled), "warnings": warnings,
            "gate_ready": bool(labeled) and len(trusted) == len(labeled) and not warnings}
