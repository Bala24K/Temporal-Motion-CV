"""Subject-disjoint, exercise-stratified splitting with a leakage assertion."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence

from .schema import ClipMeta


class LeakageError(AssertionError):
    pass


def subject_disjoint_split(metas: Sequence[ClipMeta], fractions=(0.7, 0.15, 0.15), seed: int = 0,
                           names=("train", "val", "test")) -> Dict[str, List[str]]:
    """Whole subjects go to exactly one split. Greedy: place each subject (largest first, shuffled within
    ties) into the split currently furthest below its target for that subject's dominant exercises."""
    by_subj: Dict[str, List[ClipMeta]] = defaultdict(list)
    for m in metas:
        by_subj[m.subject_id].append(m)
    rng = random.Random(seed)
    subjects = list(by_subj)
    rng.shuffle(subjects)
    subjects.sort(key=lambda s: -len(by_subj[s]))
    target_total = len(metas)
    counts = {n: 0 for n in names}
    ex_counts = {n: defaultdict(int) for n in names}
    ex_total = defaultdict(int)
    for m in metas:
        ex_total[m.exercise_id] += 1
    out = {n: [] for n in names}
    for s in subjects:
        clips = by_subj[s]
        def deficit(n, f):
            d = f * target_total - counts[n]
            for m in clips:
                d += (f * ex_total[m.exercise_id] - ex_counts[n][m.exercise_id]) / max(1, ex_total[m.exercise_id])
            return d
        pick = max(zip(names, fractions), key=lambda nf: deficit(*nf))[0]
        for m in clips:
            out[pick].append(m.clip_id)
            ex_counts[pick][m.exercise_id] += 1
        counts[pick] += len(clips)
    assert_no_leakage(metas, out)
    return out


def assert_no_leakage(metas: Sequence[ClipMeta], split: Dict[str, List[str]]) -> None:
    subj = {m.clip_id: m.subject_id for m in metas}
    seen: Dict[str, str] = {}
    for name, ids in split.items():
        for cid in ids:
            s = subj[cid]
            if s in seen and seen[s] != name:
                raise LeakageError(f"subject {s!r} appears in both {seen[s]!r} and {name!r}")
            seen[s] = name
    all_ids = [c for ids in split.values() for c in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(subj):
        raise LeakageError("split must cover every clip exactly once")
