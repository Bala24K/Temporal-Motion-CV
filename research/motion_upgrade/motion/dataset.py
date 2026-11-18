"""
Dataset assembly for the learned motion model.

Two things this module treats as first-class that the previous pipeline
treated as afterthoughts:

LABEL PROVENANCE
    Every sample carries where its exercise label came from. The current
    QEVD labels are produced by `recognition.qevd_dataset.
    categorize_exercise_from_sequence`, a deterministic kinematic rule —
    so a model trained and tested on them is being scored on its ability
    to reproduce that rule, not on exercise recognition. That is a real
    finding about the experiment, not a reason to stop working, but it
    has to be visible at every stage rather than rediscovered in the
    write-up. `MotionDataset.provenance_summary()` makes it visible, and
    `motion/train.py` refuses to write a release checkpoint from
    rule-labelled test data unless explicitly overridden.

GROUP-DISJOINT SPLITTING
    Splits are by group, never by window. The current benchmark already
    splits by video; the group key here defaults to the video but accepts
    a subject id from a manifest, because two clips of the same person
    are not independent samples. A model can score well by recognising
    the person's proportions and habitual movement style, and a
    video-level split does not catch that.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import os
import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pose.landmarks import PoseFrame
from motion.graph import NUM_JOINTS
from motion.phase import (
    NUM_PHASES,
    PHASE_LABEL_SOURCE_DERIVED,
    derive_phase_labels,
)
from motion.tensors import (
    STREAM_NAMES,
    SkeletonWindow,
    build_streams,
    build_window,
)

# The four supported exercises, in fixed order, plus the rejection class.
# `other` is not optional decoration: without it every unrelated movement
# in front of the camera is forced into one of four buckets, usually with
# high confidence, which is the single most visible failure mode in a
# live demo.
EXERCISE_CLASSES: List[str] = ["bicep_curl", "squat", "shoulder_press", "lateral_raise"]
OTHER_CLASS = "other"

LABEL_SOURCE_RULE = "kinematic_rule"      # derived by categorize_exercise_from_sequence
LABEL_SOURCE_HUMAN = "human_verified"     # reviewed by a person
LABEL_SOURCE_DATASET = "dataset_provided" # shipped with the source corpus

TRUSTED_LABEL_SOURCES = (LABEL_SOURCE_HUMAN, LABEL_SOURCE_DATASET)

DEFAULT_WINDOW_FRAMES = 48
DEFAULT_SOURCE_FPS = 7.5  # 30 fps video sampled at frame_step=4


@dataclass
class MotionSample:
    """One training window and everything needed to audit it."""
    window: SkeletonWindow
    label: str
    label_source: str
    group: str                 # split key — subject id where known, else clip id
    clip_id: str
    phase_labels: np.ndarray   # (T,) int64
    phase_label_source: str
    phase_diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def valid_ratio(self) -> float:
        return self.window.valid_ratio


class MotionDataset:
    """A list of MotionSamples with the tensor-building and auditing helpers."""

    def __init__(self, samples: Sequence[MotionSample], classes: Optional[Sequence[str]] = None):
        self.samples: List[MotionSample] = list(samples)
        if classes is not None:
            self.classes: List[str] = list(classes)
        else:
            present = {s.label for s in self.samples}
            self.classes = [c for c in EXERCISE_CLASSES + [OTHER_CLASS] if c in present]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def subset(self, indices: Iterable[int]) -> "MotionDataset":
        return MotionDataset([self.samples[i] for i in indices], classes=self.classes)

    @property
    def class_to_index(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    @property
    def groups(self) -> List[str]:
        return [s.group for s in self.samples]

    def to_arrays(
        self,
        streams: Sequence[str] = STREAM_NAMES,
        include_validity_channel: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Materialise the dataset as (X, y, phases, group_ids).

            X       (N, C, T, J) float32
            y       (N,)         int64, index into self.classes
            phases  (N, T)       int64
            groups  (N,)         object array of group keys

        Built eagerly. At the current corpus size (hundreds of clips of
        ~48 frames) the whole thing is a few tens of MB, and eager
        construction keeps training reproducible and profiling honest. If
        the corpus grows past memory, this is the seam to replace with a
        lazy Dataset — nothing else needs to change.
        """
        if not self.samples:
            raise ValueError("cannot build arrays from an empty dataset")

        idx = self.class_to_index
        xs = [build_streams(s.window, streams, include_validity_channel) for s in self.samples]
        X = np.stack(xs, axis=0).astype(np.float32)
        y = np.array([idx[s.label] for s in self.samples], dtype=np.int64)
        phases = np.stack([s.phase_labels for s in self.samples], axis=0).astype(np.int64)
        groups = np.array([s.group for s in self.samples], dtype=object)
        return X, y, phases, groups

    def class_counts(self) -> Dict[str, int]:
        return dict(Counter(s.label for s in self.samples))

    def provenance_summary(self) -> Dict[str, object]:
        """What a reader needs in order to know how much a score on this
        dataset is worth."""
        by_source = Counter(s.label_source for s in self.samples)
        trusted = sum(n for src, n in by_source.items() if src in TRUSTED_LABEL_SOURCES)
        n = max(1, len(self.samples))
        return {
            "n_samples": len(self.samples),
            "n_groups": len(set(self.groups)),
            "class_counts": self.class_counts(),
            "label_sources": dict(by_source),
            "trusted_label_fraction": trusted / n,
            "mean_valid_ratio": float(np.mean([s.valid_ratio for s in self.samples])),
            "phase_label_sources": dict(Counter(s.phase_label_source for s in self.samples)),
        }

    def has_only_derived_labels(self) -> bool:
        return all(s.label_source not in TRUSTED_LABEL_SOURCES for s in self.samples)


def _windows_from_frames(
    pose_frames: Sequence[PoseFrame],
    window_frames: int,
    windows_per_clip: int,
    min_frames: int,
) -> List[Sequence[PoseFrame]]:
    """
    Split a clip into one or more overlapping frame ranges.

    Multiple windows per clip buy extra training examples cheaply, but
    they are not independent — they share a person, a camera, and usually
    the same reps. They therefore always inherit the clip's group key so
    the splitter cannot scatter them across train and test.
    """
    n = len(pose_frames)
    if n < min_frames:
        return []
    if windows_per_clip <= 1 or n <= window_frames:
        return [pose_frames]

    span = min(window_frames, n)
    starts = np.linspace(0, n - span, num=windows_per_clip).round().astype(int)
    seen: set = set()
    out: List[Sequence[PoseFrame]] = []
    for s in starts:
        key = int(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(pose_frames[key:key + span])
    return out


def build_dataset_from_pose_cache(
    cache_path: str = "data/qevd_pose_cache.pkl",
    label_source: str = LABEL_SOURCE_RULE,
    subject_map: Optional[Dict[str, str]] = None,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    windows_per_clip: int = 1,
    min_frames: int = 8,
    min_valid_ratio: float = 0.35,
    max_per_class: Optional[int] = None,
    classes: Optional[Sequence[str]] = None,
    seed: int = 42,
    source_fps: float = DEFAULT_SOURCE_FPS,
) -> MotionDataset:
    """
    Build a dataset from the cached pose extraction.

    The cache (`data/qevd_pose_cache.pkl`) maps clip filename ->
    {video_id, pose_frames, label}. Reading from it instead of decoding
    video again turns dataset construction from a multi-hour job into
    seconds, which is what makes iterating on the representation
    practical.

    subject_map — clip id -> subject id. When supplied, splits become
    subject-disjoint; without it the group key falls back to the clip id
    and the split is only video-disjoint. See `load_subject_map`.

    min_valid_ratio drops windows where most joint-frames were
    interpolated rather than detected. Those windows are mostly a record
    of MediaPipe's failure modes, and training on them teaches the model
    the shape of the gap-filler.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"pose cache not found at {cache_path!r}. "
            "Run recognition/qevd_dataset.py's extraction first, or pass a different --cache."
        )
    with open(cache_path, "rb") as fh:
        cache = pickle.load(fh)

    rng = np.random.default_rng(seed)
    allowed = set(classes) if classes is not None else set(EXERCISE_CLASSES + [OTHER_CLASS])

    by_class: Dict[str, List[MotionSample]] = defaultdict(list)
    skipped = Counter()

    for clip_id in sorted(cache.keys()):
        entry = cache[clip_id]
        label = entry.get("label")
        if label not in allowed:
            skipped["label_not_in_classes"] += 1
            continue

        pose_frames = entry.get("pose_frames") or []
        group = (subject_map or {}).get(clip_id, clip_id)

        for chunk in _windows_from_frames(pose_frames, window_frames, windows_per_clip, min_frames):
            window = build_window(chunk, target_len=window_frames, source_fps=source_fps)
            if window is None:
                skipped["no_normalisable_frame"] += 1
                continue
            if window.valid_ratio < min_valid_ratio:
                skipped["low_valid_ratio"] += 1
                continue

            phase_labels, phase_diag = derive_phase_labels(window.coords, label)
            by_class[label].append(
                MotionSample(
                    window=window,
                    label=label,
                    label_source=label_source,
                    group=str(group),
                    clip_id=str(clip_id),
                    phase_labels=phase_labels,
                    phase_label_source=PHASE_LABEL_SOURCE_DERIVED,
                    phase_diagnostics=phase_diag,
                )
            )

    samples: List[MotionSample] = []
    for label in sorted(by_class):
        items = by_class[label]
        if max_per_class is not None and len(items) > max_per_class:
            # Subsample by GROUP, not by window: taking N random windows
            # from a class can silently keep several windows of one clip
            # and none of another, skewing the group distribution the
            # splitter then has to work with.
            groups = sorted({s.group for s in items})
            rng.shuffle(groups)
            kept: List[MotionSample] = []
            for g in groups:
                if len(kept) >= max_per_class:
                    break
                kept.extend(s for s in items if s.group == g)
            items = kept[:max_per_class]
        samples.extend(items)

    dataset = MotionDataset(samples, classes=classes or None)
    dataset.build_report = {  # type: ignore[attr-defined]
        "cache_path": cache_path,
        "skipped": dict(skipped),
        "window_frames": window_frames,
        "windows_per_clip": windows_per_clip,
        "subject_map_used": subject_map is not None,
    }
    return dataset


def load_subject_map(path: str) -> Dict[str, str]:
    """
    Read a two-column CSV, `clip_id,subject_id`, with an optional header.

    Kept deliberately trivial so that producing a subject map is a
    twenty-minute job with a spreadsheet rather than a blocked task —
    subject-disjoint evaluation is the single change that most affects
    whether the reported accuracy means anything, and it should not be
    gated behind tooling.
    """
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            if i == 0 and parts[1].lower() in ("subject", "subject_id"):
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def group_disjoint_split(
    dataset: MotionDataset,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[MotionDataset, MotionDataset, MotionDataset]:
    """
    Split into train/val/test with no group appearing in more than one
    part.

    Groups are assigned greedily, largest first, to whichever split is
    furthest below its target size — with a small corpus a purely random
    group split routinely produces a test set missing an entire class,
    and a stratified group split is worth the twenty lines.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    if len(dataset) == 0:
        raise ValueError("cannot split an empty dataset")

    rng = np.random.default_rng(seed)
    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        by_group[s.group].append(i)

    groups = sorted(by_group.keys())
    rng.shuffle(groups)

    n_total = len(dataset)
    targets = [r * n_total for r in ratios]
    buckets: List[List[int]] = [[], [], []]

    # Largest groups first so a single big group can't overshoot a small split.
    for g in sorted(groups, key=lambda g: (-len(by_group[g]), g)):
        deficits = [targets[k] - len(buckets[k]) for k in range(3)]
        k = int(np.argmax(deficits))
        buckets[k].extend(by_group[g])

    train, val, test = (dataset.subset(sorted(b)) for b in buckets)

    for name, part in (("train", train), ("val", val), ("test", test)):
        if len(part) == 0:
            raise ValueError(
                f"the {name} split came out empty — there are only "
                f"{len(groups)} groups for ratios {ratios}. Use more clips, "
                "more windows per clip, or a coarser split."
            )
    return train, val, test


def assert_no_group_leakage(*parts: MotionDataset) -> None:
    """
    Hard check that no group key appears in two splits.

    Called by the training entry point rather than only by tests: a
    leakage bug produces a *better* number, so nothing about the run will
    look wrong, and the failure will be invisible until someone tries to
    reproduce it on new data.
    """
    seen: Dict[str, int] = {}
    for i, part in enumerate(parts):
        for g in set(part.groups):
            if g in seen and seen[g] != i:
                raise AssertionError(
                    f"group {g!r} appears in split {seen[g]} and split {i} — "
                    "this invalidates the evaluation"
                )
            seen[g] = i


def class_weights(dataset: MotionDataset) -> np.ndarray:
    """
    Inverse-frequency weights over `dataset.classes`, normalised to mean 1.

    The rule-labelled cache is roughly 68% squat; unweighted training on
    it produces a model that predicts squat and scores well on accuracy
    while being useless. Macro F1 is the metric that catches this, and
    these weights are what stop it happening in the first place.
    """
    counts = dataset.class_counts()
    w = np.array([1.0 / max(1, counts.get(c, 0)) for c in dataset.classes], dtype=np.float32)
    return (w / w.mean()).astype(np.float32)
