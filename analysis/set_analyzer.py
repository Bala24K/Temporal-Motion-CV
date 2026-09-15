"""
Set-level analysis (V2 sections 9-10): aggregate a list of per-rep metrics
into the kind of set summary the doc's mockup shows —

    ROM          Avg 119   Best 125   Lowest 104
    Tempo        Avg 2.14s  Variation 8%
    Movement consistency   91%
    Primary observation:  ROM decreased during final 3 reps.

Trend detection here is intentionally simple (first-vs-last segment
comparison, not a full changepoint model) and always phrased as an
observation about the movement, never a claim about *why* it happened
(fatigue, motivation, etc.) — see the doc's section 10 distinction between
"movement degradation associated with fatigue" and claiming to measure
fatigue itself.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from analysis.rep_analyzer import RepMetrics


@dataclass
class SetSummary:
    rep_count: int
    avg_rom: float
    best_rom: float
    lowest_rom: float
    rom_consistency: float          # 0-100, higher = more consistent
    avg_tempo: float
    tempo_consistency: float        # 0-100
    avg_symmetry: Optional[float]
    avg_trajectory_similarity: Optional[float]
    observations: List[str] = field(default_factory=list)


def _consistency_score(values: List[float]) -> float:
    """100 * (1 - coefficient of variation), clamped to [0, 100]. A tight
    cluster of values -> near 100; wildly varying values -> near 0."""
    if len(values) < 2:
        return 100.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if abs(mean) < 1e-9:
        return 100.0
    cv = float(np.std(arr)) / abs(mean)
    return float(max(0.0, min(100.0, 100.0 * (1.0 - cv))))


def _segment_trend(values: List[float], metric_label: str,
                    pct_threshold: float = 0.08, min_reps: int = 5) -> Optional[str]:
    """
    Compares the mean of the first ~40% of reps against the last ~40%
    (skipping the middle) and reports a plain-language observation if the
    change exceeds pct_threshold. Requires at least min_reps to say
    anything — trend claims from 3 data points aren't defensible.
    """
    n = len(values)
    if n < min_reps:
        return None
    seg = max(1, int(round(n * 0.4)))
    first = values[:seg]
    last = values[-seg:]
    first_mean = float(np.mean(first))
    last_mean = float(np.mean(last))
    if abs(first_mean) < 1e-9:
        return None
    pct_change = (last_mean - first_mean) / abs(first_mean)
    if abs(pct_change) < pct_threshold:
        return None
    direction = "decreased" if pct_change < 0 else "increased"
    return f"{metric_label} {direction} during the final {seg} reps."


def summarize_set(reps: List[RepMetrics]) -> Optional[SetSummary]:
    if not reps:
        return None

    roms = [r.rom for r in reps]
    tempos = [r.tempo for r in reps]
    symmetries = [r.symmetry.symmetry_score for r in reps if r.symmetry is not None]
    trajs = [r.trajectory_similarity for r in reps if r.trajectory_similarity is not None]

    observations = []
    rom_trend = _segment_trend(roms, "ROM")
    if rom_trend:
        observations.append(rom_trend)
    tempo_trend = _segment_trend(tempos, "Tempo")
    if tempo_trend:
        observations.append(tempo_trend)
    if symmetries:
        sym_trend = _segment_trend(symmetries, "Symmetry")
        if sym_trend:
            observations.append(sym_trend)

    return SetSummary(
        rep_count=len(reps),
        avg_rom=float(np.mean(roms)),
        best_rom=float(np.max(roms)),
        lowest_rom=float(np.min(roms)),
        rom_consistency=_consistency_score(roms),
        avg_tempo=float(np.mean(tempos)),
        tempo_consistency=_consistency_score(tempos),
        avg_symmetry=float(np.mean(symmetries)) if symmetries else None,
        avg_trajectory_similarity=float(np.mean(trajs)) if trajs else None,
        observations=observations,
    )
