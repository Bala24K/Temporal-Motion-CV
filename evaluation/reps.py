"""
Rep-counting error metrics (V1 §18 / V2 §19).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class RepCountingMetrics:
    total_actual: int
    total_detected: int
    absolute_error: int          # sum of |detected - actual| across sets
    signed_error: int            # sum of (detected - actual); positive = net over-counting
    per_set_errors: List[int] = field(default_factory=list)

    @property
    def mean_absolute_error(self) -> float:
        if not self.per_set_errors:
            return 0.0
        return sum(abs(e) for e in self.per_set_errors) / len(self.per_set_errors)


def rep_counting_error(actual_counts: List[int], detected_counts: List[int]) -> RepCountingMetrics:
    if len(actual_counts) != len(detected_counts):
        raise ValueError("actual_counts and detected_counts must be the same length")
    per_set_errors = [d - a for a, d in zip(actual_counts, detected_counts)]
    return RepCountingMetrics(
        total_actual=sum(actual_counts),
        total_detected=sum(detected_counts),
        absolute_error=sum(abs(e) for e in per_set_errors),
        signed_error=sum(per_set_errors),
        per_set_errors=per_set_errors,
    )
