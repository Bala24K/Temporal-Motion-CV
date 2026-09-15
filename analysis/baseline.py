"""
Personal baseline (V2 section 11).

Instead of one universal threshold ("ROM < 110 = bad") for every body and
every exercise, learn what's normal FOR THIS PERSON and flag deviation
from their own baseline. Uses Welford's online algorithm so it doesn't
need to store raw history to keep updating mean/variance — cheap to persist
across sessions (see to_dict/from_dict) as the set of running stats grows.

A baseline needs a minimum number of samples before it's meaningful — see
`is_established`. Don't flag deviations against a baseline built from 2
data points.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import json


@dataclass
class RunningStat:
    n: int = 0
    mean: float = 0.0
    _m2: float = 0.0  # sum of squared differences from the mean (Welford's algorithm)

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        # sample variance (n-1 denominator); undefined below 2 samples
        if self.n < 2:
            return 0.0
        return self._m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self._m2}

    @classmethod
    def from_dict(cls, d: dict) -> "RunningStat":
        s = cls()
        s.n = d["n"]
        s.mean = d["mean"]
        s._m2 = d["m2"]
        return s


class PersonalBaseline:
    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self._stats: Dict[str, RunningStat] = {}

    def update(self, metric: str, value: float):
        self._stats.setdefault(metric, RunningStat()).update(value)

    def is_established(self, metric: str) -> bool:
        stat = self._stats.get(metric)
        return stat is not None and stat.n >= self.min_samples

    def z_score(self, metric: str, value: float) -> Optional[float]:
        """How many personal-baseline standard deviations `value` is from
        this person's own mean for `metric`. None if the baseline isn't
        established yet or has (near-)zero variance."""
        if not self.is_established(metric):
            return None
        stat = self._stats[metric]
        if stat.std < 1e-9:
            return None
        return (value - stat.mean) / stat.std

    def is_deviation(self, metric: str, value: float, threshold: float = 2.0) -> Optional[bool]:
        z = self.z_score(metric, value)
        if z is None:
            return None
        return abs(z) >= threshold

    def summary(self, metric: str) -> Optional[dict]:
        stat = self._stats.get(metric)
        if stat is None or stat.n == 0:
            return None
        return {"n": stat.n, "mean": stat.mean, "std": stat.std,
                "established": self.is_established(metric)}

    def to_dict(self) -> dict:
        return {"min_samples": self.min_samples,
                "stats": {k: v.to_dict() for k, v in self._stats.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalBaseline":
        b = cls(min_samples=d.get("min_samples", 5))
        b._stats = {k: RunningStat.from_dict(v) for k, v in d.get("stats", {}).items()}
        return b

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PersonalBaseline":
        with open(path) as f:
            return cls.from_dict(json.load(f))
