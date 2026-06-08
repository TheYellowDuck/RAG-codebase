"""Bootstrap confidence intervals (outline §6.5).

With N ≈ 40 questions, report CIs rather than bare point estimates — resample
questions with replacement, recompute the mean, take the 2.5/97.5 percentiles.
Cheap, and it shows a 0.85 vs 0.88 difference on 40 questions might be noise.
"""
from __future__ import annotations

import random
from typing import Sequence


def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """Return {mean, lo, hi} for the mean of `values` via the percentile bootstrap."""
    values = list(values)
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1
    return {
        "mean": sum(values) / n,
        "lo": means[lo_idx],
        "hi": means[min(hi_idx, n_resamples - 1)],
        "n": n,
    }
