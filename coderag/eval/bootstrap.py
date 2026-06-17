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


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Paired comparison of two configs scored on the SAME questions (a[i], b[i]).

    Returns the mean per-question difference (a − b), a bootstrap CI for it, and a
    two-sided p-value (fraction of resamples whose mean diff lands on the opposite
    side of 0). Lets us say "significant / not" instead of eyeballing overlapping
    CIs — the difference is paired, so it's far more sensitive than comparing two
    independent intervals."""
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("paired_bootstrap needs two equal, non-empty score lists")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    resampled = []
    for _ in range(n_resamples):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        resampled.append(s)
    resampled.sort()
    lo = resampled[int((1 - ci) / 2 * n_resamples)]
    hi = resampled[min(int((1 + ci) / 2 * n_resamples) - 1, n_resamples - 1)]
    mean_diff = sum(diffs) / n
    # two-sided p: how often the resampled mean diff sits on the null side of 0
    frac_le0 = sum(1 for d in resampled if d <= 0) / n_resamples
    frac_ge0 = sum(1 for d in resampled if d >= 0) / n_resamples
    p = 2 * min(frac_le0, frac_ge0)
    return {"mean_diff": mean_diff, "lo": lo, "hi": hi,
            "p_value": min(1.0, p), "n": n, "significant": p < 0.05}
