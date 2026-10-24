"""Paired bootstrap confidence intervals (pre-registered: 10k resamples, seed 20261024)."""
from __future__ import annotations

import numpy as np


def bootstrap_ci_mean(indicators: list[int], n_boot: int = 10_000,
                      seed: int = 20261024, alpha: float = 0.05) -> dict[str, float]:
    x = np.asarray(indicators, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n": n,
        "rate": float(x.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot": n_boot,
        "seed": seed,
    }
