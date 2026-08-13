"""Moving-block bootstrap utilities for serially dependent forecast scores."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def block_length_sqrt_n(n: int) -> int:
    """Conservative documented default block length: ceil(sqrt(n))."""
    if n < 1:
        raise ValueError("n must be positive")
    return max(1, int(np.ceil(np.sqrt(n))))


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Sample contiguous blocks with replacement until n dependent observations are drawn."""
    if n < 1 or block_length < 1:
        raise ValueError("n and block_length must be positive")
    starts = rng.integers(0, max(1, n - block_length + 1), size=int(np.ceil(n / block_length)))
    sample = np.concatenate([np.arange(start, min(start + block_length, n)) for start in starts])
    return sample[:n]


def bootstrap_metric_difference(
    y_true: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = 1000,
    block_length: int | None = None,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap candidate-minus-baseline performance with contiguous resampling."""
    y_true = np.asarray(y_true)
    candidate = np.asarray(candidate)
    baseline = np.asarray(baseline)
    valid = np.isfinite(y_true) & np.isfinite(candidate) & np.isfinite(baseline)
    y_true, candidate, baseline = y_true[valid], candidate[valid], baseline[valid]
    if len(y_true) < 2:
        raise ValueError("At least two finite paired predictions are required")
    length = block_length or block_length_sqrt_n(len(y_true))
    rng = np.random.default_rng(seed)
    observed = float(metric(y_true, candidate) - metric(y_true, baseline))
    draws = np.empty(iterations, dtype=float)
    for draw in range(iterations):
        idx = moving_block_indices(len(y_true), length, rng)
        draws[draw] = metric(y_true[idx], candidate[idx]) - metric(y_true[idx], baseline[idx])
    alpha = (1.0 - confidence_level) / 2.0
    two_sided_p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    two_sided_p = float(np.clip(two_sided_p, 0.0, 1.0))
    return {
        "n": len(y_true),
        "block_length": length,
        "observed_difference": observed,
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
        "bootstrap_two_sided_p": two_sided_p,
    }
