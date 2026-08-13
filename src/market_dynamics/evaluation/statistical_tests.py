"""Forecast comparison summaries with time-series aware uncertainty inputs."""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_test(loss_differential: np.ndarray, horizon: int = 1) -> dict[str, float | int]:
    """Approximate Diebold-Mariano test with a Newey-West autocovariance truncation."""
    differential = np.asarray(loss_differential, dtype=float)
    differential = differential[np.isfinite(differential)]
    if len(differential) < 10:
        raise ValueError("Diebold-Mariano test needs at least 10 finite loss differences")
    n = len(differential)
    lag = max(0, horizon - 1)
    long_run_variance = np.var(differential, ddof=1)
    for value in range(1, lag + 1):
        covariance = np.cov(differential[:-value], differential[value:], ddof=1)[0, 1]
        long_run_variance += 2.0 * (1.0 - value / (lag + 1)) * covariance
    statistic = differential.mean() / np.sqrt(max(long_run_variance, 1e-12) / n)
    return {"n": n, "dm_statistic": float(statistic), "p_value": float(2.0 * stats.t.sf(abs(statistic), df=n - 1))}


def paired_accuracy_difference(candidate_correct: np.ndarray, baseline_correct: np.ndarray) -> dict[str, float | int]:
    """McNemar-style paired classification disagreement summary."""
    candidate = np.asarray(candidate_correct, dtype=bool)
    baseline = np.asarray(baseline_correct, dtype=bool)
    if candidate.shape != baseline.shape:
        raise ValueError("Paired correctness arrays must have matching shape")
    candidate_only = int(np.sum(candidate & ~baseline))
    baseline_only = int(np.sum(~candidate & baseline))
    discordant = candidate_only + baseline_only
    statistic = ((abs(candidate_only - baseline_only) - 1.0) ** 2 / discordant) if discordant else 0.0
    return {"candidate_only_correct": candidate_only, "baseline_only_correct": baseline_only, "mcnemar_statistic": float(statistic), "p_value": float(stats.chi2.sf(statistic, 1)) if discordant else 1.0}
