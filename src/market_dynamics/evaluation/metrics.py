"""Metrics and uncertainty hooks for Phase 1 forecasting experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute binary classification metrics with class-balance diagnostics."""
    y_true_array, y_pred_array, y_score_array = _aligned_arrays(y_true, y_pred, y_score)
    metrics: dict[str, float] = {
        "n_obs": float(len(y_true_array)),
        "class_balance": float(np.mean(y_true_array)) if len(y_true_array) else np.nan,
        "prediction_positive_rate": float(np.mean(y_pred_array)) if len(y_pred_array) else np.nan,
        "prediction_unique_values": float(len(np.unique(y_pred_array))) if len(y_pred_array) else np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
    }
    if len(y_true_array) == 0:
        return metrics
    metrics["accuracy"] = float(accuracy_score(y_true_array, y_pred_array))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_array, y_pred_array))
    metrics["f1"] = float(f1_score(y_true_array, y_pred_array, zero_division=0))
    if y_score_array is not None and len(np.unique(y_true_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score_array))
    return metrics


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for volatility targets."""
    y_true_array, y_pred_array, _ = _aligned_arrays(y_true, y_pred, None)
    metrics = {
        "n_obs": float(len(y_true_array)),
        "prediction_mean": float(np.mean(y_pred_array)) if len(y_pred_array) else np.nan,
        "prediction_std": float(np.std(y_pred_array)) if len(y_pred_array) else np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "pearson_corr": np.nan,
        "r2": np.nan,
    }
    if len(y_true_array) == 0:
        return metrics
    metrics["mae"] = float(mean_absolute_error(y_true_array, y_pred_array))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    if len(y_true_array) > 2 and _is_nonconstant(y_true_array) and _is_nonconstant(y_pred_array):
        metrics["pearson_corr"] = float(pearsonr(y_true_array, y_pred_array).statistic)
    if len(y_true_array) > 1:
        metrics["r2"] = float(r2_score(y_true_array, y_pred_array))
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    block_size: int | None = None,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval hook, with optional moving-block sampling."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (np.nan, np.nan)

    estimates: list[float] = []
    for _ in range(n_resamples):
        indices = (
            _block_bootstrap_indices(n, block_size, rng)
            if block_size and block_size > 1
            else rng.integers(0, n, size=n)
        )
        estimates.append(metric_fn(y_true[indices], y_pred[indices]))
    return (float(np.nanpercentile(estimates, 2.5)), float(np.nanpercentile(estimates, 97.5)))


def metrics_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert metric dictionaries into a consistently sorted table."""
    if not rows:
        return pd.DataFrame()
    order = ["task", "asset", "target", "split", "model"]
    frame = pd.DataFrame(rows)
    existing = [column for column in order if column in frame.columns]
    remaining = [column for column in frame.columns if column not in existing]
    return frame[existing + remaining].sort_values(existing).reset_index(drop=True)


def _aligned_arrays(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    score = None
    if y_score is not None:
        score_raw = np.asarray(y_score, dtype=float)
        mask &= np.isfinite(score_raw)
        score = score_raw[mask]
    return true[mask].astype(int) if _binary_like(true[mask]) else true[mask], pred[mask], score


def _binary_like(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return len(finite) > 0 and set(np.unique(finite)).issubset({0.0, 1.0})


def _is_nonconstant(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return len(finite) > 1 and float(np.nanmax(finite) - np.nanmin(finite)) > 1e-15


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
    indices = np.concatenate([np.arange(start, start + block_size) % n for start in starts])
    return indices[:n]
