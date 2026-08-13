"""Bounded post-freeze evaluation helpers with explicit validation selection.

These helpers are intentionally model-agnostic.  They operate on persisted
predictions and never choose a calibration, threshold, variance scale or model
from held-out test labels.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from market_dynamics.evaluation.calibration import calibration_summary
from market_dynamics.evaluation.classification_postprocessing import (
    Calibrator,
    IdentityCalibrator,
    fit_probability_calibrator,
    validation_optimal_threshold,
)
from market_dynamics.evaluation.metrics import classification_metrics, regression_metrics


@dataclass(frozen=True)
class CalibrationCandidate:
    """A validation-fitted probability calibration and F1 threshold."""

    method: str
    calibrator: Calibrator
    threshold: float
    validation_threshold_f1: float
    validation_metrics: dict[str, float]


def binary_probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Return discrimination, decision and probability-quality diagnostics."""
    y, probability = _finite_binary_arrays(y_true, probabilities)
    predicted = (probability >= float(threshold)).astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        warnings.filterwarnings("ignore", message="A single label was found.*", category=UserWarning)
        metrics = classification_metrics(y, predicted, probability)
        metrics["pr_auc"] = float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else np.nan
    calibration, _ = calibration_summary(y, probability, bins=calibration_bins)
    metrics.update(calibration)
    metrics["log_loss"] = float(log_loss(y, np.clip(probability, 1e-6, 1.0 - 1e-6), labels=[0, 1])) if len(np.unique(y)) == 2 else np.nan
    intercept, slope = calibration_slope_intercept(y, probability)
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope"] = slope
    metrics.update(_confusion_counts(y, predicted))
    metrics["degenerate_prediction"] = bool(
        metrics["prediction_positive_rate"] <= 0.05
        or metrics["prediction_positive_rate"] >= 0.95
        or metrics["prediction_unique_values"] < 2
    )
    return {key: float(value) if isinstance(value, np.floating) else value for key, value in metrics.items()}


def calibration_slope_intercept(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Fit logistic calibration intercept and slope on a fixed evaluation set."""
    y, probability = _finite_binary_arrays(y_true, probabilities)
    if len(np.unique(y)) < 2 or np.nanstd(probability) <= 1e-12:
        return (np.nan, np.nan)
    model = LogisticRegression(max_iter=1000).fit(logit(np.clip(probability, 1e-6, 1.0 - 1e-6)).reshape(-1, 1), y)
    return (float(model.intercept_[0]), float(model.coef_[0, 0]))


def fit_calibration_candidates(
    y_validation: np.ndarray,
    raw_validation_probability: np.ndarray,
    methods: Iterable[str],
    threshold_metric: str = "f1",
    threshold_grid_size: int = 101,
    calibration_bins: int = 10,
) -> list[CalibrationCandidate]:
    """Fit and rank only validation-side calibration candidates.

    Ranking prioritises Brier score, then log loss and ECE.  This deliberately
    ranks proper probability scores rather than the validation F1 threshold.
    """
    y, raw = _finite_binary_arrays(y_validation, raw_validation_probability)
    candidates: list[CalibrationCandidate] = []
    seen: set[str] = set()
    for raw_method in methods:
        method = str(raw_method).lower()
        if method in seen:
            continue
        seen.add(method)
        calibrator: Calibrator = IdentityCalibrator() if method in {"raw", "none", "identity"} else fit_probability_calibrator(y, raw, method)
        probability = calibrator.predict(raw)
        threshold, threshold_score = validation_optimal_threshold(
            y,
            probability,
            metric=threshold_metric,
            grid_size=threshold_grid_size,
        )
        candidates.append(
            CalibrationCandidate(
                method="raw" if method in {"none", "identity"} else method,
                calibrator=calibrator,
                threshold=float(threshold),
                validation_threshold_f1=float(threshold_score),
                validation_metrics=binary_probability_metrics(y, probability, threshold, calibration_bins),
            )
        )
    if not candidates:
        raise ValueError("At least one calibration method is required")
    return sorted(
        candidates,
        key=lambda item: (
            _ascending_nan_last(item.validation_metrics.get("brier_score")),
            _ascending_nan_last(item.validation_metrics.get("log_loss")),
            _ascending_nan_last(item.validation_metrics.get("expected_calibration_error")),
            item.method,
        ),
    )


def aligned_probability_ensemble(
    frames: list[pd.DataFrame],
    probability_column: str,
    key_columns: tuple[str, ...] = ("split", "Date", "source_index", "asset_id"),
) -> pd.DataFrame:
    """Strictly align same-model seed predictions and return their mean probability."""
    if not frames:
        raise ValueError("At least one seed prediction frame is required")
    required = {*key_columns, "y_true", probability_column}
    normalized: list[pd.DataFrame] = []
    for seed_index, frame in enumerate(frames):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Seed frame {seed_index} missing columns: {sorted(missing)}")
        selected = frame[list(key_columns) + ["y_true", probability_column]].copy()
        if selected.duplicated(list(key_columns)).any():
            raise ValueError(f"Seed frame {seed_index} has duplicate alignment keys")
        selected = selected.rename(columns={probability_column: f"probability_seed_{seed_index}"})
        normalized.append(selected.sort_values(list(key_columns)).reset_index(drop=True))
    result = normalized[0]
    for frame in normalized[1:]:
        compare = frame.drop(columns=["y_true"])
        result = result.merge(compare, on=list(key_columns), how="outer", validate="one_to_one", indicator=True)
        if not result["_merge"].eq("both").all():
            raise ValueError("Seed prediction frames do not have identical endpoint membership")
        result = result.drop(columns=["_merge"])
        y_other = frame["y_true"].to_numpy(dtype=float)
        y_base = result["y_true"].to_numpy(dtype=float)
        if len(y_other) != len(y_base) or not np.allclose(y_base, y_other, equal_nan=True):
            raise ValueError("Seed prediction frames disagree on held-out labels")
    columns = [column for column in result.columns if column.startswith("probability_seed_")]
    result["ensemble_probability"] = result[columns].mean(axis=1)
    result["ensemble_seed_count"] = len(columns)
    return result


def gaussian_ensemble(
    frames: list[pd.DataFrame],
    mean_column: str = "prediction_mean",
    variance_column: str = "prediction_variance",
    key_columns: tuple[str, ...] = ("split", "Date", "source_index", "asset_id"),
) -> pd.DataFrame:
    """Align Gaussian seed forecasts and apply the mixture variance identity."""
    if not frames:
        raise ValueError("At least one seed prediction frame is required")
    required = {*key_columns, "y_true", mean_column, variance_column}
    normalized: list[pd.DataFrame] = []
    for seed_index, frame in enumerate(frames):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Seed frame {seed_index} missing columns: {sorted(missing)}")
        selected = frame[list(key_columns) + ["y_true", mean_column, variance_column]].copy()
        if selected.duplicated(list(key_columns)).any():
            raise ValueError(f"Seed frame {seed_index} has duplicate alignment keys")
        selected = selected.rename(
            columns={mean_column: f"mean_seed_{seed_index}", variance_column: f"variance_seed_{seed_index}"}
        )
        normalized.append(selected.sort_values(list(key_columns)).reset_index(drop=True))
    result = normalized[0]
    for frame in normalized[1:]:
        compare = frame.drop(columns=["y_true"])
        result = result.merge(compare, on=list(key_columns), how="outer", validate="one_to_one", indicator=True)
        if not result["_merge"].eq("both").all():
            raise ValueError("Seed Gaussian forecasts do not have identical endpoint membership")
        result = result.drop(columns=["_merge"])
        if not np.allclose(result["y_true"].to_numpy(dtype=float), frame["y_true"].to_numpy(dtype=float), equal_nan=True):
            raise ValueError("Seed Gaussian forecasts disagree on held-out labels")
    means = result[[column for column in result if column.startswith("mean_seed_")]].to_numpy(dtype=float)
    variances = result[[column for column in result if column.startswith("variance_seed_")]].to_numpy(dtype=float)
    result["ensemble_mean"] = np.mean(means, axis=1)
    result["ensemble_variance"] = np.maximum(
        np.mean(variances + means**2, axis=1) - result["ensemble_mean"].to_numpy(dtype=float) ** 2,
        1e-12,
    )
    result["ensemble_seed_count"] = means.shape[1]
    return result


def fit_variance_scale(y_validation: np.ndarray, mean_validation: np.ndarray, variance_validation: np.ndarray) -> float:
    """Fit a scalar multiplicative variance recalibration from validation only."""
    y, mean, variance = _finite_regression_arrays(y_validation, mean_validation, variance_validation)
    scale = float(np.mean((y - mean) ** 2 / variance))
    return float(np.clip(scale, 1e-4, 1e4))


def gaussian_uncertainty_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    coverage_levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, float]:
    """Return point and interval-quality diagnostics for Gaussian forecasts."""
    y, mu, var = _finite_regression_arrays(y_true, mean, variance)
    metrics = regression_metrics(y, mu)
    error = (y - mu) ** 2
    standard_deviation = np.sqrt(var)
    metrics["spearman_corr"] = _safe_correlation(y, mu, spearmanr)
    metrics["gaussian_nll"] = float(np.mean(0.5 * (np.log(var) + error / var)))
    metrics["variance_error_pearson"] = _safe_correlation(var, error, pearsonr)
    metrics["variance_error_spearman"] = _safe_correlation(var, error, spearmanr)
    for level in coverage_levels:
        z = float(norm.ppf((1.0 + float(level)) / 2.0))
        covered = (y >= mu - z * standard_deviation) & (y <= mu + z * standard_deviation)
        label = f"interval_{int(round(float(level) * 100))}_"
        metrics[f"{label}coverage"] = float(np.mean(covered))
        metrics[f"{label}width"] = float(np.mean(2.0 * z * standard_deviation))
    return {key: float(value) if isinstance(value, np.floating) else value for key, value in metrics.items()}


def asset_block_bootstrap_ci(
    frame: pd.DataFrame,
    value_columns: tuple[str, ...],
    metric: Callable[..., float],
    asset_column: str = "asset_id",
    date_column: str = "Date",
    iterations: int = 500,
    block_size: int = 12,
    seed: int = 42,
) -> dict[str, float]:
    """Return an asset-stratified moving-block bootstrap confidence interval."""
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty frame")
    required = {asset_column, date_column, *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Bootstrap frame missing columns: {sorted(missing)}")
    ordered = frame.sort_values([asset_column, date_column]).reset_index(drop=True)
    values = [ordered[column].to_numpy() for column in value_columns]
    point = float(metric(*values))
    groups = [group.index.to_numpy(dtype=int) for _, group in ordered.groupby(asset_column, observed=True)]
    if not groups or any(len(group) == 0 for group in groups):
        raise ValueError("Bootstrap requires at least one observation per asset")
    rng = np.random.default_rng(seed)
    draws = np.empty(int(iterations), dtype=float)
    for draw in range(int(iterations)):
        selected = []
        for positions in groups:
            starts = rng.integers(0, len(positions), size=int(np.ceil(len(positions) / block_size)))
            blocks = [np.take(positions, np.arange(start, start + block_size) % len(positions)) for start in starts]
            selected.append(np.concatenate(blocks)[: len(positions)])
        indices = np.concatenate(selected)
        draws[draw] = float(metric(*(value[indices] for value in values)))
    return {
        "estimate": point,
        "ci_lower": float(np.nanquantile(draws, 0.025)),
        "ci_upper": float(np.nanquantile(draws, 0.975)),
        "iterations": int(iterations),
        "block_size": int(block_size),
        "asset_count": int(len(groups)),
    }


def asset_block_bootstrap_difference(
    frame: pd.DataFrame,
    value_columns: tuple[str, ...],
    metric_difference: Callable[..., float],
    asset_column: str = "asset_id",
    date_column: str = "Date",
    iterations: int = 500,
    block_size: int = 12,
    seed: int = 42,
) -> dict[str, float]:
    """Return an asset-stratified moving-block interval and bounded two-sided p-value."""
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty frame")
    required = {asset_column, date_column, *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Bootstrap frame missing columns: {sorted(missing)}")
    ordered = frame.sort_values([asset_column, date_column]).reset_index(drop=True)
    values = [ordered[column].to_numpy() for column in value_columns]
    observed = float(metric_difference(*values))
    groups = [group.index.to_numpy(dtype=int) for _, group in ordered.groupby(asset_column, observed=True)]
    rng = np.random.default_rng(seed)
    draws = np.empty(int(iterations), dtype=float)
    for draw in range(int(iterations)):
        selected = _asset_block_indices(groups, int(block_size), rng)
        draws[draw] = float(metric_difference(*(value[selected] for value in values)))
    # A finite-resample correction avoids reporting a misleading literal zero
    # p-value when no bootstrap draw crosses zero.
    lower_tail = (int(np.sum(draws <= 0.0)) + 1) / (len(draws) + 1)
    upper_tail = (int(np.sum(draws >= 0.0)) + 1) / (len(draws) + 1)
    p_value = float(np.clip(2.0 * min(lower_tail, upper_tail), 0.0, 1.0))
    return {
        "observed_difference": observed,
        "ci_lower": float(np.nanquantile(draws, 0.025)),
        "ci_upper": float(np.nanquantile(draws, 0.975)),
        "bootstrap_two_sided_p": p_value,
        "iterations": int(iterations),
        "block_size": int(block_size),
        "asset_count": int(len(groups)),
    }


def _finite_binary_arrays(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y) & np.isfinite(probability)
    if not np.any(valid):
        raise ValueError("No finite binary labels and probabilities")
    return y[valid].astype(int), np.clip(probability[valid], 1e-6, 1.0 - 1e-6)


def _finite_regression_arrays(y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mean, dtype=float)
    var = np.asarray(variance, dtype=float)
    valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(var) & (var > 0.0)
    if not np.any(valid):
        raise ValueError("No finite regression targets, means and positive variances")
    return y[valid], mu[valid], np.maximum(var[valid], 1e-12)


def _confusion_counts(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    return {
        "true_negative": float(np.sum((y == 0) & (pred == 0))),
        "false_positive": float(np.sum((y == 0) & (pred == 1))),
        "false_negative": float(np.sum((y == 1) & (pred == 0))),
        "true_positive": float(np.sum((y == 1) & (pred == 1))),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray, statistic: Callable[..., object]) -> float:
    if len(left) < 3 or np.nanstd(left) <= 1e-15 or np.nanstd(right) <= 1e-15:
        return np.nan
    value = statistic(left, right)
    return float(getattr(value, "statistic", value[0]))


def _ascending_nan_last(value: float | None) -> float:
    return float(value) if value is not None and np.isfinite(value) else float("inf")


def _asset_block_indices(groups: list[np.ndarray], block_size: int, rng: np.random.Generator) -> np.ndarray:
    if not groups or any(len(group) == 0 for group in groups):
        raise ValueError("Bootstrap requires at least one observation per asset")
    selected: list[np.ndarray] = []
    for positions in groups:
        starts = rng.integers(0, len(positions), size=int(np.ceil(len(positions) / block_size)))
        blocks = [np.take(positions, np.arange(start, start + block_size) % len(positions)) for start in starts]
        selected.append(np.concatenate(blocks)[: len(positions)])
    return np.concatenate(selected)
