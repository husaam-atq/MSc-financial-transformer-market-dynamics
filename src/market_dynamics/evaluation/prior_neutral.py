"""Prior-neutral and event-level diagnostics for pooled binary forecasts."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def fit_group_priors(
    frame: pd.DataFrame,
    group_column: str,
    target_column: str = "y_true",
    smoothing: float = 1.0,
) -> pd.Series:
    """Estimate smoothed group prevalences from training observations only."""
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    grouped = frame.groupby(group_column, observed=True)[target_column].agg(["sum", "count"])
    global_rate = float(frame[target_column].mean())
    return (grouped["sum"] + smoothing * global_rate) / (grouped["count"] + smoothing)


def centre_logits_within_group(
    frame: pd.DataFrame,
    probability_column: str,
    group_column: str,
    reference: pd.DataFrame,
) -> np.ndarray:
    """Remove each group's reference-period mean logit from forecast logits."""
    ref = reference[[group_column, probability_column]].copy()
    ref["_logit"] = logit(np.clip(ref[probability_column].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6))
    centres = ref.groupby(group_column, observed=True)["_logit"].mean()
    score = logit(np.clip(frame[probability_column].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6))
    group_centre = frame[group_column].map(centres)
    fallback = float(ref["_logit"].mean())
    return expit(score - group_centre.fillna(fallback).to_numpy(dtype=float))


def remove_prior_logit(
    probabilities: np.ndarray,
    groups: pd.Series,
    priors: pd.Series,
) -> np.ndarray:
    """Return dynamic residual probabilities after subtracting a static prior logit."""
    score = logit(np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6))
    mapped = groups.map(priors)
    fallback = float(np.nanmean(priors.to_numpy(dtype=float)))
    prior = np.clip(mapped.fillna(fallback).to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    return expit(score - logit(prior))


def grouped_binary_metrics(
    frame: pd.DataFrame,
    probability_column: str,
    group_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Compute one metric row per group without hiding one-class groups."""
    rows: list[dict[str, object]] = []
    for group, part in frame.groupby(group_column, observed=True):
        y = part["y_true"].to_numpy(dtype=int)
        p = part[probability_column].to_numpy(dtype=float)
        pred = (p >= threshold).astype(int)
        rows.append(
            {
                group_column: group,
                "n_obs": len(part),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "balanced_accuracy": _balanced_accuracy(y, pred),
                "roc_auc": _ranking_metric(y, p, roc_auc_score),
                "pr_auc": _ranking_metric(y, p, average_precision_score),
                "prediction_positive_rate": float(pred.mean()),
                "comparable_pairs": int(y.sum() * (len(y) - y.sum())),
            }
        )
    return pd.DataFrame(rows)


def equal_group_weights(frame: pd.DataFrame, group_column: str) -> np.ndarray:
    """Give every represented group equal total mass."""
    counts = frame.groupby(group_column, observed=True)[group_column].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def nonoverlapping_rows(
    frame: pd.DataFrame,
    asset_column: str = "asset_ticker",
    date_column: str = "Date",
    stride: int = 10,
    offset: int = 0,
) -> pd.DataFrame:
    """Take every fixed-stride endpoint within each asset without outcome-based selection."""
    if stride < 1:
        raise ValueError("stride must be positive")
    if offset < 0 or offset >= stride:
        raise ValueError("offset must be in [0, stride)")
    ordered = frame.sort_values([asset_column, date_column]).copy()
    keep = ordered.groupby(asset_column, observed=True).cumcount().mod(stride).eq(offset)
    return ordered.loc[keep].reset_index(drop=True)


def contiguous_positive_events(
    frame: pd.DataFrame,
    probability_column: str,
    threshold: float,
    asset_column: str = "asset_ticker",
    date_column: str = "Date",
) -> pd.DataFrame:
    """Collapse contiguous positive labels into fixed per-asset event episodes."""
    rows: list[dict[str, object]] = []
    for asset, part in frame.sort_values([asset_column, date_column]).groupby(asset_column, observed=True):
        part = part.reset_index(drop=True)
        y = part["y_true"].to_numpy(dtype=int)
        starts = np.flatnonzero((y == 1) & np.r_[True, y[:-1] == 0])
        for event_id, start in enumerate(starts):
            stop = start
            while stop + 1 < len(y) and y[stop + 1] == 1:
                stop += 1
            event = part.iloc[start : stop + 1]
            maximum = float(event[probability_column].max())
            onset_probability = float(event[probability_column].iloc[0])
            first_detection = event.loc[event[probability_column].ge(threshold), date_column]
            rows.append(
                {
                    asset_column: asset,
                    "event_id": int(event_id),
                    "event_start": event[date_column].iloc[0],
                    "event_end": event[date_column].iloc[-1],
                    "event_windows": len(event),
                    "maximum_probability": maximum,
                    "onset_probability": onset_probability,
                    "onset_detected": bool(onset_probability >= threshold),
                    "any_window_detected": bool(maximum >= threshold),
                    "first_detection_date": first_detection.iloc[0] if len(first_detection) else pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def false_alarm_episodes(
    frame: pd.DataFrame,
    probability_column: str,
    threshold: float,
    asset_column: str = "asset_ticker",
) -> int:
    """Count contiguous predicted-positive episodes containing no positive label."""
    total = 0
    for _, part in frame.sort_values([asset_column, "Date"]).groupby(asset_column, observed=True):
        predicted = part[probability_column].to_numpy(dtype=float) >= threshold
        labels = part["y_true"].to_numpy(dtype=int)
        starts = np.flatnonzero(predicted & np.r_[True, ~predicted[:-1]])
        for start in starts:
            stop = start
            while stop + 1 < len(predicted) and predicted[stop + 1]:
                stop += 1
            total += int(labels[start : stop + 1].sum() == 0)
    return total


def macro_average(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    """Average finite group metrics with equal group weight."""
    return {column: float(frame[column].dropna().mean()) for column in columns}


def _ranking_metric(y: np.ndarray, probability: np.ndarray, metric: object) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(metric(y, probability))


def _balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A single label was found.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        return float(balanced_accuracy_score(y, predicted))
