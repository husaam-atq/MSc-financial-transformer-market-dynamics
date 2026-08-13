"""Volatility-specific Phase 1 baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_dynamics.models.baselines import ModelPrediction

LOGGER = logging.getLogger(__name__)

HAR_FEATURES = ["volatility_5d", "volatility_10d", "volatility_20d"]


@dataclass(frozen=True)
class VolatilityTargetSpec:
    """Metadata parsed from a volatility target column."""

    horizon: int
    is_log: bool
    frequency: str


def parse_volatility_target(target_column: str) -> VolatilityTargetSpec | None:
    """Parse supported realised-volatility target names."""
    is_log = target_column.startswith("target_log_realized_vol_")
    prefix = "target_log_realized_vol_" if is_log else "target_realized_vol_"
    if not target_column.startswith(prefix):
        return None
    suffix = target_column.removeprefix(prefix)
    if not suffix or suffix[-1] not in {"d", "h"}:
        return None
    horizon = int(suffix[:-1])
    return VolatilityTargetSpec(horizon=horizon, is_log=is_log, frequency=suffix[-1])


def volatility_baseline_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    target_column: str,
) -> list[ModelPrediction]:
    """Return EWMA and HAR-RV style volatility forecasts."""
    spec = parse_volatility_target(target_column)
    if spec is None:
        return []

    predictions: list[ModelPrediction] = []
    predictions.append(_ewma_vol_prediction(X_eval, spec))

    har_candidates = HAR_FEATURES if spec.frequency == "d" else [
        "hourly_realized_volatility_24h",
        "hourly_realized_volatility_168h",
        "hourly_range_volatility_24h",
    ]
    available_har = [column for column in har_candidates if column in X_train.columns]
    if available_har:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        try:
            pipeline.fit(X_train[available_har], y_train)
            predictions.append(
                ModelPrediction(
                    "har_rv_ridge",
                    np.asarray(pipeline.predict(X_eval[available_har])),
                )
            )
        except Exception as exc:
            LOGGER.warning("Skipping HAR-RV baseline after failure: %s", exc)
    return predictions


def garch_predictions(
    asset_frame: pd.DataFrame,
    eval_idx: np.ndarray,
    target_column: str,
    config: dict[str, Any],
) -> ModelPrediction | None:
    """Optional rolling GARCH(1,1) volatility baseline using only past returns.

    This is intentionally off by default in the base config because repeated
    GARCH refits are slow. Enable ``models.garch.enabled`` when the ``arch``
    package is installed and runtime is acceptable.
    """
    garch_config = config.get("models", {}).get("garch", {})
    if not garch_config.get("enabled", False):
        return None

    spec = parse_volatility_target(target_column)
    if spec is None:
        return None

    try:
        from arch import arch_model
    except ImportError:
        LOGGER.warning("Optional dependency arch is not installed; skipping GARCH baseline")
        return None

    return_column = _garch_return_column(asset_frame, spec)
    if return_column is None:
        LOGGER.warning("Skipping GARCH baseline because no observed return column is available for %s", target_column)
        return None

    returns = asset_frame[return_column].astype(float) * 100.0
    refit_every = int(garch_config.get("refit_frequency", garch_config.get("refit_every", 63)))
    max_train_size = int(garch_config.get("max_train_size", 1500))
    y_pred: list[float] = []
    cached_prediction: float | None = None

    for position, idx in enumerate(eval_idx):
        if cached_prediction is None or position % refit_every == 0:
            history = returns.iloc[:idx].dropna()
            if len(history) > max_train_size:
                history = history.iloc[-max_train_size:]
            if len(history) < 250:
                cached_prediction = np.nan
            else:
                try:
                    model = arch_model(history, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
                    result = model.fit(disp="off", show_warning=False)
                    forecast = result.forecast(horizon=spec.horizon, reindex=False)
                    variances = forecast.variance.iloc[-1].to_numpy(dtype=float)
                    cached_prediction = float(np.sqrt(np.nansum(variances)) / 100.0)
                except Exception as exc:
                    LOGGER.warning("GARCH fit failed at index %s: %s", idx, exc)
                    cached_prediction = np.nan
        y_pred.append(cached_prediction if cached_prediction is not None else np.nan)

    predictions = np.asarray(y_pred, dtype=float)
    if spec.is_log:
        predictions = np.log(predictions + 1e-12)
    return ModelPrediction("garch_1_1", predictions)


def _garch_return_column(asset_frame: pd.DataFrame, spec: VolatilityTargetSpec) -> str | None:
    """Choose the observed return series matching the target frequency."""
    preferred = ["log_return", "close_to_close_return"] if spec.frequency == "d" else ["hourly_log_return", "hourly_return"]
    for column in preferred:
        if column in asset_frame.columns:
            return column
    fallback = ["log_return", "hourly_log_return", "close_to_close_return", "hourly_return"]
    for column in fallback:
        if column in asset_frame.columns:
            return column
    return None


def _ewma_vol_prediction(X_eval: pd.DataFrame, spec: VolatilityTargetSpec) -> ModelPrediction:
    column = "ewma_volatility_20d" if spec.frequency == "d" else "hourly_ewma_volatility_24h"
    if column not in X_eval.columns:
        return ModelPrediction("ewma_volatility", np.full(len(X_eval), np.nan))
    pred = X_eval[column].astype(float).to_numpy() * np.sqrt(spec.horizon)
    if spec.is_log:
        pred = np.log(pred + 1e-12)
    return ModelPrediction("ewma_volatility", pred)
