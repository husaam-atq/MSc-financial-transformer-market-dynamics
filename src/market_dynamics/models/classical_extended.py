"""Leakage-safe ARIMA and ARIMAX direct forecast baselines."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from market_dynamics.models.baselines import ModelPrediction

LOGGER = logging.getLogger(__name__)


def arima_direct_prediction(
    y_train: pd.Series,
    n_eval: int,
    classification: bool,
    order: tuple[int, int, int] = (1, 0, 1),
) -> ModelPrediction | None:
    """Fit ARIMA to already observed training labels and forecast the evaluation path.

    This static benchmark is intentionally simple: it never consumes validation
    or test labels while producing the test path. Direct horizon labels are used
    for consistency with the supervised task being evaluated.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA

        result = ARIMA(y_train.astype(float).dropna(), order=order).fit()
        forecast = np.asarray(result.forecast(steps=n_eval), dtype=float)
        if classification:
            probability = 1.0 / (1.0 + np.exp(-forecast))
            return ModelPrediction("arima_direct", (probability >= 0.5).astype(int), probability)
        return ModelPrediction("arima_direct", forecast)
    except Exception as exc:
        LOGGER.warning("ARIMA baseline failed: %s", exc)
        return None


def arimax_direct_prediction(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    classification: bool,
    max_exog: int = 8,
    order: tuple[int, int, int] = (1, 0, 1),
) -> ModelPrediction | None:
    """Fit direct-horizon SARIMAX with contemporaneously observed covariates only."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        selected = list(X_train.select_dtypes(include=[np.number]).columns[:max_exog])
        if not selected:
            return None
        medians = X_train[selected].median()
        train_exog = X_train[selected].replace([np.inf, -np.inf], np.nan).fillna(medians)
        eval_exog = X_eval[selected].replace([np.inf, -np.inf], np.nan).fillna(medians)
        result = SARIMAX(
            y_train.astype(float).to_numpy(),
            exog=train_exog.to_numpy(),
            order=order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = np.asarray(result.forecast(steps=len(X_eval), exog=eval_exog.to_numpy()), dtype=float)
        if classification:
            probability = 1.0 / (1.0 + np.exp(-forecast))
            return ModelPrediction("arimax_direct", (probability >= 0.5).astype(int), probability)
        return ModelPrediction("arimax_direct", forecast)
    except Exception as exc:
        LOGGER.warning("ARIMAX baseline failed: %s", exc)
        return None
