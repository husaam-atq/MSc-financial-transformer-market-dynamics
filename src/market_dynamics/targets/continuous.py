"""Leakage-safe continuous future outcomes for Phase 9."""

from __future__ import annotations

import numpy as np
import pandas as pd


def future_maximum_adverse_loss(price: pd.Series, *, horizon: int) -> pd.Series:
    """Maximum origin-to-path loss over observations t+1 through t+h."""

    _validate_horizon(horizon)
    values = pd.to_numeric(price, errors="coerce")
    future_minimum = values.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]
    return (1.0 - future_minimum / values).clip(lower=0.0).rename(
        f"future_maximum_adverse_loss_{horizon}"
    )


def future_downside_realized_volatility(log_return: pd.Series, *, horizon: int) -> pd.Series:
    """Square-root sum of squared negative returns from t+1 through t+h."""

    _validate_horizon(horizon)
    downside_squared = pd.to_numeric(log_return, errors="coerce").clip(upper=0.0).pow(2.0).shift(-1)
    future_sum = downside_squared.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    return np.sqrt(future_sum).rename(f"future_downside_realized_volatility_{horizon}")


def future_realized_volatility(log_return: pd.Series, *, horizon: int) -> pd.Series:
    """Square-root sum of squared returns from t+1 through t+h."""

    _validate_horizon(horizon)
    future_squared = pd.to_numeric(log_return, errors="coerce").pow(2.0).shift(-1)
    future_sum = future_squared.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    return np.sqrt(future_sum).rename(f"future_realized_volatility_{horizon}")


def future_rolling_maximum(value: pd.Series, *, horizon: int, name: str | None = None) -> pd.Series:
    """Maximum of an observed state over t+1 through t+h."""

    _validate_horizon(horizon)
    numeric = pd.to_numeric(value, errors="coerce")
    result = numeric.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]
    return result.rename(name or f"future_maximum_{horizon}")


def future_state_change(value: pd.Series, *, horizon: int, name: str | None = None) -> pd.Series:
    """Change from state observed at t to state observed at t+h."""

    _validate_horizon(horizon)
    numeric = pd.to_numeric(value, errors="coerce")
    return (numeric.shift(-horizon) - numeric).rename(name or f"future_state_change_{horizon}")


def fit_ordinal_severity(
    continuous: pd.Series,
    train_dates: pd.DatetimeIndex,
    *,
    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90),
) -> tuple[pd.Series, tuple[float, ...]]:
    """Fit ordered-severity cut points on training origins only."""

    if tuple(sorted(quantiles)) != quantiles or any(not 0.0 < q < 1.0 for q in quantiles):
        raise ValueError("quantiles must be strictly increasing values between zero and one")
    training = pd.to_numeric(continuous.reindex(train_dates), errors="coerce").dropna()
    if training.empty:
        raise ValueError("No finite training outcomes for ordinal severity")
    cutoffs = tuple(float(training.quantile(q)) for q in quantiles)
    values = pd.to_numeric(continuous, errors="coerce")
    labels = pd.Series(np.nan, index=values.index, dtype=float, name="ordinal_tail_severity")
    finite = values.notna()
    labels.loc[finite] = np.searchsorted(np.asarray(cutoffs), values.loc[finite], side="right")
    return labels, cutoffs


def _validate_horizon(horizon: int) -> None:
    if horizon < 1:
        raise ValueError("horizon must be positive")
