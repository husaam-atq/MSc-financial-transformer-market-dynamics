"""Forward-looking target generation with explicit forecast timestamp semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetRelativeStressDefinition:
    """Training-fitted thresholds for an asset-relative downside label."""

    horizon: int
    quantile: float
    thresholds: pd.Series
    price_sources: pd.Series
    target_column: str


def add_targets(featured: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add classification and volatility targets independently for each asset.

    Features at timestamp ``t`` are interpreted as known by close ``t``. Targets
    use only future data from ``t+1`` through ``t+h``.
    """
    if "Ticker" not in featured.columns:
        raise ValueError("Input dataframe must include a Ticker column")

    target_config = (config or {}).get("targets", {})
    direction_horizons = target_config.get("direction_horizons", [1, 5, 10])
    volatility_horizons = target_config.get("volatility_horizons", [5, 10])
    stress_config = target_config.get("stress", {})

    frames = []
    for _, asset_frame in featured.groupby("Ticker", sort=False):
        asset = asset_frame.sort_index().copy()
        observed = asset[asset["Close"].notna()].copy()
        targeted = _add_asset_targets(
            observed,
            direction_horizons=direction_horizons,
            volatility_horizons=volatility_horizons,
            stress_config=stress_config,
        )
        target_columns = [
            column
            for column in targeted.columns
            if column.startswith("target_")
        ]
        for column in target_columns:
            asset[column] = targeted[column].reindex(asset.index)
        frames.append(asset)
    result = (
        pd.concat(frames, axis=0)
        .reset_index()
        .sort_values(["Ticker", "Date"])
        .set_index("Date")
    )
    result.index.name = "Date"
    return result


def classification_target_columns(horizons: list[int] | tuple[int, ...] = (1, 5, 10)) -> list[str]:
    return [f"target_direction_{h}d" for h in horizons]


def volatility_target_columns(horizons: list[int] | tuple[int, ...] = (5, 10)) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        columns.append(f"target_realized_vol_{horizon}d")
        columns.append(f"target_log_realized_vol_{horizon}d")
    return columns


def add_future_maximum_loss_target(
    panel: pd.DataFrame,
    *,
    horizon: int = 10,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Add per-asset maximum origin-to-path total-return loss.

    At forecast origin ``t`` the target is the largest non-negative loss from
    ``P_t`` to any observed session in ``t+1`` through ``t+horizon``. Adjusted
    close is used only when it is complete and positive for the full observed
    asset history; otherwise the full asset series uses raw close. The final
    ``horizon`` observed rows for each asset remain missing.
    """
    if "Ticker" not in panel or "Close" not in panel:
        raise ValueError("Maximum-loss target requires Ticker and Close columns")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    name = target_column or f"target_future_maximum_loss_{horizon}d"
    frames: list[pd.DataFrame] = []
    for ticker, raw_asset in panel.groupby("Ticker", observed=True, sort=False):
        asset = raw_asset.sort_index().copy()
        observed = asset[asset["Close"].notna()].copy()
        adjusted = observed.get("Adj Close")
        if adjusted is not None and adjusted.notna().all() and adjusted.gt(0.0).all():
            price = adjusted.astype(float)
        else:
            close = pd.to_numeric(observed["Close"], errors="coerce")
            if not close.notna().all() or not close.gt(0.0).all():
                raise ValueError(f"Asset {ticker} has no complete positive target price series")
            price = close.astype(float)
        future_minimum = (
            price.shift(-1)
            .iloc[::-1]
            .rolling(horizon, min_periods=horizon)
            .min()
            .iloc[::-1]
        )
        maximum_loss = (1.0 - future_minimum / price).clip(lower=0.0)
        asset[name] = maximum_loss.reindex(asset.index)
        frames.append(asset)
    output = pd.concat(frames).reset_index().sort_values(["Ticker", "Date"]).set_index("Date")
    output.index.name = "Date"
    return output


def fit_asset_relative_stress_target(
    panel: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    *,
    horizon: int = 10,
    quantile: float = 0.90,
    target_column: str = "target_asset_relative_max_loss_10d_q90",
) -> tuple[pd.DataFrame, AssetRelativeStressDefinition]:
    """Fit per-asset downside thresholds on training dates and label all rows.

    At close ``t``, maximum loss is the largest adjusted-close decline from
    ``P_t`` over observed sessions ``t+1`` through ``t+h``. Thresholds use only
    rows whose forecast origins are in ``train_dates``. Callers must supply a
    purge-safe training segment; validation and test values never affect fit.
    """
    if "Ticker" not in panel or "Close" not in panel:
        raise ValueError("Asset-relative target requires Ticker and Close columns")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    train_index = pd.DatetimeIndex(train_dates)
    frames: list[pd.DataFrame] = []
    threshold_rows: dict[str, float] = {}
    price_source_rows: dict[str, str] = {}
    for ticker, raw_asset in panel.groupby("Ticker", observed=True, sort=True):
        asset = raw_asset.sort_index().copy()
        observed = asset[asset["Close"].notna()].copy()
        adjusted = observed.get("Adj Close")
        training_mask = observed.index.isin(train_index)
        use_adjusted = bool(
            adjusted is not None
            and training_mask.any()
            and adjusted.loc[training_mask].notna().all()
        )
        price = adjusted.astype(float) if use_adjusted else observed["Close"].astype(float)
        future_minimum = price.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]
        maximum_loss = (1.0 - future_minimum / price).clip(lower=0.0)
        training_values = maximum_loss[maximum_loss.index.isin(train_index)].dropna()
        threshold = float(training_values.quantile(quantile)) if not training_values.empty else np.nan
        if np.isfinite(threshold):
            threshold_rows[str(ticker)] = threshold
        price_source_rows[str(ticker)] = "Adj Close" if use_adjusted else "Close"
        labels = (maximum_loss > threshold).where(maximum_loss.notna() & np.isfinite(threshold), np.nan).astype(float)
        asset["target_future_maximum_loss_10d"] = maximum_loss.reindex(asset.index)
        asset[target_column] = labels.reindex(asset.index)
        frames.append(asset)
    output = pd.concat(frames).reset_index().sort_values(["Ticker", "Date"]).set_index("Date")
    output.index.name = "Date"
    definition = AssetRelativeStressDefinition(
        horizon=int(horizon),
        quantile=float(quantile),
        thresholds=pd.Series(threshold_rows, name="maximum_loss_threshold", dtype=float).sort_index(),
        price_sources=pd.Series(price_source_rows, name="price_source", dtype="string").sort_index(),
        target_column=target_column,
    )
    return output, definition


def _add_asset_targets(
    frame: pd.DataFrame,
    direction_horizons: list[int],
    volatility_horizons: list[int],
    stress_config: dict[str, Any],
) -> pd.DataFrame:
    price = _target_price(frame)
    log_return = np.log(price).diff()

    for horizon in direction_horizons:
        future_return = price.shift(-horizon) / price - 1.0
        frame[f"target_forward_return_{horizon}d"] = future_return
        frame[f"target_direction_{horizon}d"] = np.where(
            future_return.notna(),
            (future_return > 0.0).astype(float),
            np.nan,
        )

    for horizon in volatility_horizons:
        realized_vol = _future_realized_volatility(log_return, horizon)
        frame[f"target_realized_vol_{horizon}d"] = realized_vol
        frame[f"target_log_realized_vol_{horizon}d"] = np.log(realized_vol + 1e-12)

    if stress_config.get("enabled", False):
        horizon = int(stress_config.get("horizon", 10))
        large_negative = float(stress_config.get("large_negative_return", -0.05))
        vol_multiplier = float(stress_config.get("volatility_spike_multiplier", 2.0))
        drawdown_threshold = float(stress_config.get("drawdown_threshold", -0.07))
        frame[f"target_stress_{horizon}d"] = _future_stress_target(
            price=price,
            log_return=log_return,
            horizon=horizon,
            large_negative_return=large_negative,
            volatility_spike_multiplier=vol_multiplier,
            drawdown_threshold=drawdown_threshold,
        )

    return frame


def _target_price(frame: pd.DataFrame) -> pd.Series:
    adjusted = frame.get("Adj Close")
    if adjusted is not None and adjusted.notna().any():
        return adjusted.astype(float).where(adjusted.notna(), frame["Close"].astype(float))
    return frame["Close"].astype(float)


def _future_realized_volatility(log_return: pd.Series, horizon: int) -> pd.Series:
    future_returns = log_return.shift(-1)
    future_squared_sum = (
        future_returns.pow(2.0).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    )
    return np.sqrt(future_squared_sum)


def _future_stress_target(
    price: pd.Series,
    log_return: pd.Series,
    horizon: int,
    large_negative_return: float,
    volatility_spike_multiplier: float,
    drawdown_threshold: float,
) -> pd.Series:
    future_return = price.shift(-horizon) / price - 1.0
    future_min_price = price.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]
    future_drawdown = future_min_price / price - 1.0
    future_vol = _future_realized_volatility(log_return, horizon)
    past_vol = log_return.rolling(20, min_periods=20).std() * np.sqrt(horizon)
    stress = (
        (future_return <= large_negative_return)
        | (future_drawdown <= drawdown_threshold)
        | (future_vol >= volatility_spike_multiplier * past_vol)
    )
    return stress.where(future_return.notna() & future_vol.notna(), np.nan).astype(float)
