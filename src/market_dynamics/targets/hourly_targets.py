"""Future-only target construction for the hourly crypto panel."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_hourly_targets(featured: pd.DataFrame, horizons: tuple[int, ...] = (1, 4, 24)) -> pd.DataFrame:
    """Attach direction, realised-volatility, log-volatility and stress targets.

    At timestamp ``t`` every target consumes only hourly returns from ``t+1``
    through ``t+h``. It deliberately does not cross an asset's own history.
    """
    if "Ticker" not in featured.columns:
        raise ValueError("Hourly feature panel must include Ticker")
    frames = [_asset_targets(part.sort_index().copy(), horizons) for _, part in featured.groupby("Ticker", sort=False)]
    return pd.concat(frames).reset_index().sort_values(["Ticker", "Date"]).set_index("Date").rename_axis("Date")


def hourly_target_columns(horizons: tuple[int, ...] = (1, 4, 24)) -> list[str]:
    """Return all default hourly supervised target names."""
    columns: list[str] = []
    for horizon in horizons:
        columns.extend(
            [
                f"target_direction_{horizon}h",
                f"target_realized_vol_{horizon}h",
                f"target_log_realized_vol_{horizon}h",
            ]
        )
    return columns + ["target_stress_24h"]


def _asset_targets(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    close = frame["Close"].astype(float)
    log_return = np.log(close).diff()
    for horizon in horizons:
        forward_return = close.shift(-horizon) / close - 1.0
        frame[f"target_forward_return_{horizon}h"] = forward_return
        frame[f"target_direction_{horizon}h"] = np.where(forward_return.notna(), (forward_return > 0).astype(float), np.nan)
        realized = _future_realized_volatility(log_return, horizon)
        frame[f"target_realized_vol_{horizon}h"] = realized
        frame[f"target_log_realized_vol_{horizon}h"] = np.log(realized + 1e-12)
    future_return = close.shift(-24) / close - 1.0
    future_low = log_return.shift(-1).iloc[::-1].rolling(24, min_periods=24).min().iloc[::-1]
    future_vol = _future_realized_volatility(log_return, 24)
    past_vol = log_return.rolling(168, min_periods=168).std() * np.sqrt(24)
    frame["target_stress_24h"] = (
        (future_return <= -0.08) | (future_low <= -0.05) | (future_vol >= 2.0 * past_vol)
    ).where(future_return.notna() & future_vol.notna(), np.nan).astype(float)
    return frame


def _future_realized_volatility(log_return: pd.Series, horizon: int) -> pd.Series:
    future = log_return.shift(-1)
    squared_sum = future.pow(2).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    return np.sqrt(squared_sum)
