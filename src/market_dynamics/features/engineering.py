"""Leakage-safe technical feature engineering for daily OHLCV data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

PRICE_RETURN_FEATURES = [
    "return_close",
    "log_return",
    "open_close_return",
    "high_low_range",
    "overnight_gap",
    "intraday_range_proxy",
]

VOLATILITY_FEATURES = [
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "downside_volatility_20d",
    "rolling_high_low_range_20d",
    "ewma_volatility_20d",
]

MOMENTUM_TREND_FEATURES = [
    "cum_return_5d",
    "cum_return_20d",
    "ma_distance_20d",
    "ma_crossover_5_20d",
    "rsi_14d",
    "macd_12_26",
]

VOLUME_ACTIVITY_FEATURES = [
    "volume_change",
    "volume_zscore_20d",
    "volume_ma_ratio_20d",
    "price_volume_interaction",
]

STRESS_DRAWDOWN_FEATURES = [
    "rolling_drawdown_60d",
    "distance_from_20d_high",
    "distance_from_60d_high",
    "negative_return_streak",
    "downside_move_indicator",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "price_return": PRICE_RETURN_FEATURES,
    "volatility": VOLATILITY_FEATURES,
    "momentum_trend": MOMENTUM_TREND_FEATURES,
    "volume_activity": VOLUME_ACTIVITY_FEATURES,
    "stress_drawdown": STRESS_DRAWDOWN_FEATURES,
}


def all_feature_columns() -> list[str]:
    """Return all engineered feature columns in stable group order."""
    return [column for columns in FEATURE_GROUPS.values() for column in columns]


def add_features(ohlcv: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add leakage-safe features independently for each asset.

    All rolling, EWMA and lagged calculations use observations available at or before
    timestamp ``t``. No feature uses future rows.
    """
    if "Ticker" not in ohlcv.columns:
        raise ValueError("Input OHLCV dataframe must include a Ticker column")

    prefer_adjusted = True
    if config is not None:
        prefer_adjusted = bool(config.get("features", {}).get("prefer_adjusted_close", True))

    frames = []
    for _, asset_frame in ohlcv.groupby("Ticker", sort=False):
        asset = asset_frame.sort_index().copy()
        observed = asset[asset["Close"].notna()].copy()
        engineered = _add_asset_features(observed, prefer_adjusted)
        for column in all_feature_columns():
            asset[column] = engineered[column].reindex(asset.index)
        frames.append(asset)
    result = (
        pd.concat(frames, axis=0)
        .reset_index()
        .sort_values(["Ticker", "Date"])
        .set_index("Date")
    )
    result.index.name = "Date"
    return result


def _add_asset_features(frame: pd.DataFrame, prefer_adjusted: bool) -> pd.DataFrame:
    price = _select_price(frame, prefer_adjusted)
    close = frame["Close"].astype(float)
    open_ = frame["Open"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].replace(0, np.nan).astype(float)

    frame["return_close"] = price.pct_change()
    frame["log_return"] = np.log(price).diff()
    frame["open_close_return"] = close / open_ - 1.0
    frame["high_low_range"] = high / low - 1.0
    frame["overnight_gap"] = open_ / close.shift(1) - 1.0
    frame["intraday_range_proxy"] = (high - low) / close

    log_return = frame["log_return"]
    for window in [5, 10, 20]:
        frame[f"volatility_{window}d"] = log_return.rolling(window, min_periods=window).std()

    downside = log_return.where(log_return < 0.0, 0.0)
    frame["downside_volatility_20d"] = downside.rolling(20, min_periods=20).std()
    frame["rolling_high_low_range_20d"] = frame["high_low_range"].rolling(20, min_periods=20).mean()
    frame["ewma_volatility_20d"] = log_return.ewm(span=20, adjust=False, min_periods=20).std()

    frame["cum_return_5d"] = price / price.shift(5) - 1.0
    frame["cum_return_20d"] = price / price.shift(20) - 1.0
    ma_5 = price.rolling(5, min_periods=5).mean()
    ma_20 = price.rolling(20, min_periods=20).mean()
    frame["ma_distance_20d"] = price / ma_20 - 1.0
    frame["ma_crossover_5_20d"] = ma_5 / ma_20 - 1.0
    frame["rsi_14d"] = _rsi(price, window=14)
    ema_12 = price.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = price.ewm(span=26, adjust=False, min_periods=26).mean()
    frame["macd_12_26"] = (ema_12 - ema_26) / price

    frame["volume_change"] = volume.pct_change()
    volume_mean_20 = volume.rolling(20, min_periods=20).mean()
    volume_std_20 = volume.rolling(20, min_periods=20).std()
    frame["volume_zscore_20d"] = (volume - volume_mean_20) / volume_std_20
    frame["volume_ma_ratio_20d"] = volume / volume_mean_20 - 1.0
    frame["price_volume_interaction"] = frame["return_close"] * frame["volume_zscore_20d"]

    rolling_high_60 = price.rolling(60, min_periods=60).max()
    rolling_high_20 = price.rolling(20, min_periods=20).max()
    frame["rolling_drawdown_60d"] = price / rolling_high_60 - 1.0
    frame["distance_from_20d_high"] = price / rolling_high_20 - 1.0
    frame["distance_from_60d_high"] = price / rolling_high_60 - 1.0
    frame["negative_return_streak"] = _negative_streak(frame["return_close"])
    frame["downside_move_indicator"] = (
        frame["return_close"] < -frame["volatility_20d"].fillna(np.inf)
    ).astype(float)

    feature_columns = all_feature_columns()
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    return frame


def _select_price(frame: pd.DataFrame, prefer_adjusted: bool) -> pd.Series:
    adjusted = frame.get("Adj Close")
    if prefer_adjusted and adjusted is not None and adjusted.notna().any():
        return adjusted.astype(float).where(adjusted.notna(), frame["Close"].astype(float))
    return frame["Close"].astype(float)


def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).where(avg_gain.notna(), np.nan)


def _negative_streak(returns: pd.Series) -> pd.Series:
    streak: list[float] = []
    count = 0
    for value in returns:
        if pd.isna(value):
            count = 0
            streak.append(np.nan)
        elif value < 0:
            count += 1
            streak.append(float(count))
        else:
            count = 0
            streak.append(0.0)
    return pd.Series(streak, index=returns.index)


def get_feature_groups(columns: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Return feature groups, optionally restricted to an available column set."""
    if columns is None:
        return {group: values.copy() for group, values in FEATURE_GROUPS.items()}
    available = set(columns)
    return {
        group: [column for column in values if column in available]
        for group, values in FEATURE_GROUPS.items()
    }
