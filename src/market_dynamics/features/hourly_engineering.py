"""Leakage-safe market features specialised for continuous hourly crypto data."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURLY_FEATURE_COLUMNS = [
    "hourly_log_return",
    "hourly_return",
    "hourly_open_close_return",
    "hourly_high_low_range",
    "hourly_range_volatility_24h",
    "hourly_realized_volatility_24h",
    "hourly_realized_volatility_168h",
    "hourly_ewma_volatility_24h",
    "hourly_momentum_4h",
    "hourly_momentum_24h",
    "hourly_momentum_168h",
    "hourly_ma_distance_24h",
    "hourly_rsi_24h",
    "hourly_volume_change",
    "hourly_volume_zscore_24h",
    "hourly_volume_ma_ratio_24h",
    "hourly_price_volume_interaction",
    "hourly_drawdown_168h",
    "hourly_distance_from_24h_high",
    "hourly_negative_return_streak",
    "hour_of_day",
    "day_of_week",
]


def add_hourly_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Create features known no later than the close of each hourly candle."""
    if "Ticker" not in ohlcv.columns:
        raise ValueError("Hourly OHLCV must include a Ticker column")
    frames = [_add_asset_features(part.sort_index().copy()) for _, part in ohlcv.groupby("Ticker", sort=False)]
    return (
        pd.concat(frames).reset_index().sort_values(["Ticker", "Date"]).set_index("Date").rename_axis("Date")
    )


def _add_asset_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["Close"].astype(float)
    open_ = frame["Open"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].replace(0, np.nan).astype(float)
    log_return = np.log(close).diff()
    frame["hourly_log_return"] = log_return
    frame["hourly_return"] = close.pct_change()
    frame["hourly_open_close_return"] = close / open_ - 1.0
    frame["hourly_high_low_range"] = high / low - 1.0
    frame["hourly_range_volatility_24h"] = frame["hourly_high_low_range"].rolling(24, min_periods=24).mean()
    frame["hourly_realized_volatility_24h"] = log_return.rolling(24, min_periods=24).std()
    frame["hourly_realized_volatility_168h"] = log_return.rolling(168, min_periods=168).std()
    frame["hourly_ewma_volatility_24h"] = log_return.ewm(span=24, adjust=False, min_periods=24).std()
    for window in (4, 24, 168):
        frame[f"hourly_momentum_{window}h"] = close / close.shift(window) - 1.0
    ma_24 = close.rolling(24, min_periods=24).mean()
    frame["hourly_ma_distance_24h"] = close / ma_24 - 1.0
    frame["hourly_rsi_24h"] = _rsi(close, 24)
    volume_mean = volume.rolling(24, min_periods=24).mean()
    volume_std = volume.rolling(24, min_periods=24).std()
    frame["hourly_volume_change"] = volume.pct_change()
    frame["hourly_volume_zscore_24h"] = (volume - volume_mean) / volume_std
    frame["hourly_volume_ma_ratio_24h"] = volume / volume_mean - 1.0
    frame["hourly_price_volume_interaction"] = frame["hourly_return"] * frame["hourly_volume_zscore_24h"]
    rolling_high_168 = close.rolling(168, min_periods=168).max()
    rolling_high_24 = close.rolling(24, min_periods=24).max()
    frame["hourly_drawdown_168h"] = close / rolling_high_168 - 1.0
    frame["hourly_distance_from_24h_high"] = close / rolling_high_24 - 1.0
    frame["hourly_negative_return_streak"] = _negative_streak(frame["hourly_return"])
    frame["hour_of_day"] = frame.index.hour.astype(float)
    frame["day_of_week"] = frame.index.dayofweek.astype(float)
    frame[HOURLY_FEATURE_COLUMNS] = frame[HOURLY_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return frame


def _rsi(price: pd.Series, window: int) -> pd.Series:
    change = price.diff()
    gain = change.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-change.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).where(gain.notna())


def _negative_streak(returns: pd.Series) -> pd.Series:
    values: list[float] = []
    streak = 0
    for value in returns:
        if pd.isna(value):
            streak = 0
            values.append(np.nan)
        elif value < 0:
            streak += 1
            values.append(float(streak))
        else:
            streak = 0
            values.append(0.0)
    return pd.Series(values, index=returns.index)
