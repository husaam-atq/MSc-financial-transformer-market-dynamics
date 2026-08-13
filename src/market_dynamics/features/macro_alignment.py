"""Point-in-time-safe alignment of FRED macro observations to market dates."""

from __future__ import annotations

import pandas as pd


def lag_fred_to_market_availability(
    macro: pd.DataFrame,
    market_day_lag: int = 1,
) -> pd.DataFrame:
    """Conservatively delay all FRED values by market business days.

    FRED's observation date is not a universal release timestamp. Applying one
    business-day lag to every series is deliberately conservative and avoids
    same-close information leakage. Low-frequency observations retain their
    last published value only after this availability lag.
    """
    if not isinstance(macro.index, pd.DatetimeIndex):
        raise TypeError("FRED macro frame must use a DatetimeIndex")
    if market_day_lag < 1:
        raise ValueError("market_day_lag must be at least one for safe FRED alignment")
    columns: list[pd.Series] = []
    for column in macro.columns:
        series = macro[column].dropna().sort_index().copy()
        series.index = series.index + pd.offsets.BDay(market_day_lag)
        # Weekend/holiday observation dates can map to one business date.
        series = series.groupby(level=0).last().rename(column)
        columns.append(series)
    if not columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="available_date"))
    result = pd.concat(columns, axis=1, sort=True).sort_index().ffill()
    result.index.name = "available_date"
    return result


def merge_macro_asof(
    market: pd.DataFrame,
    macro_available: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only macro values available on or before each market timestamp."""
    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("Market panel must use a DatetimeIndex")
    if not isinstance(macro_available.index, pd.DatetimeIndex):
        raise TypeError("Macro availability frame must use a DatetimeIndex")
    left = market.reset_index().rename(columns={market.index.name or "index": "Date"})
    right = macro_available.reset_index().rename(columns={macro_available.index.name or "index": "available_date"})
    # pandas 3 can retain different datetime storage resolutions after parquet
    # round-trips. merge_asof requires exact dtype agreement, not just equal dates.
    left["Date"] = pd.DatetimeIndex(pd.to_datetime(left["Date"])).as_unit("ns")
    right["available_date"] = pd.DatetimeIndex(pd.to_datetime(right["available_date"])).as_unit("ns")
    left["_row_order"] = range(len(left))
    merged = pd.merge_asof(
        left.sort_values("Date"),
        right.sort_values("available_date"),
        left_on="Date",
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_row_order").drop(columns="_row_order").set_index("Date")
    merged.index.name = "Date"
    return merged
