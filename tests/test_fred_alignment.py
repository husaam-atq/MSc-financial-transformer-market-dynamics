from __future__ import annotations

import pandas as pd

from market_dynamics.features.macro_alignment import (
    lag_fred_to_market_availability,
    merge_macro_asof,
)


def test_fred_alignment_requires_one_business_day_availability_lag() -> None:
    macro = pd.DataFrame({"DFF": [5.0]}, index=pd.DatetimeIndex(["2024-01-05"], name="Date"))
    market = pd.DataFrame({"Ticker": ["SPY", "SPY"]}, index=pd.DatetimeIndex(["2024-01-05", "2024-01-08"], name="Date"))

    merged = merge_macro_asof(market, lag_fred_to_market_availability(macro, market_day_lag=1))

    assert pd.isna(merged.loc[pd.Timestamp("2024-01-05"), "DFF"])
    assert merged.loc[pd.Timestamp("2024-01-08"), "DFF"] == 5.0
    assert merged.loc[pd.Timestamp("2024-01-08"), "available_date"] <= pd.Timestamp("2024-01-08")


def test_fred_alignment_carries_each_series_independently() -> None:
    macro = pd.DataFrame(
        {
            "A": [1.0, float("nan"), 2.0],
            "B": [10.0, 11.0, float("nan")],
        },
        index=pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08"], name="Date"),
    )

    available = lag_fred_to_market_availability(macro, market_day_lag=1)

    assert available.loc[pd.Timestamp("2024-01-08"), "A"] == 1.0
    assert available.loc[pd.Timestamp("2024-01-08"), "B"] == 11.0
    assert available.loc[pd.Timestamp("2024-01-09"), "A"] == 2.0
    assert available.loc[pd.Timestamp("2024-01-09"), "B"] == 11.0
