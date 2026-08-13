from __future__ import annotations

import pandas as pd

from market_dynamics.data.providers.base import STANDARD_OHLCV_COLUMNS, standardise_ohlcv
from market_dynamics.data.providers.yfinance_provider import _extract_ticker_frame


def test_provider_output_schema_is_standardised() -> None:
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    frame = standardise_ohlcv(raw, ticker="TEST", provider="unit")

    assert frame.index.name == "Date"
    assert list(frame.columns) == STANDARD_OHLCV_COLUMNS
    assert frame["Ticker"].eq("TEST").all()
    assert frame["Provider"].eq("unit").all()
    assert frame.index.is_monotonic_increasing


def test_yahoo_batch_extraction_drops_other_ticker_calendar_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product([["ETF", "CRYPTO"], ["Close", "Volume"]])
    raw = pd.DataFrame(index=dates, columns=columns, dtype=float)
    raw.loc[dates[[0, 2]], ("ETF", "Close")] = [100.0, 101.0]
    raw.loc[dates[[0, 2]], ("ETF", "Volume")] = [10.0, 11.0]
    raw.loc[:, ("CRYPTO", "Close")] = [1.0, 2.0, 3.0]

    extracted = _extract_ticker_frame(raw, "ETF", batch_size=2)

    assert extracted.index.tolist() == [dates[0], dates[2]]
