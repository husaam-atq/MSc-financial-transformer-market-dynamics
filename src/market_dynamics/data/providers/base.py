"""Provider abstraction and OHLCV schema standardisation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

STANDARD_OHLCV_COLUMNS: list[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Ticker",
    "Provider",
]

NUMERIC_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass(frozen=True)
class ProviderRequest:
    """OHLCV request shared by all providers."""

    ticker: str
    start_date: str
    end_date: str | None = None


class BaseOHLCVProvider(ABC):
    """Base interface for providers returning standard OHLCV data."""

    name: ClassVar[str]

    @abstractmethod
    def fetch(self, request: ProviderRequest) -> pd.DataFrame:
        """Fetch and standardise OHLCV data for one ticker."""


def standardise_ohlcv(
    raw: pd.DataFrame,
    ticker: str,
    provider: str,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a standard OHLCV dataframe with a DatetimeIndex named ``Date``."""
    if raw is None or raw.empty:
        raise ValueError(f"No OHLCV rows returned for {ticker} from {provider}")

    frame = _flatten_columns(raw.copy())
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "Date"

    if column_map:
        frame = frame.rename(columns=column_map)

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in frame.columns:
            raise ValueError(
                f"{provider} data for {ticker} is missing required column '{column}'. "
                f"Available columns: {list(frame.columns)}"
            )

    if "Adj Close" not in frame.columns:
        frame["Adj Close"] = np.nan

    frame["Ticker"] = ticker
    frame["Provider"] = provider
    frame = frame[STANDARD_OHLCV_COLUMNS].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    for column in NUMERIC_OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    validate_ohlcv_schema(frame)
    return frame


def validate_ohlcv_schema(frame: pd.DataFrame) -> None:
    """Validate the standard provider output schema."""
    missing = [column for column in STANDARD_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV schema missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("OHLCV dataframe index must be a pandas DatetimeIndex")
    if frame.index.name != "Date":
        raise ValueError("OHLCV dataframe index must be named 'Date'")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("OHLCV dataframe index must be sorted ascending")
    if frame["Ticker"].isna().any() or frame["Provider"].isna().any():
        raise ValueError("Ticker and Provider columns cannot contain null values")


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            next((part for part in parts if str(part).lower() in _canonical_names()), parts[0])
            for parts in frame.columns.to_flat_index()
        ]
    return frame


def _canonical_names() -> set[str]:
    return {"open", "high", "low", "close", "adj close", "volume"}
