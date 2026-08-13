"""yfinance OHLCV provider for Phase 1."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import pandas as pd

from market_dynamics.data.providers.base import (
    BaseOHLCVProvider,
    ProviderRequest,
    standardise_ohlcv,
)

LOGGER = logging.getLogger(__name__)


class YFinanceProvider(BaseOHLCVProvider):
    """Fetch daily OHLCV data from Yahoo Finance through yfinance."""

    name = "yfinance"

    def fetch(self, request: ProviderRequest) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for the Phase 1 primary provider. "
                "Install it with `pip install yfinance`."
            ) from exc

        LOGGER.info("Downloading %s from yfinance", request.ticker)
        raw = yf.download(
            request.ticker,
            start=request.start_date,
            end=request.end_date,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError(
                f"yfinance returned no data for {request.ticker} "
                f"from {request.start_date} to {request.end_date or 'latest'}"
            )
        return standardise_ohlcv(raw, ticker=request.ticker, provider=self.name)

    def fetch_many(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str | None = None,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Download a Yahoo batch and return each successful ticker independently.

        Yahoo can return partial batches. Failed symbols are returned in the error
        mapping, then callers can record rather than silently discard them.
        """
        requested = list(dict.fromkeys(tickers))
        if not requested:
            return {}, {}
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for Yahoo ingestion") from exc
        raw: pd.DataFrame | None = None
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                raw = yf.download(
                    tickers=requested,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:  # yfinance raises provider-specific exceptions.
                last_error = exc
            time.sleep(backoff_seconds * (2**attempt))
        if raw is None or raw.empty:
            message = f"Yahoo batch download returned no data for {requested}"
            if last_error is not None:
                message += f": {last_error}"
            return {}, {ticker: message for ticker in requested}
        successful: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}
        for ticker in requested:
            try:
                asset_raw = _extract_ticker_frame(raw, ticker, len(requested))
                successful[ticker] = standardise_ohlcv(asset_raw, ticker=ticker, provider=self.name)
            except Exception as exc:
                failures[ticker] = str(exc)
        return successful, failures


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, batch_size: int) -> pd.DataFrame:
    """Extract one ticker from either yfinance batch MultiIndex layout."""
    if not isinstance(raw.columns, pd.MultiIndex):
        if batch_size == 1:
            return raw.dropna(how="all")
        raise ValueError("Yahoo batch response lacks a ticker MultiIndex")
    if ticker in raw.columns.get_level_values(0):
        return raw[ticker].dropna(how="all")
    if ticker in raw.columns.get_level_values(-1):
        return raw.xs(ticker, axis=1, level=-1).dropna(how="all")
    raise ValueError(f"Yahoo batch contains no columns for {ticker}")
