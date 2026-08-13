"""Stooq provider placeholder with graceful optional dependency handling."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from market_dynamics.data.providers.base import (
    BaseOHLCVProvider,
    ProviderRequest,
    standardise_ohlcv,
)
from market_dynamics.data.providers.pandas_datareader_compat import import_datareader

LOGGER = logging.getLogger(__name__)


class ManualStooqImportRequired(RuntimeError):
    """Raised when Stooq blocks automated access and a manual CSV is required."""


class StooqProvider(BaseOHLCVProvider):
    """Fetch Stooq daily OHLCV, with a compliant direct/manual fallback path."""

    name = "stooq"

    def __init__(self, manual_root: str | Path | None = None) -> None:
        self.manual_root = Path(manual_root) if manual_root is not None else None

    def fetch(self, request: ProviderRequest) -> pd.DataFrame:
        manual_file = self._manual_file(request.ticker)
        if manual_file is not None and manual_file.exists():
            return self.import_manual_csv(manual_file, request.ticker)
        web = import_datareader()

        LOGGER.info("Downloading %s from Stooq", request.ticker)
        end_date = request.end_date or pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
        try:
            raw = web.DataReader(
                request.ticker,
                "stooq",
                start=request.start_date,
                end=end_date,
            )
            if raw.empty:
                raise RuntimeError(f"Stooq returned no data for {request.ticker}")
            return standardise_ohlcv(raw.sort_index(), ticker=request.ticker, provider=self.name)
        except Exception as reader_error:
            try:
                return self.fetch_direct(request)
            except ManualStooqImportRequired as direct_error:
                required = self._manual_file(request.ticker)
                location = required if required is not None else Path("data/external/stooq_manual") / f"{request.ticker}.csv"
                raise ManualStooqImportRequired(
                    f"Automated Stooq access failed ({type(reader_error).__name__}); "
                    f"direct CSV access is unavailable or challenged ({direct_error}). "
                    f"Download the daily CSV manually from Stooq and place it at {location}."
                ) from reader_error

    def fetch_direct(self, request: ProviderRequest) -> pd.DataFrame:
        """Attempt Stooq's documented direct CSV endpoint; never solve access challenges."""
        params = {"s": request.ticker, "i": "d"}
        response = requests.get("https://stooq.com/q/d/l/", params=params, timeout=30)
        content_type = response.headers.get("Content-Type", "").lower()
        text = response.text.lstrip()
        if response.status_code != 200 or text.startswith("<") or "Date,Open,High,Low,Close" not in text:
            raise ManualStooqImportRequired(
                f"Stooq direct endpoint did not provide a CSV (HTTP {response.status_code}; content type {content_type or 'unknown'})."
            )
        raw = pd.read_csv(StringIO(response.text))
        if "Date" not in raw.columns:
            raise ManualStooqImportRequired("Stooq direct response did not contain a Date column.")
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.set_index("Date")
        return standardise_ohlcv(raw, ticker=request.ticker, provider="stooq_direct")

    def import_manual_csv(self, path: str | Path, ticker: str) -> pd.DataFrame:
        """Validate a manually downloaded Stooq daily CSV without modifying source values."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Manual Stooq CSV not found: {source}")
        raw = pd.read_csv(source)
        if "Date" not in raw.columns:
            raise ValueError(f"Manual Stooq CSV must include Date: {source}")
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date"]).set_index("Date")
        return standardise_ohlcv(raw, ticker=ticker, provider="stooq_manual")

    def _manual_file(self, ticker: str) -> Path | None:
        return None if self.manual_root is None else self.manual_root / f"{ticker.lower()}.csv"
