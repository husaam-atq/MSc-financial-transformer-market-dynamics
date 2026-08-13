"""Resumable Binance Vision monthly 1-hour kline archive downloader."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests

from market_dynamics.data.providers.base import standardise_ohlcv

LOGGER = logging.getLogger(__name__)
BINANCE_VISION_ROOT = "https://data.binance.vision/data/spot/monthly/klines"
KLINE_COLUMNS = [
    "open_time", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume",
    "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


@dataclass(frozen=True)
class ArchiveDownload:
    """A downloaded archive and its parsed coverage."""

    path: Path
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp


class BinanceVisionProvider:
    """Download official Binance Vision spot kline archives, without API keys."""

    name = "binance_vision"

    def __init__(self, cache_root: str | Path, interval: str = "1h", timeout_seconds: int = 60) -> None:
        self.cache_root = Path(cache_root)
        self.interval = interval
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def archive_url(self, symbol: str, month: pd.Timestamp) -> str:
        period = pd.Timestamp(month).strftime("%Y-%m")
        filename = f"{symbol}-{self.interval}-{period}.zip"
        return f"{BINANCE_VISION_ROOT}/{symbol}/{self.interval}/{filename}"

    def download_month(
        self,
        symbol: str,
        month: pd.Timestamp,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
    ) -> ArchiveDownload | None:
        """Fetch and validate a single immutable monthly archive; return None for HTTP 404."""
        month = pd.Timestamp(month).to_period("M").to_timestamp()
        target = self.cache_root / "spot" / "monthly" / "klines" / symbol / self.interval
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{symbol}-{self.interval}-{month:%Y-%m}.zip"
        if not path.exists():
            response: requests.Response | None = None
            for attempt in range(max_retries):
                try:
                    response = self.session.get(self.archive_url(symbol, month), timeout=self.timeout_seconds)
                    if response.status_code == 404:
                        LOGGER.warning("Binance Vision archive not found: %s", response.url)
                        return None
                    response.raise_for_status()
                    temporary = path.with_suffix(".zip.part")
                    temporary.write_bytes(response.content)
                    temporary.replace(path)
                    break
                except requests.RequestException as exc:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to download {symbol} {month:%Y-%m}: {exc}") from exc
                    time.sleep(backoff_seconds * (2**attempt))
            if response is None or not path.exists():
                raise RuntimeError(f"Archive retrieval unexpectedly failed for {symbol} {month:%Y-%m}")
        frame = self.read_archive(path, symbol)
        return ArchiveDownload(path=path, rows=len(frame), start=frame.index.min(), end=frame.index.max())

    def read_archive(self, path: str | Path, symbol: str) -> pd.DataFrame:
        """CRC-test and normalise a Binance Vision archive into standard OHLCV."""
        archive = Path(path)
        try:
            with ZipFile(archive) as zipped:
                bad_member = zipped.testzip()
                if bad_member:
                    raise RuntimeError(f"Archive checksum failed for {archive}: {bad_member}")
                csv_members = [name for name in zipped.namelist() if name.endswith(".csv")]
                if len(csv_members) != 1:
                    raise RuntimeError(f"Expected one CSV in {archive}, found {csv_members}")
                payload = zipped.read(csv_members[0])
        except BadZipFile as exc:
            raise RuntimeError(f"Corrupt Binance Vision ZIP archive: {archive}") from exc
        raw = pd.read_csv(BytesIO(payload), header=None, names=KLINE_COLUMNS)
        if raw.empty:
            raise RuntimeError(f"Empty Binance Vision archive: {archive}")
        unit = "us" if raw["open_time"].iloc[0] > 10**14 else "ms"
        raw["Date"] = pd.to_datetime(raw["open_time"], unit=unit, utc=True).dt.tz_localize(None)
        frame = raw.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        return standardise_ohlcv(frame, ticker=symbol, provider=self.name)
