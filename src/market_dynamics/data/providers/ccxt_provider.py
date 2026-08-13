"""CCXT provider hook for later exchange-level crypto OHLCV robustness checks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from market_dynamics.data.providers.base import (
    BaseOHLCVProvider,
    ProviderRequest,
    standardise_ohlcv,
)

LOGGER = logging.getLogger(__name__)


class CCXTOHLCVProvider(BaseOHLCVProvider):
    """Fetch daily crypto OHLCV candles from an exchange through CCXT."""

    name = "ccxt"

    def __init__(self, exchange_id: str = "binance", timeframe: str = "1d", market_type: str = "spot") -> None:
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.market_type = market_type

    def fetch(self, request: ProviderRequest) -> pd.DataFrame:
        try:
            import ccxt
        except ImportError as exc:
            raise ImportError(
                "ccxt is required for exchange-level crypto OHLCV. "
                "Install it with `pip install ccxt`."
            ) from exc

        exchange_cls = getattr(ccxt, self.exchange_id)
        exchange = exchange_cls({"enableRateLimit": True, "options": {"defaultType": self.market_type}})
        symbol = request.ticker if "/" in request.ticker else request.ticker.replace("-USD", "/USDT")
        since = int(pd.Timestamp(request.start_date).timestamp() * 1000)
        end_ms = (
            int(pd.Timestamp(request.end_date).timestamp() * 1000)
            if request.end_date
            else int(datetime.now(tz=UTC).timestamp() * 1000)
        )

        LOGGER.info("Downloading %s from CCXT exchange %s", symbol, self.exchange_id)
        rows: list[list[float]] = []
        while since < end_ms:
            batch = exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            next_since = int(batch[-1][0]) + 1
            if next_since <= since:
                break
            since = next_since
            if len(batch) < 1000:
                break

        if not rows:
            raise RuntimeError(f"CCXT returned no data for {symbol} on {self.exchange_id}")

        raw = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        raw["Date"] = pd.to_datetime(raw["Date"], unit="ms")
        raw = raw.set_index("Date")
        frame = standardise_ohlcv(raw, ticker=request.ticker, provider=f"{self.name}:{self.exchange_id}")
        frame["Exchange"] = self.exchange_id
        frame["MarketType"] = self.market_type
        frame["ExchangeSymbol"] = symbol
        frame["RetrievedAtUTC"] = datetime.now(tz=UTC).isoformat()
        return frame
