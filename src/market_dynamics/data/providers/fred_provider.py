"""FRED provider hook for later macro/rates/risk features."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from market_dynamics.data.providers.pandas_datareader_compat import import_datareader

LOGGER = logging.getLogger(__name__)


class FREDProvider:
    """Fetch macro series through pandas-datareader for Phase 2 feature extensions."""

    name = "fred"

    @staticmethod
    def credential_available() -> bool:
        """Load a local project .env when present and report only key presence."""
        _load_project_dotenv()
        return bool(os.getenv("FRED_API_KEY"))

    def fetch_series(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        _load_project_dotenv()
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FRED_API_KEY is required for FRED ingestion. Set it in the environment, "
                "for example in PowerShell: $env:FRED_API_KEY = 'your_key'."
            )
        web = import_datareader()

        LOGGER.info("Downloading FRED series: %s", ", ".join(symbols))
        frame = web.DataReader(symbols, "fred", start=start_date, end=end_date, api_key=api_key)
        if frame.empty:
            raise RuntimeError(f"FRED returned no data for series: {symbols}")
        frame.index.name = "Date"
        return frame.sort_index()


def _load_project_dotenv() -> None:
    """Load a project-local .env without logging its values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        dotenv_path = candidate / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            return
