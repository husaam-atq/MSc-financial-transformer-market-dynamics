"""Universe metadata loading and inclusion decisions for large-scale panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DAILY_UNIVERSE_COLUMNS = [
    "ticker",
    "asset_name",
    "asset_class",
    "region",
    "source_symbol_yahoo",
    "source_symbol_stooq",
    "inclusion_reason",
    "expected_start_date",
]
CRYPTO_UNIVERSE_COLUMNS = [
    "symbol",
    "binance_symbol",
    "ccxt_symbol",
    "asset_name",
    "asset_class",
    "expected_start_date",
    "inclusion_reason",
]


def load_universe(path: str | Path, required_columns: list[str]) -> pd.DataFrame:
    """Read a universe file and reject ambiguous or incomplete metadata."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Universe file does not exist: {source}")
    universe = pd.read_csv(source)
    missing = [column for column in required_columns if column not in universe.columns]
    if missing:
        raise ValueError(f"Universe {source} is missing required columns: {missing}")
    if universe.empty:
        raise ValueError(f"Universe {source} is empty")
    identifier = required_columns[0]
    if universe[identifier].isna().any() or universe[identifier].duplicated().any():
        raise ValueError(f"Universe identifier '{identifier}' must be populated and unique")
    return universe


def load_daily_universe(path: str | Path) -> pd.DataFrame:
    """Load the daily multi-asset universe."""
    return load_universe(path, DAILY_UNIVERSE_COLUMNS)


def load_crypto_hourly_universe(path: str | Path) -> pd.DataFrame:
    """Load the hourly crypto universe."""
    return load_universe(path, CRYPTO_UNIVERSE_COLUMNS)


def final_inclusion_manifest(
    universe: pd.DataFrame,
    observations: pd.DataFrame,
    identifier: str,
    min_history_rows: int,
) -> pd.DataFrame:
    """Record inclusion and exclusion without silently discarding short histories."""
    if identifier not in observations.columns:
        raise KeyError(f"Observation frame is missing identifier '{identifier}'")
    dates = observations.reset_index().rename(columns={observations.index.name or "index": "Date"})
    if "Close" in dates.columns:
        dates = dates[dates["Close"].notna()].copy()
    grouped = dates.groupby(identifier, observed=True)["Date"].agg(["count", "min", "max"])
    manifest = universe.merge(grouped, how="left", left_on=identifier, right_index=True)
    manifest = manifest.rename(columns={"count": "row_count", "min": "actual_start", "max": "actual_end"})
    manifest["row_count"] = manifest["row_count"].fillna(0).astype(int)
    manifest["included"] = manifest["row_count"] >= int(min_history_rows)
    manifest["decision_reason"] = manifest.apply(
        lambda row: "included" if row["included"] else f"history below minimum ({min_history_rows})",
        axis=1,
    )
    return manifest
