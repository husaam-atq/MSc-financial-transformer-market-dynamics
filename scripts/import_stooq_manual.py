"""Validate manually downloaded Stooq CSVs placed in the project external-data directory."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from market_dynamics.config import load_config
from market_dynamics.data.providers.stooq_provider import StooqProvider
from market_dynamics.data.universes import load_daily_universe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase2b_large_scale_config.yaml"))
    parser.add_argument("--directory", default=str(PROJECT_ROOT / "data" / "external" / "stooq_manual"))
    args = parser.parse_args()
    config = load_config(args.config)
    source_dir = Path(args.directory)
    source_dir.mkdir(parents=True, exist_ok=True)
    universe = load_daily_universe(PROJECT_ROOT / config["phase2b"]["daily"]["universe"])
    provider = StooqProvider(source_dir)
    rows: list[dict[str, object]] = []
    for row in universe[universe["source_symbol_stooq"].notna()].itertuples(index=False):
        source = source_dir / f"{row.source_symbol_stooq.lower()}.csv"
        try:
            frame = provider.import_manual_csv(source, row.ticker)
            target = PROJECT_ROOT / "data" / "raw" / "stooq" / "manual" / f"ticker={row.ticker}" / f"imported_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.reset_index().to_parquet(target, index=False)
            rows.append({"ticker": row.ticker, "source_file": str(source), "status": "imported", "rows": len(frame), "start": frame.index.min(), "end": frame.index.max()})
        except Exception as exc:
            rows.append({"ticker": row.ticker, "source_file": str(source), "status": "missing_or_invalid", "reason": f"{type(exc).__name__}: {exc}"})
    output = pd.DataFrame(rows)
    target = PROJECT_ROOT / "reports" / "tables" / "stooq_manual_import_status.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    print(target)


if __name__ == "__main__":
    main()
