"""Run the independent Stooq and CCXT reconciliation checks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from market_dynamics.config import ensure_project_dirs, load_config
from market_dynamics.data.panel_builders import reconcile_ccxt_recent, reconcile_daily_providers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2b_large_scale_config.yaml")
    parser.add_argument("--track", choices=["daily", "crypto", "all"], default="all")
    args = parser.parse_args([value.replace("â€“", "--") if value.startswith("â€“") else value for value in sys.argv[1:]])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    ensure_project_dirs(config)
    if args.track in {"daily", "all"}:
        report = reconcile_daily_providers(config, config["_meta"]["project_root"])
        logging.info("Stooq reconciliation: %d successful / %d total", (report["status"] == "success").sum(), len(report))
    if args.track in {"crypto", "all"}:
        report = reconcile_ccxt_recent(config, config["_meta"]["project_root"])
        logging.info("CCXT reconciliation: %d successful / %d total", (report["status"] == "success").sum(), len(report))


if __name__ == "__main__":
    main()
