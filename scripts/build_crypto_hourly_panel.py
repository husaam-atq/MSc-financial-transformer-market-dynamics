"""Command-line builder for the Phase 2B hourly Binance Vision panel."""

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
from market_dynamics.data.panel_builders import build_crypto_hourly_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2b_large_scale_config.yaml")
    args = parser.parse_args([value.replace("â€“", "--") if value.startswith("â€“") else value for value in sys.argv[1:]])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    ensure_project_dirs(config)
    result = build_crypto_hourly_panel(config, config["_meta"]["project_root"])
    logging.info("Hourly crypto panel complete: %s rows across %s included assets", result["rows"], result["assets"])


if __name__ == "__main__":
    main()
