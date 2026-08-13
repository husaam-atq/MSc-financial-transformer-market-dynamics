"""Run the preregistered Layer 1 Phase 2B model screening."""

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
from market_dynamics.experiments.run_large_scale_screening import run_large_scale_screening


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2B Layer 1 model screening.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase2b_large_scale_config.yaml"))
    parser.add_argument("--track", choices=["daily", "crypto", "all"], default="all")
    parser.add_argument("--run-dir", default=None, help="Optional existing or new run directory for progress snapshots.")
    args = parser.parse_args([value.replace("â€“", "--") if value.startswith("â€“") else value for value in sys.argv[1:]])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    ensure_project_dirs(config)
    tracks = ("daily",) if args.track == "daily" else (("crypto_hourly",) if args.track == "crypto" else ("daily", "crypto_hourly"))
    result = run_large_scale_screening(config, tracks=tracks, run_dir=args.run_dir)
    logging.info("Screening complete: %d metric/status rows", len(result))


if __name__ == "__main__":
    main()
