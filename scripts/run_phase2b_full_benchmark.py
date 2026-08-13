"""Run Layer 2 full-universe local and pooled benchmarks."""

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
from market_dynamics.experiments.run_pooled_models import run_full_universe_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase2b_large_scale_config.yaml"))
    parser.add_argument("--run-dir", default=None, help="Optional existing or new run directory for progress snapshots.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    ensure_project_dirs(config)
    metrics = run_full_universe_benchmark(config, run_dir=args.run_dir)
    logging.info("Layer 2 completed with %d rows", len(metrics))


if __name__ == "__main__":
    main()
