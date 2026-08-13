"""Run Phase 2C walk-forward robustness evaluation."""

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
from market_dynamics.experiments.run_walkforward_robustness import run_walkforward_robustness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2b-config", default=str(PROJECT_ROOT / "configs" / "phase2b_large_scale_config.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase2c_robustness_config.yaml"))
    parser.add_argument("--run-dir", default=None, help="Optional existing or new run directory for progress snapshots.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    phase2b = load_config(args.phase2b_config)
    phase2c = load_config(args.config)
    phase2b["phase2c"] = phase2c["phase2c"]
    ensure_project_dirs(phase2b)
    metrics = run_walkforward_robustness(phase2b, run_dir=args.run_dir)
    logging.info("Phase 2C complete with %d rows", len(metrics))


if __name__ == "__main__":
    main()
