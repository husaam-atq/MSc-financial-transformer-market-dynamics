"""Run historical-only Phase 5 family failure diagnosis and post-processing selection."""

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
from market_dynamics.experiments.run_phase5_diagnosis import run_phase5_cross_asset_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase2b_large_scale_config.yaml"))
    parser.add_argument("--phase2c-config", default=str(PROJECT_ROOT / "configs" / "phase2c_robustness_config.yaml"))
    parser.add_argument("--phase5-config", default=str(PROJECT_ROOT / "configs" / "phase5_config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    config["phase2c"] = load_config(args.phase2c_config)["phase2c"]
    phase5 = load_config(args.phase5_config)
    ensure_project_dirs(config)
    result = run_phase5_cross_asset_diagnosis(config, phase5)
    logging.info("Phase 5 diagnosis completed: performance_rows=%d calibration_rows=%d", len(result["performance"]), len(result["calibration"]))


if __name__ == "__main__":
    main()
