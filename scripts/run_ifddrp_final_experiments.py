"""Run the three frozen IFDDRP final bounded experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from market_dynamics.config import load_config
from market_dynamics.experiments.ifddrp_final_experiments import (
    build_final_context,
    run_continuous_downside,
    run_static_dynamic,
    run_within_asset_objectives,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "static-dynamic", "within-objective", "continuous", "all"])
    parser.add_argument("--phase2b-config", default=str(PROJECT_ROOT / "configs/phase2b_large_scale_config.yaml"))
    parser.add_argument("--phase2c-config", default=str(PROJECT_ROOT / "configs/phase2c_robustness_config.yaml"))
    parser.add_argument("--phase5-config", default=str(PROJECT_ROOT / "configs/phase5_config.yaml"))
    parser.add_argument("--phase6-config", default=str(PROJECT_ROOT / "configs/phase6_config.yaml"))
    parser.add_argument("--static-config", default=str(PROJECT_ROOT / "configs/ifddrp_static_dynamic_decomposition.yaml"))
    parser.add_argument("--objective-config", default=str(PROJECT_ROOT / "configs/ifddrp_within_asset_objective.yaml"))
    parser.add_argument("--continuous-config", default=str(PROJECT_ROOT / "configs/ifddrp_continuous_downside.yaml"))
    parser.add_argument("--phase6-run-dir", default=str(PROJECT_ROOT / "results/runs/phase6_transformer_falsification_20260712"))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "results/runs/ifddrp_final_bounded_20260808"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.phase2b_config)
    config["phase2c"] = load_config(args.phase2c_config)["phase2c"]
    config["phase5"] = load_config(args.phase5_config)["phase5"]
    context = build_final_context(
        config,
        load_config(args.phase6_config),
        load_config(args.static_config),
        load_config(args.objective_config),
        load_config(args.continuous_config),
        phase6_run_dir=args.phase6_run_dir,
        run_dir=args.run_dir,
    )
    logging.info(
        "Final bounded context: assets=%d train=%d validation=%d test=%d device=%s",
        len(context.bundle.preprocessors),
        len(context.bundle.train),
        len(context.bundle.validation),
        len(context.bundle.test),
        context.phase6.device,
    )
    if args.stage in {"static-dynamic", "all"}:
        outputs = run_static_dynamic(context)
        logging.info("Static/dynamic complete: %d metric rows", len(outputs["results"]))
    if args.stage in {"within-objective", "all"}:
        outputs = run_within_asset_objectives(context)
        logging.info("Within-objective complete: %d metric rows", len(outputs["results"]))
    if args.stage in {"continuous", "all"}:
        outputs = run_continuous_downside(context)
        logging.info("Continuous downside complete: %d metric rows", len(outputs["results"]))


if __name__ == "__main__":
    main()
