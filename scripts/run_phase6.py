"""Run the bounded Phase 6 Transformer falsification protocol."""

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
from market_dynamics.experiments.run_phase6 import (
    build_context,
    run_fresh_eligibility_audit,
    run_identity_swap,
    run_market_dynamics_dependence,
    run_phase6_analysis,
    run_phase6_data_audit,
    run_phase6_training,
    run_representation_probes,
    run_temporal_destruction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["data-audit", "train", "analyse", "temporal", "probes", "identity-swap", "market-dynamics", "fresh-audit", "all"])
    parser.add_argument("--phase2b-config", default=str(PROJECT_ROOT / "configs/phase2b_large_scale_config.yaml"))
    parser.add_argument("--phase2c-config", default=str(PROJECT_ROOT / "configs/phase2c_robustness_config.yaml"))
    parser.add_argument("--phase5-config", default=str(PROJECT_ROOT / "configs/phase5_config.yaml"))
    parser.add_argument("--phase6-config", default=str(PROJECT_ROOT / "configs/phase6_config.yaml"))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "results/runs/phase6_transformer_falsification_20260712"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.phase2b_config)
    config["phase2c"] = load_config(args.phase2c_config)["phase2c"]
    config["phase5"] = load_config(args.phase5_config)["phase5"]
    phase6 = load_config(args.phase6_config)
    context = build_context(config, phase6, args.run_dir)
    logging.info("Phase 6 split purge=%d device=%s", context.split.purge, context.device)
    if args.stage in {"data-audit", "all"}:
        outputs = run_phase6_data_audit(context)
        logging.info("Completed %d data-audit tables", len(outputs))
    if args.stage in {"train", "all"}:
        metrics = run_phase6_training(context)
        logging.info("Completed %d training summary rows", len(metrics))
    if args.stage in {"analyse", "all"}:
        outputs = run_phase6_analysis(context)
        logging.info("Wrote %d analysis tables", len(outputs))
    if args.stage in {"temporal", "all"}:
        rows = run_temporal_destruction(context)
        logging.info("Completed %d temporal perturbations", len(rows))
    if args.stage in {"probes", "all"}:
        rows = run_representation_probes(context)
        logging.info("Completed %d representation probe rows", len(rows))
    if args.stage in {"identity-swap", "all"}:
        rows = run_identity_swap(context)
        logging.info("Completed %d identity-swap rows", len(rows))
    if args.stage in {"market-dynamics", "all"}:
        rows = run_market_dynamics_dependence(context)
        logging.info("Completed %d dependence rows", len(rows))
    if args.stage in {"fresh-audit", "all"}:
        rows = run_fresh_eligibility_audit(context)
        logging.info("Completed %d fresh eligibility rows", len(rows))


if __name__ == "__main__":
    main()
