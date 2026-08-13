"""Run the frozen PRP-1 fixed cross-model comparison."""

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
from market_dynamics.experiments.run_prp1_fixed_cross_model import (
    build_fixed_cross_model_context,
    run_evaluation,
    run_identity_swap,
    run_preflight,
    run_representation_probes,
    run_smoke,
    run_temporal_order,
    run_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["preflight", "smoke", "train", "evaluate", "temporal", "identity-swap", "probes", "all"],
    )
    parser.add_argument("--phase2b-config", default=str(PROJECT_ROOT / "configs/phase2b_large_scale_config.yaml"))
    parser.add_argument("--phase2c-config", default=str(PROJECT_ROOT / "configs/phase2c_robustness_config.yaml"))
    parser.add_argument("--phase5-config", default=str(PROJECT_ROOT / "configs/phase5_config.yaml"))
    parser.add_argument("--phase6-config", default=str(PROJECT_ROOT / "configs/phase6_config.yaml"))
    parser.add_argument("--cross-model-config", default=str(PROJECT_ROOT / "configs/prp1_fixed_cross_model_config.yaml"))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "results/runs/prp1_fixed_cross_model_20260715"))
    parser.add_argument("--skip-logistic", action="store_true", help="Development-only staged execution; not a complete registered run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    base = load_config(args.phase2b_config)
    base["phase2c"] = load_config(args.phase2c_config)["phase2c"]
    base["phase5"] = load_config(args.phase5_config)["phase5"]
    phase6 = load_config(args.phase6_config)
    cross_model = load_config(args.cross_model_config)
    context = build_fixed_cross_model_context(base, phase6, cross_model, args.run_dir)
    logging.info(
        "Cross-model contract device=%s train=%d validation=%d test=%d endpoint=%s",
        context.phase6.device,
        len(context.bundle.train),
        len(context.bundle.validation),
        len(context.bundle.test),
        context.endpoint_sha256[:12],
    )

    if args.stage in {"preflight", "all"}:
        logging.info("Preflight rows=%d", len(run_preflight(context)))
    if args.stage in {"smoke", "all"}:
        logging.info("Smoke rows=%d", len(run_smoke(context)))
    if args.stage in {"train", "all"}:
        logging.info("Execution rows=%d", len(run_training(context, include_logistic=not args.skip_logistic)))
    if args.stage in {"evaluate", "all"}:
        logging.info("Metric rows=%d", len(run_evaluation(context)))
    if args.stage in {"temporal", "all"}:
        logging.info("Temporal rows=%d", len(run_temporal_order(context)))
    if args.stage in {"identity-swap", "all"}:
        logging.info("Identity-swap rows=%d", len(run_identity_swap(context)))
    if args.stage in {"probes", "all"}:
        identity, states = run_representation_probes(context)
        logging.info("Probe rows identity=%d state=%d", len(identity), len(states))


if __name__ == "__main__":
    main()
