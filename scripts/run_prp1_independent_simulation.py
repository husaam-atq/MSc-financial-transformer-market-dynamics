"""Execute and report the preregistered independent shortcut simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from market_dynamics.config import load_config
from market_dynamics.experiments.run_prp1_independent_simulation import (
    run_simulation_programme,
    summarize_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "prp1_milestone2_config.yaml"))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "results" / "runs" / "prp1_independent_simulation_20260715"))
    args = parser.parse_args()
    config = load_config(args.config)
    options = config["prp1_milestone2"]["independent_simulation"]
    rows = run_simulation_programme(options, Path(args.run_dir))
    summary, assessment = summarize_simulation(rows, options)
    tables = Path(config["_meta"]["project_root"]) / "reports" / "tables"
    summary.to_csv(tables / "prp1_study_a_independent_simulation_results.csv", index=False)
    gate_inference = assessment["gate_inference"]
    pd.DataFrame(gate_inference).to_csv(
        tables / "prp1_study_a_independent_simulation_gate_inference.csv", index=False
    )
    lines = [
        "# PRP-1 Study A Independent Simulation Mechanisms",
        "",
        f"Completed {assessment['completed']} registered runs; failures: {assessment['failed']}.",
        "",
        "## Frozen estimates",
        "",
        *[f"- `{name}`: {float(value):.4f}." for name, value in assessment["estimates"].items()],
        "",
        "## Frozen gates",
        "",
        *[f"- `{name}`: {'passed' if value else 'failed'}." for name, value in assessment["gates"].items()],
        "",
        "## Registered-gate scope sensitivity",
        "",
        "The coded primary strong-signal gate uses persistence 0.7. The frozen prose did not state that qualifier, so the all-persistence estimates are reported as a non-selective sensitivity:",
        "",
        *[f"- `{name}`: {float(value):.4f}." for name, value in assessment["sensitivity_estimates"].items()],
        "",
        "The preregistered gates above use the frozen point-estimate rules. A separate seed-clustered table reports ordinary 95% and post-hoc Bonferroni simultaneous intervals; those intervals strengthen inference but do not retroactively redefine the gates.",
        "",
        "This independent stylised DGP establishes mechanism sufficiency, not prevalence in real markets or a phase transition. Robustness anchors cannot rescue failed core gates.",
    ]
    (tables / "prp1_study_a_independent_simulation_mechanisms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
