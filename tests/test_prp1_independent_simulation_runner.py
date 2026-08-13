from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from market_dynamics.experiments.run_prp1_independent_simulation import (
    _run_id,
    build_simulation_design,
    evaluate_simulation_cell,
    perturb_sequences,
    run_simulation_programme,
    simulation_config,
    summarize_simulation,
)
from market_dynamics.research.independent_shortcut_simulation import IndependentSimulationConfig


def test_design_contains_all_core_and_bounded_robustness_runs() -> None:
    options = {
        "base_seed": 10,
        "replications_per_core_cell": 2,
        "prior_heterogeneity": [0.0, 2.5],
        "dynamic_signal": [0.0, 1.5],
        "persistence": [0.0, 0.7],
        "robustness_scenarios": [{"id": "missingness", "missingness": 0.1}],
        "robustness_anchor_cells": [{"prior_heterogeneity": 0.0, "dynamic_signal": 0.0, "persistence": 0.0}],
    }
    design = build_simulation_design(options)
    assert len(design) == 18
    assert sum(row["scenario"] == "core" for row in design) == 16


def test_frozen_simulation_config_expands_to_1040_unique_registered_runs() -> None:
    frozen = yaml.safe_load(
        Path("configs/prp1_milestone2_config.yaml").read_text(encoding="utf-8")
    )["prp1_milestone2"]["independent_simulation"]

    design = build_simulation_design(frozen)
    core = [row for row in design if row["scenario"] == "core"]
    robustness = [row for row in design if row["scenario"] != "core"]
    run_ids = {_run_id(row) for row in design}
    concrete_configs = {
        json.dumps(asdict(simulation_config(frozen, row)), sort_keys=True)
        for row in design
    }

    assert len(core) == 640
    assert len(robustness) == 400
    assert len(design) == 1_040
    assert len(run_ids) == len(design)
    assert len(concrete_configs) == len(design)
    assert set(row["scenario"] for row in robustness) == {
        scenario["id"] for scenario in frozen["robustness_scenarios"]
    }


def test_sequence_perturbation_does_not_change_identity_or_targets() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a"],
            "family_id": ["f"],
            "target": [1],
            **{f"{prefix}{lag}": [float(lag)] for prefix in ["observed_dynamic_lag_", "observed_common_lag_", "event_lag_", "missing_lag_"] for lag in range(7, -1, -1)},
        }
    )
    reversed_frame = perturb_sequences(frame, 8, "reversed")
    assert reversed_frame.loc[0, "asset_id"] == "a"
    assert reversed_frame.loc[0, "target"] == 1
    assert reversed_frame.loc[0, "observed_dynamic_lag_7"] == 0.0
    assert reversed_frame.loc[0, "observed_dynamic_lag_0"] == 7.0


def test_cell_evaluation_distinguishes_no_signal_from_ordered_dynamic_signal() -> None:
    common = dict(
        n_assets=12,
        n_families=3,
        train_periods=180,
        purge_periods=8,
        validation_periods=100,
        prior_heterogeneity=0.0,
        persistence=0.7,
        common_shock_strength=0.0,
        cross_sectional_dependence=0.0,
        event_onset_probability=0.0,
        event_effect=0.0,
        missing_rate=0.0,
        seed=123,
    )
    no_signal = evaluate_simulation_cell(
        IndependentSimulationConfig(dynamic_signal_strength=0.0, **common)
    )
    ordered_signal = evaluate_simulation_cell(
        IndependentSimulationConfig(dynamic_signal_strength=1.5, **common)
    )

    no_signal_within = no_signal[
        "pooled_classifier_pair_weighted_within_asset_roc_auc"
    ]
    ordered_within = ordered_signal[
        "pooled_classifier_pair_weighted_within_asset_roc_auc"
    ]
    assert no_signal_within == pytest.approx(0.5, abs=0.06)
    assert ordered_within > 0.70
    assert ordered_within > no_signal_within + 0.20
    assert ordered_signal["reversal_auc_drop"] > 0.10
    assert ordered_signal["permutation_auc_drop"] > 0.03
    assert np.isfinite(ordered_signal["asset_prior_pooled_roc_auc"])


def test_gate_intervals_cluster_factorial_repeats_by_common_seed() -> None:
    rows: list[dict[str, object]] = []
    for seed in range(20):
        for prior, dynamic, persistence, asset_auc, within, reversal, permutation in [
            (0.0, 0.0, 0.0, 0.50, 0.50, 0.00, 0.00),
            (2.5, 0.0, 0.0, 0.90, 0.50, 0.00, 0.00),
            (0.0, 1.5, 0.7, 0.50, 0.78, 0.38, 0.10),
            (2.5, 1.5, 0.7, 0.90, 0.79, 0.39, 0.11),
        ]:
            rows.append(
                {
                    "status": "completed",
                    "scenario": "core",
                    "replicate": seed,
                    "seed": seed,
                    "prior_heterogeneity": prior,
                    "dynamic_signal": dynamic,
                    "persistence": persistence,
                    "asset_prior_pooled_roc_auc": asset_auc,
                    "global_prior_pooled_roc_auc": 0.5,
                    "global_prior_pr_auc": 0.2,
                    "global_prior_brier": 0.16,
                    "global_prior_log_loss": 0.5,
                    "family_prior_pooled_roc_auc": 0.5,
                    "family_prior_pr_auc": 0.2,
                    "family_prior_brier": 0.16,
                    "family_prior_log_loss": 0.5,
                    "asset_prior_pr_auc": 0.2,
                    "asset_prior_brier": 0.16,
                    "asset_prior_log_loss": 0.5,
                    "pooled_classifier_pooled_roc_auc": within,
                    "pooled_classifier_pr_auc": 0.3,
                    "pooled_classifier_brier": 0.14,
                    "pooled_classifier_log_loss": 0.45,
                    "pooled_classifier_pair_weighted_within_asset_roc_auc": within,
                    "pooled_classifier_asset_macro_roc_auc": within,
                    "pooled_classifier_eligible_assets": 30,
                    "train_rows": 12_000,
                    "validation_rows": 6_000,
                    "train_positives": 2_000,
                    "validation_positives": 1_000,
                    "train_prevalence": 1 / 6,
                    "validation_prevalence": 1 / 6,
                    "reversal_auc_drop": reversal,
                    "permutation_auc_drop": permutation,
                }
            )
    options = {
        "gates": {
            "minimum_static_prior_auc_increase": 0.15,
            "no_signal_within_auc_lower": 0.47,
            "no_signal_within_auc_upper": 0.53,
            "minimum_strong_signal_within_auc": 0.65,
            "minimum_strong_signal_reversal_drop": 0.10,
            "minimum_strong_signal_permutation_drop": 0.03,
        },
        "inference": {"confidence_level": 0.95},
    }
    _, assessment = summarize_simulation(pd.DataFrame(rows), options)

    inference = pd.DataFrame(assessment["gate_inference"])
    assert inference["seed_clusters"].eq(20).all()
    assert inference["preregistered_point_gate_passed"].all()
    assert inference["post_hoc_simultaneous_interval_support"].all()


def test_resume_rejects_configuration_change(tmp_path) -> None:
    options = {
        "base_seed": 10,
        "replications_per_core_cell": 1,
        "n_assets": 4,
        "n_families": 2,
        "train_periods": 30,
        "validation_periods": 20,
        "sequence_length": 4,
        "purge_periods": 4,
        "prior_heterogeneity": [0.0],
        "dynamic_signal": [0.0],
        "persistence": [0.0],
        "robustness_scenarios": [],
        "robustness_anchor_cells": [],
    }
    first = run_simulation_programme(options, tmp_path)
    assert len(first) == 1
    changed = {**options, "train_periods": 31}

    with pytest.raises(RuntimeError, match="different configuration"):
        run_simulation_programme(changed, tmp_path)
