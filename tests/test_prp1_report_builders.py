from __future__ import annotations

import pandas as pd

from scripts.build_prp1_cross_model_reports import MODELS, _gate_table


def _registered_gate_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    temporal_rows = []
    for model in MODELS:
        for seed in (7, 42, 123):
            result_rows.append(
                {
                    "model": model,
                    "identity_variant": "asset_conditioned",
                    "aggregation": "pooled",
                    "seed": seed,
                    "pair_weighted_within_asset_roc_auc": 0.56,
                    "within_asset_ci_lower": float("nan"),
                }
            )
            for method in ("reverse", "deterministic_permutation", "circular_shift"):
                temporal_rows.append(
                    {
                        "model": model,
                        "identity_variant": "asset_conditioned",
                        "seed": seed,
                        "method": method,
                        "within_asset_auc_drop": 0.01,
                        "within_asset_auc_drop_ci_lower": float("nan"),
                    }
                )
        result_rows.append(
            {
                "model": model,
                "identity_variant": "asset_conditioned",
                "aggregation": "pooled",
                "seed": "ensemble",
                "pair_weighted_within_asset_roc_auc": 0.57,
                "within_asset_ci_lower": 0.51,
            }
        )
        for method in ("reverse", "deterministic_permutation", "circular_shift"):
            temporal_rows.append(
                {
                    "model": model,
                    "identity_variant": "asset_conditioned",
                    "seed": "ensemble",
                    "method": method,
                    "within_asset_auc_drop": 0.03,
                    "within_asset_auc_drop_ci_lower": 0.01,
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(temporal_rows)


def test_gate_requires_all_seed_and_ensemble_temporal_conditions() -> None:
    """Unit-test gate aggregation; chronology is covered by split and prediction tests."""
    results, temporal = _registered_gate_inputs()

    passing = _gate_table(results, temporal)

    assert passing["strict_model_gate_pass"].all()

    affected = (
        temporal["model"].eq("lstm")
        & temporal["seed"].eq(42)
        & temporal["method"].eq("reverse")
    )
    temporal.loc[affected, "within_asset_auc_drop"] = -0.001

    revised = _gate_table(results, temporal).set_index("model")

    assert not bool(revised.loc["lstm", "strict_model_gate_pass"])
    assert bool(revised.loc["mlp", "strict_model_gate_pass"])
