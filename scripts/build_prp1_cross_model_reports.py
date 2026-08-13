"""Build registered PRP-1 cross-model verdict and Study B stop/go reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "reports/tables"
RUN_DIR = PROJECT_ROOT / "results/runs/prp1_fixed_cross_model_20260715"
MODELS = ("mlp", "lstm", "tcn", "transformer_encoder", "flattened_logistic")
SEQUENCE_MODELS = {"lstm", "tcn", "transformer_encoder"}


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = [column.replace("_", " ") for column in columns]
    rows = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(_fmt(value) if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _load_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _gate_table(results: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    pooled = results[
        results["aggregation"].eq("pooled")
        & results["identity_variant"].eq("asset_conditioned")
        & results["model"].isin(MODELS)
    ].copy()
    ensemble = pooled[pooled["seed"].astype(str).eq("ensemble")].set_index("model")
    seed_rows = pooled[~pooled["seed"].astype(str).eq("ensemble")]
    temporal_conditioned = temporal[
        temporal["identity_variant"].eq("asset_conditioned") & temporal["model"].isin(MODELS)
    ]
    rows: list[dict[str, object]] = []
    for model in MODELS:
        base = ensemble.loc[model]
        seeds = seed_rows[seed_rows["model"].eq(model)]
        temporal_ensemble = temporal_conditioned[
            temporal_conditioned["model"].eq(model) & temporal_conditioned["seed"].astype(str).eq("ensemble")
        ]
        temporal_seeds = temporal_conditioned[
            temporal_conditioned["model"].eq(model) & ~temporal_conditioned["seed"].astype(str).eq("ensemble")
        ]
        base_point = float(base["pair_weighted_within_asset_roc_auc"])
        base_lower = float(base["within_asset_ci_lower"])
        all_seed_above_chance = bool((seeds["pair_weighted_within_asset_roc_auc"] > 0.5).all() and len(seeds) == 3)
        temporal_rows_complete = len(temporal_ensemble) == 3 and len(temporal_seeds) == 9
        ensemble_temporal_pass = bool(
            temporal_rows_complete
            and (temporal_ensemble["within_asset_auc_drop"] >= 0.02).all()
            and (temporal_ensemble["within_asset_auc_drop_ci_lower"] > 0.0).all()
        )
        seed_temporal_pass = bool(temporal_rows_complete and (temporal_seeds["within_asset_auc_drop"] >= 0.0).all())
        base_pass = bool(base_point >= 0.55 and base_lower > 0.5 and all_seed_above_chance)
        rows.append(
            {
                "model": model,
                "ensemble_within_asset_auc": base_point,
                "ensemble_within_auc_ci_lower": base_lower,
                "all_seeds_within_auc_above_0_5": all_seed_above_chance,
                "base_temporal_skill_gate": base_pass,
                "all_ensemble_perturbation_drops_ge_0_02_with_positive_lower_ci": ensemble_temporal_pass,
                "all_seed_perturbation_drops_nonnegative": seed_temporal_pass,
                "strict_model_gate_pass": bool(base_pass and ensemble_temporal_pass and seed_temporal_pass),
            }
        )
    return pd.DataFrame(rows)


def _failure_table(
    execution: pd.DataFrame,
    results: pd.DataFrame,
    temporal: pd.DataFrame,
    swap: pd.DataFrame,
    probes: pd.DataFrame,
    smoke: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, frame, accepted in (
        ("training", execution, {"completed", "reused"}),
        ("evaluation", results, {"completed"}),
        ("temporal_order", temporal, {"completed"}),
        ("identity_swap", swap, {"completed"}),
        ("identity_probes", probes, {"completed"}),
    ):
        if "status" not in frame:
            rows.append({"stage": stage, "status": "invalid_output", "model": "NA", "reason": "status column absent"})
            continue
        failed = frame[~frame["status"].isin(accepted)]
        for _, row in failed.iterrows():
            rows.append(
                {
                    "stage": stage,
                    "status": row.get("status", "unknown"),
                    "model": row.get("model", "NA"),
                    "reason": row.get("reason", row.get("failure_reason", "unspecified")),
                }
            )
    smoke_covered = set(zip(smoke["model"], ["asset_conditioned"] * len(smoke), strict=True))
    smoke_expected = set((model, variant) for model in MODELS for variant in ("asset_conditioned", "no_explicit_asset_id"))
    missing_smoke = sorted(smoke_expected.difference(smoke_covered))
    if missing_smoke:
        rows.append(
            {
                "stage": "smoke_gate",
                "status": "protocol_noncompliance",
                "model": ";".join(f"{model}/{variant}" for model, variant in missing_smoke),
                "reason": "execution proceeded without smoke coverage for all registered model/identity combinations",
            }
        )
    if not rows:
        rows.append(
            {
                "stage": "all_registered_cross_model_stages",
                "status": "no_training_or_evaluation_failures",
                "model": "all",
                "reason": "30 training/reuse cells, 163 evaluation rows, 120 temporal rows, 15 ID swaps and 48 identity probes completed",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    execution = _load_required(TABLE_DIR / "prp1_fixed_cross_model_execution_manifest.csv")
    results = _load_required(TABLE_DIR / "prp1_fixed_cross_model_results.csv")
    temporal = _load_required(TABLE_DIR / "prp1_fixed_cross_model_temporal_order.csv")
    probes = _load_required(TABLE_DIR / "prp1_fixed_cross_model_identity_probes.csv")
    dynamic = _load_required(TABLE_DIR / "prp1_fixed_cross_model_dynamic_state_probes.csv")
    swap = _load_required(RUN_DIR / "fixed_cross_model_identity_swap.csv")
    smoke = _load_required(RUN_DIR / "manifests" / "smoke.csv")

    if len(execution) != 30 or len(results) != 163 or len(temporal) != 120 or len(swap) != 15 or len(probes) != 48:
        raise RuntimeError(
            "Registered output counts changed: "
            f"execution={len(execution)} results={len(results)} temporal={len(temporal)} swap={len(swap)} probes={len(probes)}"
        )

    failures = _failure_table(execution, results, temporal, swap, probes, smoke)
    failures.to_csv(TABLE_DIR / "prp1_fixed_cross_model_failures.csv", index=False)
    gates = _gate_table(results, temporal)
    gates.to_csv(TABLE_DIR / "prp1_fixed_cross_model_temporal_skill_gates.csv", index=False)

    ensemble = results[
        results["aggregation"].eq("pooled")
        & results["seed"].astype(str).eq("ensemble")
        & results["model"].isin(MODELS)
    ].copy()
    static = results[results["model"].str.startswith("static_") & results["aggregation"].eq("pooled")].copy()
    passed = set(gates.loc[gates["strict_model_gate_pass"], "model"])
    sequence_passes = len(passed.intersection(SEQUENCE_MODELS))
    recurrence_pass = len(passed) >= 3 and sequence_passes >= 2
    strongest_pooled = ensemble.loc[ensemble["roc_auc"].idxmax()]
    strongest_within = ensemble.loc[ensemble["pair_weighted_within_asset_roc_auc"].idxmax()]
    static_asset_auc = float(static.loc[static["model"].eq("static_asset_prior"), "roc_auc"].iloc[0])

    empirical_failures = failures[failures["stage"].ne("smoke_gate")]
    verdict = f"""# PRP-1 Fixed Cross-Model Verdict

## Execution status

The frozen comparison completed all 30 registered model/identity/seed cells: 24 newly fitted cells and six immutable historical Transformer reconstructions. Evaluation produced 163 rows. Temporal falsification produced 90 per-seed and 30 ensemble rows. Identity swapping produced 15 rows. Asset/family probing produced 48 rows. Training/evaluation-stage failures: {len(empirical_failures)}.

The empirical cells are complete, but strict protocol completion is not claimed. The smoke manifest covered only four of ten model/identity combinations; flattened logistic and every no-ID arm were omitted before execution proceeded. The post-execution audit also repaired the ignored temporal summary, computed the registered stride-10 sensitivity from existing predictions, and added artefact hashes. These qualifications do not change the negative scientific verdict.

This is opened, adaptive historical evidence for methodological falsification. It is not independent confirmation and cannot be promoted without a distinct-provider replication.

## Ensemble results

{_markdown_table(ensemble, ['model', 'identity_variant', 'roc_auc', 'pr_auc', 'pair_weighted_within_asset_roc_auc', 'per_asset_macro_roc_auc', 'f1', 'balanced_accuracy', 'brier_score', 'degenerate_prediction'])}

## Static controls

{_markdown_table(static, ['model', 'roc_auc', 'pr_auc', 'pair_weighted_within_asset_roc_auc', 'f1', 'balanced_accuracy', 'brier_score'])}

The strongest pooled model/variant was `{strongest_pooled['model']}` / `{strongest_pooled['identity_variant']}` at ROC-AUC {_fmt(strongest_pooled['roc_auc'])}. The strongest ensemble within-asset point estimate was `{strongest_within['model']}` / `{strongest_within['identity_variant']}` at {_fmt(strongest_within['pair_weighted_within_asset_roc_auc'])}. The train-only static asset prior reached {_fmt(static_asset_auc)} pooled ROC-AUC.

## Frozen temporal-skill gate

{_markdown_table(gates, ['model', 'ensemble_within_asset_auc', 'ensemble_within_auc_ci_lower', 'all_seeds_within_auc_above_0_5', 'base_temporal_skill_gate', 'all_ensemble_perturbation_drops_ge_0_02_with_positive_lower_ci', 'all_seed_perturbation_drops_nonnegative', 'strict_model_gate_pass'])}

Strict model passes: {len(passed)} of 5 ({', '.join(sorted(passed)) if passed else 'none'}). Sequence-family passes: {sequence_passes} of 3. Cross-model recurrence required at least three of five models, including two of LSTM/TCN/Transformer. **Recurrence gate: {'PASS' if recurrence_pass else 'FAIL'}.**

## Scientific decision

{'The registered evidence supports recurring genuine temporal skill under the frozen gate.' if recurrence_pass else 'The registered evidence does not support recurring genuine temporal skill across model families.'} Pooled performance is not treated as temporal skill. Order sensitivity alone is not treated as useful predictive information. No architecture is protected, and Study B is not unlocked unless the recurrence gate passes.

The no-ID historical Transformer uses an inference-equivalent 34-to-46 input adapter with twelve immutable zero channels. It is an identity ablation, not a causal capacity-matched retraining contrast. All ranking metrics use raw scores; calibration and validation-selected thresholds affect probability and decision metrics only.
"""
    (TABLE_DIR / "prp1_fixed_cross_model_verdict.md").write_text(verdict, encoding="utf-8")

    temporal_ensemble = temporal[temporal["seed"].astype(str).eq("ensemble")].copy()
    order_report = f"""# PRP-1 Cross-Model Order-Dependence Analysis

The analysis applies reverse order, one deterministic permutation and a half-window circular shift to every registered model, identity variant and seed without changing endpoint membership or labels. The table reports ensemble raw-score results. Date-block bootstrap intervals use 1,000 circular draws of 20 global dates.

{_markdown_table(temporal_ensemble, ['model', 'identity_variant', 'method', 'original_roc_auc', 'perturbed_roc_auc', 'roc_auc_drop', 'original_within_asset_roc_auc', 'perturbed_within_asset_roc_auc', 'within_asset_auc_drop', 'within_asset_auc_drop_ci_lower', 'within_asset_auc_drop_ci_upper', 'prediction_spearman', 'mean_absolute_probability_displacement'])}

Order dependence is interpreted jointly with within-asset skill. A model can react to reordering without forecasting usefully, and an off-manifold perturbation is not a causal explanation. The strict gate and cross-model recurrence decision are recorded in `prp1_fixed_cross_model_temporal_skill_gates.csv` and `prp1_fixed_cross_model_verdict.md`.
"""
    (TABLE_DIR / "prp1_cross_model_order_dependence_analysis.md").write_text(order_report, encoding="utf-8")

    probe_summary = probes.groupby(["model", "identity_variant", "probe_label"], as_index=False).agg(
        probe_balanced_accuracy=("balanced_accuracy", "mean"),
        probe_balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    asset_probe = probe_summary[probe_summary["probe_label"].eq("asset_id")].drop(columns="probe_label")
    family_probe = probe_summary[probe_summary["probe_label"].eq("family_id")].drop(columns="probe_label").rename(
        columns={
            "probe_balanced_accuracy": "family_probe_balanced_accuracy",
            "probe_balanced_accuracy_std": "family_probe_balanced_accuracy_std",
        }
    )
    frontier = ensemble.merge(asset_probe, on=["model", "identity_variant"], how="left").merge(
        family_probe, on=["model", "identity_variant"], how="left"
    )
    temporal_mean = temporal_ensemble.groupby(["model", "identity_variant"], as_index=False)["within_asset_auc_drop"].mean().rename(
        columns={"within_asset_auc_drop": "mean_registered_within_auc_drop"}
    )
    swap_mean = swap.groupby("model", as_index=False).agg(
        id_swap_auc_change=("roc_auc_change", "mean"),
        id_swap_prediction_spearman=("prediction_spearman", "mean"),
    )
    frontier = frontier.merge(temporal_mean, on=["model", "identity_variant"], how="left").merge(swap_mean, on="model", how="left")
    frontier["strict_temporal_gate_pass"] = frontier["model"].isin(passed)
    frontier["study_b_eligible"] = recurrence_pass
    frontier["evidence_class"] = "descriptive_cross_model_diagnostic"
    frontier.to_csv(TABLE_DIR / "prp1_study_b_cross_model_frontier.csv", index=False)

    if recurrence_pass:
        relationship = pd.DataFrame(
            [{"status": "deferred", "reason": "Temporal recurrence passed; dynamic-state labels and controls must be frozen before relationship testing."}]
        )
        study_b_decision = "eligible_for_separate_preregistered_dynamic_state_protocol"
    else:
        relationship = pd.DataFrame(
            [{"status": "not_executed", "reason": "Frozen cross-model temporal-skill recurrence gate failed; correlation testing would overinterpret five opened model points."}]
        )
        study_b_decision = "scientifically_stopped_at_preregistered_gate"
    relationship.to_csv(TABLE_DIR / "prp1_study_b_relationship_tests.csv", index=False)

    conditioned = ensemble[ensemble["identity_variant"].eq("asset_conditioned")].set_index("model")
    no_id = ensemble[ensemble["identity_variant"].eq("no_explicit_asset_id")].set_index("model")
    probe_lookup = asset_probe.set_index(["model", "identity_variant"])["probe_balanced_accuracy"].to_dict()
    intervention_rows: list[dict[str, object]] = [
        {
            "source_stage": "historical_phase6",
            "model": "transformer_encoder",
            "intervention": "remove_explicit_asset_id",
            "identity_probe_balanced_accuracy_before": 0.1669,
            "identity_probe_balanced_accuracy_after": 0.016,
            "pooled_roc_auc_before": 0.7891,
            "pooled_roc_auc_after": 0.7159,
            "within_asset_roc_auc_before": 0.4982,
            "within_asset_roc_auc_after": 0.4783,
            "evidence_class": "post_hoc_robustness",
            "decision": "historical_identity_reduced_but_no_temporal_skill_preserved",
            "new_intervention_training_performed": False,
        }
    ]
    for model in MODELS:
        before = conditioned.loc[model]
        after = no_id.loc[model]
        intervention_rows.append(
            {
                "source_stage": "fixed_cross_model_ensemble",
                "model": model,
                "intervention": "remove_explicit_asset_id",
                "identity_probe_balanced_accuracy_before": probe_lookup.get((model, "asset_conditioned"), np.nan),
                "identity_probe_balanced_accuracy_after": probe_lookup.get((model, "no_explicit_asset_id"), np.nan),
                "pooled_roc_auc_before": before["roc_auc"],
                "pooled_roc_auc_after": after["roc_auc"],
                "within_asset_roc_auc_before": before["pair_weighted_within_asset_roc_auc"],
                "within_asset_roc_auc_after": after["pair_weighted_within_asset_roc_auc"],
                "evidence_class": "historical_held_out_but_adaptive",
                "decision": "descriptive_existing_identity_ablation_only",
                "new_intervention_training_performed": False,
            }
        )
    pd.DataFrame(intervention_rows).to_csv(TABLE_DIR / "prp1_study_b_intervention_results.csv", index=False)

    study_b = f"""# PRP-1 Study B Final Verdict

**Decision: {study_b_decision}.**

The fixed cross-model stage provides descriptive asset/family decodability, no-ID contrasts, ID swaps and temporal-order diagnostics. Dynamic-state probes were not executed because their labels, shuffle controls, untrained references and unknown-ID handling were not frozen before outcome inspection (`{dynamic.iloc[0]['reason']}`).

The cross-model recurrence gate {'passed' if recurrence_pass else 'failed'}. Therefore {'a separate preregistered Study B protocol may now be designed, but no frontier is claimed here' if recurrence_pass else 'identity-versus-dynamics relationship tests and new interventions remain stopped'}. Five adaptively opened model points cannot establish a universal frontier. The descriptive rows in `prp1_study_b_cross_model_frontier.csv` must not be interpreted as a fitted law.
"""
    (TABLE_DIR / "prp1_study_b_final_verdict.md").write_text(study_b, encoding="utf-8")
    print(f"Built cross-model reports; recurrence_pass={recurrence_pass}, strict_model_passes={sorted(passed)}")


if __name__ == "__main__":
    main()
