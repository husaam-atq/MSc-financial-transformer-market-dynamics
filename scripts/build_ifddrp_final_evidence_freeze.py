"""Reconcile authoritative IFDDRP evidence and build the final dissertation freeze pack.

This script performs arithmetic verification against preserved local predictions. It
does not fit models, alter predictions, or inspect unregistered alternatives.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "reports" / "tables"
PHASE6_RUN = PROJECT_ROOT / "results" / "runs" / "phase6_transformer_falsification_20260712"
PREDICTIONS = PHASE6_RUN / "predictions"
FREEZE_DATE = "2026-08-08"
# Historical provenance recorded in generated evidence; it is not a runtime Git dependency.
EVIDENCE_SOURCE_COMMIT = "e66470a6de35019cf8239eb4f8e6b8aab38e0bbb"


def _require(relative: str) -> Path:
    path = PROJECT_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"Required final-freeze input is missing: {relative}")
    return path


def _write_markdown(name: str, content: str) -> None:
    path = TABLES / name
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create empty final-freeze table: {name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Inconsistent columns while writing {name}")
    path = TABLES / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: Iterable[dict[str, Any]]) -> str:
    materialized = list(rows)
    if not materialized:
        return "_No rows._"
    columns = list(materialized[0])

    def clean(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(row.get(column, "")) for column in columns) + " |"
        for row in materialized
    )
    return "\n".join(lines)


def _within_asset_auc(y: np.ndarray, score: np.ndarray, assets: np.ndarray) -> tuple[float, float, int]:
    weighted_sum = 0.0
    total_pairs = 0
    asset_aucs: list[float] = []
    for asset in np.unique(assets):
        mask = assets == asset
        asset_y = y[mask]
        positives = int(np.sum(asset_y == 1))
        negatives = int(np.sum(asset_y == 0))
        if positives == 0 or negatives == 0:
            continue
        auc = float(roc_auc_score(asset_y, score[mask]))
        pairs = positives * negatives
        weighted_sum += auc * pairs
        total_pairs += pairs
        asset_aucs.append(auc)
    if not asset_aucs or total_pairs == 0:
        raise ValueError("No assets contain both classes for within-asset ROC-AUC")
    return weighted_sum / total_pairs, float(np.mean(asset_aucs)), len(asset_aucs)


def _prediction_metrics(path: Path, variant: str) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    test = frame.loc[frame["split"].eq("test")].copy()
    if len(test) != 21_514:
        raise RuntimeError(f"Unexpected {variant} test rows: {len(test)}")
    y = test["y_true"].to_numpy(dtype=int)
    raw = test["ensemble_probability"].to_numpy(dtype=float)
    calibrated = test["selected_probability"].to_numpy(dtype=float)
    assets = test["asset_id"].to_numpy(dtype=int)
    within, macro, eligible = _within_asset_auc(y, raw, assets)
    return {
        "variant": variant,
        "rows": len(test),
        "assets": int(test["asset_ticker"].nunique()),
        "raw_roc_auc": float(roc_auc_score(y, raw)),
        "raw_pr_auc": float(average_precision_score(y, raw)),
        "raw_brier": float(brier_score_loss(y, raw)),
        "raw_log_loss": float(log_loss(y, raw, labels=[0, 1])),
        "calibrated_roc_auc": float(roc_auc_score(y, calibrated)),
        "calibrated_pr_auc": float(average_precision_score(y, calibrated)),
        "calibrated_brier": float(brier_score_loss(y, calibrated)),
        "calibrated_log_loss": float(log_loss(y, calibrated, labels=[0, 1])),
        "pair_weighted_within_auc": within,
        "per_asset_macro_auc": macro,
        "within_eligible_assets": eligible,
    }


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-6) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise RuntimeError(f"{label} changed: expected {expected:.9f}, found {actual:.9f}")


def _source_state() -> dict[str, Any]:
    phase6_config = yaml.safe_load(_require("configs/phase6_config.yaml").read_text(encoding="utf-8"))["phase6"]
    interpret_config = yaml.safe_load(
        _require("configs/ifddrp_final_interpretability.yaml").read_text(encoding="utf-8")
    )["ifddrp_final_interpretability"]
    manifest = json.loads((PHASE6_RUN / "phase6_run_manifest.json").read_text(encoding="utf-8"))
    universe = pd.read_csv(_require("configs/universes/daily_global_universe.csv"))
    conditioned_path = PREDICTIONS / "corrected_asset_conditioned_ensemble.parquet"
    no_id_path = PREDICTIONS / "no_explicit_asset_id_ensemble.parquet"
    conditioned = _prediction_metrics(conditioned_path, "corrected_asset_conditioned")
    no_id = _prediction_metrics(no_id_path, "no_explicit_asset_id")

    conditioned_frame = pd.read_parquet(conditioned_path)
    no_id_frame = pd.read_parquet(no_id_path)
    keys = ["split", "Date", "source_index", "asset_id", "y_true"]
    if not conditioned_frame[keys].equals(no_id_frame[keys]):
        raise RuntimeError("Conditioned and no-ID prediction endpoints do not align exactly")

    decomposition = pd.read_csv(_require("reports/tables/ifddrp_identity_dynamic_information_decomposition.csv"))
    static = decomposition.set_index("component")
    cross = pd.read_csv(_require("reports/tables/prp1_fixed_cross_model_results.csv"))
    gates = pd.read_csv(_require("reports/tables/prp1_fixed_cross_model_temporal_skill_gates.csv"))
    simulation = pd.read_csv(_require("reports/tables/prp1_study_a_independent_simulation_gate_inference.csv"))
    sanity = pd.read_csv(_require("reports/tables/ifddrp_interpretability_sanity_checks.csv"))
    regimes = pd.read_csv(_require("reports/tables/ifddrp_regime_conditional_results.csv"))

    _assert_close(conditioned["raw_roc_auc"], 0.7898135576855014, "conditioned raw ROC-AUC")
    _assert_close(conditioned["pair_weighted_within_auc"], 0.49163847014866996, "conditioned within ROC-AUC")
    _assert_close(no_id["raw_roc_auc"], 0.7154766551871594, "no-ID raw ROC-AUC")
    _assert_close(no_id["pair_weighted_within_auc"], 0.47257008594782257, "no-ID within ROC-AUC")
    _assert_close(float(static.loc["training_only_asset_prior", "pooled_roc_auc"]), 0.8239058561294393, "asset prior ROC-AUC")
    _assert_close(float(static.loc["training_only_family_prior", "pooled_roc_auc"]), 0.816548771506871, "family prior ROC-AUC")
    if gates["strict_model_gate_pass"].astype(bool).any():
        raise RuntimeError("A cross-model temporal-skill gate unexpectedly passed")
    if not simulation["preregistered_point_gate_passed"].astype(bool).all():
        raise RuntimeError("A registered independent-simulation gate unexpectedly failed")
    if len(regimes) != 14:
        raise RuntimeError(f"Expected 14 bounded regime rows, found {len(regimes)}")
    if int(sanity.loc[sanity["check"].eq("authoritative_prediction_reconstruction"), "passed"].sum()) != 6:
        raise RuntimeError("Authoritative prediction reconstruction is incomplete")

    feature_partition = interpret_config["feature_groups"]
    configured_features = [feature for group in feature_partition.values() for feature in group]
    if len(configured_features) != 34 or len(set(configured_features)) != 34:
        raise RuntimeError("Interpretability feature partition must contain 34 unique model inputs")
    required_groups = {
        "returns_momentum",
        "volatility",
        "range_dispersion",
        "volume_liquidity",
        "macro_context",
    }
    if set(feature_partition) != required_groups:
        raise RuntimeError("Interpretability feature groups differ from the final fixed partition")
    if len(universe) != 80:
        raise RuntimeError(f"Expected 80 configured daily instruments, found {len(universe)}")
    if phase6_config["lookback"] != 60 or phase6_config["corrected_purge"] != 18:
        raise RuntimeError("Authoritative Phase 6 lookback or purge changed")

    return {
        "source_commit": EVIDENCE_SOURCE_COMMIT,
        "phase6": phase6_config,
        "manifest": manifest,
        "conditioned": conditioned,
        "no_id": no_id,
        "static": static,
        "cross": cross,
        "gates": gates,
        "simulation": simulation,
        "sanity": sanity,
        "regimes": regimes,
    }


def _metric_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    conditioned = state["conditioned"]
    no_id = state["no_id"]
    static = state["static"]
    cross = state["cross"]
    simulation = state["simulation"].set_index("estimate_name")
    rows: list[dict[str, Any]] = []

    def add(
        metric_id: str,
        module: str,
        model: str,
        variant: str,
        metric: str,
        value: float,
        source_basis: str,
        source_file: str,
        evidence_class: str,
        authoritative_use: str,
        ci_lower: float | str = "",
        ci_upper: float | str = "",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "metric_id": metric_id,
                "module": module,
                "model": model,
                "variant": variant,
                "split": "test" if module != "independent_simulation" else "simulation",
                "metric": metric,
                "value": f"{value:.9f}",
                "ci_lower": "" if ci_lower == "" else f"{float(ci_lower):.9f}",
                "ci_upper": "" if ci_upper == "" else f"{float(ci_upper):.9f}",
                "score_basis": source_basis,
                "source_file": source_file,
                "evidence_class": evidence_class,
                "authoritative_use": authoritative_use,
                "notes": notes,
            }
        )

    prediction_source = "results/runs/phase6_transformer_falsification_20260712/predictions"
    add("M001", "phase6", "transformer_encoder", "asset_conditioned", "roc_auc", conditioned["raw_roc_auc"], "raw ensemble probability", prediction_source, "historical held-out but adaptive", "ranking")
    add("M002", "phase6", "transformer_encoder", "asset_conditioned", "pr_auc", conditioned["raw_pr_auc"], "raw ensemble probability", prediction_source, "historical held-out but adaptive", "ranking")
    add("M003", "phase6", "transformer_encoder", "asset_conditioned", "pair_weighted_within_asset_roc_auc", conditioned["pair_weighted_within_auc"], "raw ensemble probability", prediction_source, "historical held-out but adaptive", "primary shortcut diagnostic")
    add("M004", "phase6", "transformer_encoder", "asset_conditioned", "per_asset_macro_roc_auc", conditioned["per_asset_macro_auc"], "raw ensemble probability", prediction_source, "historical held-out but adaptive", "supporting grouped diagnostic")
    add("M005", "phase6", "transformer_encoder", "asset_conditioned", "brier_score", conditioned["calibrated_brier"], "validation-selected isotonic probability", prediction_source, "historical held-out but adaptive", "proper score")
    add("M006", "phase6", "transformer_encoder", "asset_conditioned", "log_loss", conditioned["calibrated_log_loss"], "validation-selected isotonic probability", prediction_source, "historical held-out but adaptive", "proper score")
    add("M007", "phase6", "transformer_encoder", "no_explicit_asset_id", "roc_auc", no_id["raw_roc_auc"], "raw ensemble probability", prediction_source, "post-hoc robustness", "identity ablation ranking")
    add("M008", "phase6", "transformer_encoder", "no_explicit_asset_id", "pair_weighted_within_asset_roc_auc", no_id["pair_weighted_within_auc"], "raw ensemble probability", prediction_source, "post-hoc robustness", "identity ablation grouped diagnostic")
    add("M009", "phase6", "static_asset_prior", "training_only", "roc_auc", float(static.loc["training_only_asset_prior", "pooled_roc_auc"]), "training-only static probability", "reports/tables/ifddrp_identity_dynamic_information_decomposition.csv", "post-hoc robustness", "strongest static comparator")
    add("M010", "phase6", "static_family_prior", "training_only", "roc_auc", float(static.loc["training_only_family_prior", "pooled_roc_auc"]), "training-only static probability", "reports/tables/ifddrp_identity_dynamic_information_decomposition.csv", "post-hoc robustness", "static comparator")

    ensembles = cross.loc[(cross["seed"].astype(str).eq("ensemble")) & cross["aggregation"].eq("pooled")]
    for index, model in enumerate(["mlp", "lstm", "tcn", "transformer_encoder", "flattened_logistic"], start=11):
        row = ensembles.loc[(ensembles["model"].eq(model)) & ensembles["identity_variant"].eq("asset_conditioned")].iloc[0]
        add(
            f"M{index:03d}",
            "fixed_cross_model",
            model,
            "asset_conditioned",
            "roc_auc",
            float(row["roc_auc"]),
            "raw ensemble ranking score",
            "reports/tables/prp1_fixed_cross_model_results.csv",
            "historical held-out but adaptive",
            "cross-model pooled comparison",
            notes=f"within_asset_auc={float(row['pair_weighted_within_asset_roc_auc']):.6f}",
        )

    for index, estimate_name in enumerate(simulation.index, start=20):
        row = simulation.loc[estimate_name]
        add(
            f"M{index:03d}",
            "independent_simulation",
            "registered_simulation",
            estimate_name,
            estimate_name,
            float(row["estimate"]),
            "20-seed clustered simulation estimate",
            "reports/tables/prp1_study_a_independent_simulation_gate_inference.csv",
            "simulation",
            "mechanism evidence",
            ci_lower=float(row["ci95_lower"]),
            ci_upper=float(row["ci95_upper"]),
            notes="Mechanism sufficiency only; not empirical market confirmation.",
        )

    add("M030", "experiment_a", "mlp", "family_prior_dynamic_residual_validation_selected", "pair_weighted_within_asset_roc_auc", 0.525360, "raw ensemble score", "reports/tables/ifddrp_static_dynamic_results.csv", "valid negative", "bounded recovery test", 0.443706, 0.602033, "Validation selection passed point estimate but failed paired-lift and chronology gates.")
    add("M031", "experiment_b", "transformer_encoder", "bce_plus_within_asset_pairwise_validation_selected", "pair_weighted_within_asset_roc_auc", 0.561924, "raw ensemble score", "reports/tables/ifddrp_within_asset_objective_results.csv", "valid negative", "bounded recovery test", 0.492527, 0.625022, "Descriptive lift did not pass paired-lift or chronology gates.")
    add("M032", "experiment_c", "transformer_encoder", "no_id_temporal_validation_selected", "equal_asset_mae", 0.037816, "original target units", "reports/tables/ifddrp_continuous_downside_results.csv", "valid negative", "bounded recovery test", 0.035374, 0.040008, "Training asset mean equal-asset MAE was 0.020896.")
    add("M033", "experiment_c", "ridge", "sequence_summary", "equal_asset_mae", 0.020691, "original target units", "reports/tables/ifddrp_continuous_downside_results.csv", "supporting evidence", "strong simple continuous comparator")
    return rows


def _result_rows() -> list[dict[str, Any]]:
    return [
        {"result_id": "R01", "module": "corrected_transformer_pipeline", "question": "Can the corrected pooled Transformer rank the original stress label?", "status": "completed", "result_direction": "descriptive_positive", "primary_estimate": "raw pooled ROC-AUC 0.789814", "uncertainty_or_gate": "opened adaptive test; not a superiority gate", "evidence_class": "historical held-out but adaptive", "paper_role": "main-paper context", "source_files": "ifddrp_authoritative_metric_registry.csv; ifddrp_transformer_authoritative_specification.md", "permitted_claim": "The corrected Transformer achieved strong pooled discrimination on the heterogeneous label.", "prohibited_claim": "The Transformer demonstrated genuine temporal skill or superiority."},
        {"result_id": "R02", "module": "static_prior_falsification", "question": "Does a training-only static prior explain pooled ranking?", "status": "completed", "result_direction": "positive_falsification", "primary_estimate": "asset prior 0.823906; family prior 0.816549", "uncertainty_or_gate": "both exceed Transformer 0.789814", "evidence_class": "post-hoc robustness", "paper_role": "principal contribution 2", "source_files": "ifddrp_identity_dynamic_information_decomposition.csv", "permitted_claim": "Static cross-sectional priors outperformed the tested Transformer in pooled ROC-AUC.", "prohibited_claim": "All pooled financial models are shortcuts."},
        {"result_id": "R03", "module": "within_asset_evaluation", "question": "Does pooled ranking translate to within-asset timing?", "status": "completed", "result_direction": "negative", "primary_estimate": "pair-weighted within-asset ROC-AUC 0.491638", "uncertainty_or_gate": "cross-model strict gate failed", "evidence_class": "post-hoc robustness", "paper_role": "principal contribution 2", "source_files": "ifddrp_identity_dynamic_information_decomposition.csv; prp1_fixed_cross_model_temporal_skill_gates.csv", "permitted_claim": "The tested Transformer did not show useful within-asset ranking.", "prohibited_claim": "No financial time series contain temporal information."},
        {"result_id": "R04", "module": "identity_ablation", "question": "Does explicit identity affect pooled ranking?", "status": "completed", "result_direction": "positive_falsification", "primary_estimate": "pooled AUC fell 0.789814 to 0.715477", "uncertainty_or_gate": "within AUC remained below chance", "evidence_class": "post-hoc robustness", "paper_role": "main supporting evidence", "source_files": "ifddrp_identity_dynamic_information_decomposition.csv", "permitted_claim": "Explicit identity materially affected pooled ranking.", "prohibited_claim": "The drop identifies a causal identity mechanism."},
        {"result_id": "R05", "module": "representation_probe", "question": "Is asset identity decodable from Transformer summaries?", "status": "completed", "result_direction": "positive_diagnostic", "primary_estimate": "test accuracy 0.166930 versus chance about 0.0127", "uncertainty_or_gate": "post-hoc probe; decodability is not use", "evidence_class": "post-hoc robustness", "paper_role": "supporting/viva", "source_files": "ifddrp_transformer_state_probe_results.csv", "permitted_claim": "Asset identity was materially decodable from learned summaries.", "prohibited_claim": "The probe proves identity caused every prediction."},
        {"result_id": "R06", "module": "temporal_order_controls", "question": "Does the Transformer require sequence order?", "status": "completed", "result_direction": "negative", "primary_estimate": "pooled AUC changes approximately -0.0014 to +0.0012", "uncertainty_or_gate": "reverse, deterministic permutation and circular shift all fail dynamic condition", "evidence_class": "post-hoc robustness", "paper_role": "principal contribution 2", "source_files": "ifddrp_interpretability_sanity_checks.csv; prp1_fixed_cross_model_temporal_order.csv", "permitted_claim": "The tested Transformer ranking was nearly invariant to the registered full-window order controls.", "prohibited_claim": "Transformers are generally order invariant."},
        {"result_id": "R07", "module": "controlled_simulation", "question": "Can target-prior heterogeneity inflate pooled AUC without dynamics?", "status": "completed", "result_direction": "positive_methodological", "primary_estimate": "static-prior AUC increase 0.400021; no-signal within AUC 0.500086", "uncertainty_or_gate": "1,040 runs; all five registered gates pass", "evidence_class": "simulation", "paper_role": "principal contribution 3", "source_files": "prp1_study_a_independent_simulation_gate_inference.csv", "permitted_claim": "Controlled heterogeneity can generate large pooled discrimination without temporal signal.", "prohibited_claim": "Simulation proves the empirical panel has exactly that data-generating process."},
        {"result_id": "R08", "module": "cross_model_recurrence", "question": "Do shortcut concerns recur across bounded model families?", "status": "completed", "result_direction": "negative", "primary_estimate": "best learned pooled MLP 0.796478; best learned within MLP 0.556967", "uncertainty_or_gate": "0 of 5 strict temporal-skill gates pass", "evidence_class": "historical held-out but adaptive", "paper_role": "principal contribution 3", "source_files": "prp1_fixed_cross_model_results.csv; prp1_fixed_cross_model_temporal_skill_gates.csv", "permitted_claim": "No tested model passed the registered chronology-dependent temporal-skill gate.", "prohibited_claim": "Every architecture or dataset would fail."},
        {"result_id": "R09", "module": "experiment_a_static_dynamic", "question": "Does a temporal residual add value after static priors?", "status": "completed_gate_failed", "result_direction": "valid_negative", "primary_estimate": "selected test within AUC 0.525360", "uncertainty_or_gate": "paired lift and order-sensitivity requirements failed", "evidence_class": "valid negative", "paper_role": "supporting robustness", "source_files": "ifddrp_static_dynamic_verdict.md", "permitted_claim": "No promotable incremental temporal residual was recovered.", "prohibited_claim": "The highest non-selected test row is a discovery."},
        {"result_id": "R10", "module": "experiment_b_within_objective", "question": "Does within-asset-aligned training recover temporal skill?", "status": "completed_gate_failed", "result_direction": "valid_negative", "primary_estimate": "selected test within AUC 0.561924", "uncertainty_or_gate": "95% interval crosses 0.5; paired lift and chronology fail", "evidence_class": "valid negative", "paper_role": "supporting robustness", "source_files": "ifddrp_within_asset_objective_verdict.md", "permitted_claim": "A descriptive improvement did not satisfy the promotion gate.", "prohibited_claim": "Within-asset ranking training solved the shortcut problem."},
        {"result_id": "R11", "module": "experiment_c_continuous_downside", "question": "Does a continuous downside outcome reveal temporal skill?", "status": "completed_gate_failed", "result_direction": "valid_negative", "primary_estimate": "Transformer equal-asset MAE 0.037816 versus asset mean 0.020896", "uncertainty_or_gate": "80.98% relative deterioration; chronology intervals cross zero", "evidence_class": "valid negative", "paper_role": "supporting robustness", "source_files": "ifddrp_continuous_downside_verdict.md", "permitted_claim": "The selected Transformer did not improve continuous downside MAE.", "prohibited_claim": "Continuous risk outcomes are generally unforecastable."},
        {"result_id": "R12", "module": "interpretability", "question": "Which registered inputs most change the conditioned model score?", "status": "completed_limited", "result_direction": "diagnostic", "primary_estimate": "macro/context occlusion 0.160730; asset embedding occlusion 0.136454", "uncertainty_or_gate": "parameter-randomisation control failed; macro provenance limited", "evidence_class": "post-hoc robustness", "paper_role": "supporting/viva", "source_files": "ifddrp_transformer_feature_attribution.csv; ifddrp_interpretability_sanity_checks.csv", "permitted_claim": "These groups change model outputs under bounded perturbations.", "prohibited_claim": "They are causal economic drivers or faithful explanations."},
        {"result_id": "R13", "module": "regime_analysis", "question": "Is there a stable regime-specific temporal result?", "status": "completed_gate_failed", "result_direction": "negative", "primary_estimate": "0 of 14 regimes pass strict dynamic gate", "uncertainty_or_gate": "full latent-state analysis remains locked", "evidence_class": "post-hoc robustness", "paper_role": "one-line negative", "source_files": "ifddrp_regime_conditional_verdict.md; ifddrp_emergent_dynamics_gate.md", "permitted_claim": "No registered regime established chronology-dependent within-asset skill.", "prohibited_claim": "Regimes do not exist."},
        {"result_id": "R14", "module": "external_replication", "question": "Does the mechanism generalise beyond the main panel?", "status": "partial_inconclusive", "result_direction": "mixed", "primary_estimate": "Japan partial shortcut pattern; ECB and Bank of Canada incremental lifts inconclusive", "uncertainty_or_gate": "not full provider-independent confirmation", "evidence_class": "adaptive external replication", "paper_role": "supporting/viva", "source_files": "prp1_study_a_publication_assessment.md", "permitted_claim": "External checks were mixed and did not independently confirm temporal lift.", "prohibited_claim": "The main finding is independently replicated."},
    ]


def _claim_rows() -> list[dict[str, Any]]:
    return [
        {"claim_id": "C01", "claim": "The corrected pooled Transformer discriminates the original multi-asset stress target.", "classification": "authoritative descriptive", "evidence_for": "raw pooled ROC-AUC 0.789814; PR-AUC 0.421056", "evidence_against": "static priors are stronger; test is adaptive", "adjudication": "Use only as pipeline context, not model superiority.", "main_source": "ifddrp_authoritative_metric_registry.csv"},
        {"claim_id": "C02", "claim": "The Transformer outperforms strong static baselines.", "classification": "rejected", "evidence_for": "none on authoritative pooled ranking", "evidence_against": "asset prior 0.823906 and family prior 0.816549 exceed 0.789814", "adjudication": "Explicitly state the opposite.", "main_source": "ifddrp_identity_dynamic_information_decomposition.csv"},
        {"claim_id": "C03", "claim": "The Transformer learned useful within-asset temporal ranking.", "classification": "rejected", "evidence_for": "some non-selected or validation rows exceed 0.5", "evidence_against": "authoritative within AUC 0.491638; no chronology gate passes", "adjudication": "No demonstrated within-asset temporal skill.", "main_source": "prp1_fixed_cross_model_temporal_skill_gates.csv"},
        {"claim_id": "C04", "claim": "Explicit asset identity materially affects pooled Transformer ranking.", "classification": "supported post-hoc", "evidence_for": "no-ID pooled AUC 0.715477; ID probe 0.166930", "evidence_against": "ablation does not isolate all implicit identity fingerprints", "adjudication": "Use as bounded shortcut evidence.", "main_source": "ifddrp_identity_dynamic_information_decomposition.csv"},
        {"claim_id": "C05", "claim": "The tested Transformer requires chronological order.", "classification": "rejected", "evidence_for": "none under registered controls", "evidence_against": "reverse/permutation/shift barely change pooled ranking", "adjudication": "Report near invariance for this model and task only.", "main_source": "ifddrp_interpretability_sanity_checks.csv"},
        {"claim_id": "C06", "claim": "Cross-asset label-prior heterogeneity can inflate pooled ROC-AUC without dynamics.", "classification": "supported mechanism", "evidence_for": "1,040-run registered simulation; all five gates pass", "evidence_against": "stylised simulation is not an empirical causal estimate", "adjudication": "Main methodological evidence with scope caveat.", "main_source": "prp1_study_a_independent_simulation_gate_inference.csv"},
        {"claim_id": "C07", "claim": "Within-asset evaluation recovers genuine signal when genuine signal exists.", "classification": "supported in simulation", "evidence_for": "strong-signal within AUC 0.783997; no-signal 0.500086", "evidence_against": "not recovered robustly in the historical panel", "adjudication": "Method validation, not empirical positive result.", "main_source": "prp1_study_a_independent_simulation_gate_inference.csv"},
        {"claim_id": "C08", "claim": "Within-asset-aligned training recovered dissertation-worthy temporal skill.", "classification": "rejected", "evidence_for": "selected point estimate 0.561924", "evidence_against": "interval crosses 0.5; paired lift and order gates fail", "adjudication": "Valid negative; retain as robustness.", "main_source": "ifddrp_within_asset_objective_verdict.md"},
        {"claim_id": "C09", "claim": "Continuous downside forecasting rescued the Transformer result.", "classification": "rejected", "evidence_for": "positive Spearman 0.255642", "evidence_against": "MAE is much worse than ridge and asset mean; no order sensitivity", "adjudication": "Primary metric governs; no rescue.", "main_source": "ifddrp_continuous_downside_verdict.md"},
        {"claim_id": "C10", "claim": "Macro/context inputs are the causal economic explanation.", "classification": "unsafe", "evidence_for": "largest mean zero-occlusion response 0.160730", "evidence_against": "current-vintage provenance limit; attribution sanity limits; perturbation is not causal", "adjudication": "Describe sensitivity only.", "main_source": "ifddrp_transformer_interpretability_verdict.md"},
        {"claim_id": "C11", "claim": "Attention weights explain the forecast.", "classification": "unsafe", "evidence_for": "attention diagnostics exist", "evidence_against": "attention is retained only as supplementary diagnostic", "adjudication": "Exclude explanatory language.", "main_source": "ifddrp_transformer_interpretability_verdict.md"},
        {"claim_id": "C12", "claim": "Emergent predictive market dynamics were discovered.", "classification": "not supported literally", "evidence_for": "state and identity information are decodable", "evidence_against": "no within-asset/order gate; 0 regimes pass", "adjudication": "Interpret title as discovering that apparent dynamics were mainly static shortcut structure.", "main_source": "ifddrp_title_alignment_verdict.md"},
        {"claim_id": "C13", "claim": "The combined falsification framework is unprecedented.", "classification": "unsafe novelty", "evidence_for": "closest-work audit found no single comparator with the full bundle", "evidence_against": "individual components are established and search is not exhaustive", "adjudication": "Call the combination apparently distinctive or underexplored, never first-ever.", "main_source": "ifddrp_novelty_claims_register.md"},
    ]


def _protocol_deviations() -> list[dict[str, Any]]:
    return [
        {"deviation_id": "D01", "stage": "historical programme", "deviation": "The final historical test was inspected repeatedly across phases.", "timing": "known before final bounded protocol", "severity": "major", "effect": "All final historical results are adaptive, not independent confirmation.", "resolution": "Preserve labels; require external/prospective confirmation for strong claims.", "status": "intrinsic_unresolved"},
        {"deviation_id": "D02", "stage": "Phase 6 reporting", "deviation": "Validation-calibrated scores were historically used for some ROC/PR and grouped-ranking summaries.", "timing": "found in final reconciliation", "severity": "moderate", "effect": "Tie-inducing isotonic calibration caused small ranking-metric discrepancies.", "resolution": "Raw scores are authoritative for ranking; calibrated scores for proper/decision metrics.", "status": "superseded_reporting"},
        {"deviation_id": "D03", "stage": "pre-Phase 6 split", "deviation": "A purge of 10 global dates left 180 boundary-crossing labels.", "timing": "historical", "severity": "major", "effect": "Threshold metrics were affected; pooled ranking changed little.", "resolution": "Retrained with measured purge 18 and one-date embargo.", "status": "corrected"},
        {"deviation_id": "D04", "stage": "pre-Phase 6 panel", "deviation": "Union-calendar placeholders contaminated eligibility and historical coverage.", "timing": "historical", "severity": "critical", "effect": "Incorrect windows and label coverage.", "resolution": "Observed-session construction enforced; corrected Transformer is authoritative.", "status": "corrected"},
        {"deviation_id": "D05", "stage": "final bounded protocol", "deviation": "Protocol amendment 1 clarified zeroed no-ID channels, pair registry, per-asset price source and raw ranking scores.", "timing": "pre-result after implementation audits", "severity": "moderate", "effect": "Made executable contract explicit without result-informed tuning.", "resolution": "Versioned amendment retained.", "status": "documented"},
        {"deviation_id": "D06", "stage": "Experiment B", "deviation": "Four validation assets lacked both-class pair support.", "timing": "execution", "severity": "moderate", "effect": "Within-asset pair metric covers 75 validation assets rather than all 79.", "resolution": "Pair support reported; no synthetic pairs or eligibility relaxation.", "status": "reported"},
        {"deviation_id": "D07", "stage": "Experiment C", "deviation": "The continuous outcome is maximum origin-to-path loss, not peak-to-trough drawdown.", "timing": "implementation audit", "severity": "moderate", "effect": "Economic interpretation is narrower than the word drawdown can imply.", "resolution": "Use exact maximum adverse total-return movement wording.", "status": "terminology_corrected"},
        {"deviation_id": "D08", "stage": "Experiment C", "deviation": "The ridge sequence-summary control uses registered summary/lag features rather than a fully flattened sequence.", "timing": "protocol execution", "severity": "minor", "effect": "Comparator label must remain precise.", "resolution": "Call it ridge sequence-summary control.", "status": "reported"},
        {"deviation_id": "D09", "stage": "Experiments A/B", "deviation": "No-ID fixed-width neural arms retain 12 immutable zero channels.", "timing": "pre-result audit", "severity": "minor", "effect": "Parameter shapes remain comparable; channels carry no identity information.", "resolution": "Document and test zero channels.", "status": "documented"},
        {"deviation_id": "D10", "stage": "final interpretability", "deviation": "Initial rerun inherited Phase 1 groups instead of the fixed final feature groups.", "timing": "found before final evidence freeze", "severity": "moderate", "effect": "Group labels and allocations were inconsistent, but checkpoints and predictions were unchanged.", "resolution": "Explicit exhaustive 34-feature partition; checkpoint-only rerun; updated reports.", "status": "corrected"},
        {"deviation_id": "D11", "stage": "macro inputs", "deviation": "Seven FRED inputs are current-vintage/provenance-limited rather than fully ALFRED-vintage reconstructed.", "timing": "known", "severity": "major", "effect": "Positive macro attribution or causal claims are inadmissible.", "resolution": "Retain only as sensitivity; require point-in-time replication.", "status": "implementation_limited"},
        {"deviation_id": "D12", "stage": "simulation", "deviation": "Simulation remains stylised despite independent extension with common shocks and dependence controls.", "timing": "design", "severity": "moderate", "effect": "Shows mechanism sufficiency, not empirical prevalence or causality.", "resolution": "Constrain claim language and seek external calibration.", "status": "caveated"},
        {"deviation_id": "D13", "stage": "fixed cross-model panel", "deviation": "The smoke manifest did not exercise every arm before full execution.", "timing": "post-execution audit", "severity": "moderate", "effect": "Weakens process assurance but did not change the negative scientific verdict.", "resolution": "Post-execution artifact and summary audit completed.", "status": "mitigated"},
        {"deviation_id": "D14", "stage": "all ten-session targets", "deviation": "Neighbouring origins have overlapping outcome windows.", "timing": "by design", "severity": "major", "effect": "Naive row-wise uncertainty would be anti-conservative.", "resolution": "Use date-block inference and non-overlapping-origin sensitivity; do not treat rows as independent.", "status": "mitigated"},
    ]


def _closest_work_rows() -> list[dict[str, Any]]:
    na = "not reported in accessible source summary"
    return [
        {"work_id": "A01", "type": "academic", "citation_or_repository": "Montero-Manso and Hyndman (2021), Principles and algorithms for forecasting groups of time series", "year": 2021, "dataset": "multiple groups of forecasting series", "asset_count": "not a financial asset panel", "frequency": "varied", "target": "series forecasting", "model": "local and global forecasting framework", "lookback": na, "split": "benchmark-specific", "metric": "forecast error", "identity_handling": "formalises local/global series grouping", "pooled_grouped_evaluation": "global versus local", "static_prior_control": "no", "temporal_order_control": "no", "direct_difference": "Closest theory of pooled/global learning, but not financial classification or identity-shortcut falsification.", "source_url": "https://doi.org/10.1016/j.ijforecast.2021.03.004", "novelty_relation": "established global/local foundation"},
        {"work_id": "A02", "type": "academic", "citation_or_repository": "Salinas et al. (2020), DeepAR", "year": 2020, "dataset": "several real-world forecasting datasets", "asset_count": "many related series", "frequency": "varied", "target": "probabilistic future values", "model": "autoregressive recurrent network", "lookback": "model/dataset-specific", "split": "out-of-sample benchmark", "metric": "probabilistic forecast accuracy", "identity_handling": "related-series global model; static covariates supported in implementations", "pooled_grouped_evaluation": "global forecasting", "static_prior_control": "no comparable classification prior", "temporal_order_control": "no", "direct_difference": "Establishes global sequence modelling, not pooled ROC shortcut analysis.", "source_url": "https://doi.org/10.1016/j.ijforecast.2019.07.001", "novelty_relation": "established global neural forecasting"},
        {"work_id": "A03", "type": "academic", "citation_or_repository": "Lim et al. (2021), Temporal Fusion Transformers", "year": 2021, "dataset": "multi-horizon benchmark datasets", "asset_count": "not a financial multi-asset panel", "frequency": "varied", "target": "multi-horizon regression/quantiles", "model": "TFT", "lookback": "dataset-specific", "split": "temporal benchmark splits", "metric": "quantile loss and forecast error", "identity_handling": "static covariate encoders", "pooled_grouped_evaluation": "dataset-specific", "static_prior_control": "no", "temporal_order_control": "no", "direct_difference": "Interpretable forecasting architecture; does not test entity-prior ROC inflation or order destruction.", "source_url": "https://doi.org/10.1016/j.ijforecast.2021.03.012", "novelty_relation": "established interpretability architecture"},
        {"work_id": "A04", "type": "academic", "citation_or_repository": "Nie et al. (2023), PatchTST", "year": 2023, "dataset": "ETT, electricity, traffic, weather and related LTSF benchmarks", "asset_count": "multivariate channels", "frequency": "benchmark-specific", "target": "long-horizon values", "model": "channel-independent patched Transformer", "lookback": "multiple, including long lookbacks", "split": "standard LTSF splits", "metric": "MSE/MAE", "identity_handling": "shared channel-independent weights", "pooled_grouped_evaluation": "channel-wise benchmark aggregation", "static_prior_control": "no", "temporal_order_control": "no registered destruction bundle", "direct_difference": "Tests tokenisation and long context, not asset identity and within-asset discrimination.", "source_url": "https://openreview.net/pdf/2e4e6db8733d24f382a7e57c9b3d53d7e0061ade.pdf", "novelty_relation": "established Transformer comparator"},
        {"work_id": "A05", "type": "academic", "citation_or_repository": "Liu et al. (2024), iTransformer", "year": 2024, "dataset": "standard multivariate forecasting benchmarks", "asset_count": "hundreds of variates on some datasets", "frequency": "benchmark-specific", "target": "future values", "model": "inverted Transformer", "lookback": "arbitrary/multiple", "split": "standard benchmark splits", "metric": "MSE/MAE", "identity_handling": "variables are tokens; no financial asset-prior audit", "pooled_grouped_evaluation": "benchmark aggregate", "static_prior_control": "no", "temporal_order_control": "no registered financial order-control bundle", "direct_difference": "Varied tokenisation, but not the dissertation's cross-sectional-prior falsification question.", "source_url": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html", "novelty_relation": "established Transformer comparator"},
        {"work_id": "A06", "type": "academic", "citation_or_repository": "Zeng et al. (2023), Are Transformers Effective for Time Series Forecasting?", "year": 2023, "dataset": "nine real-life LTSF datasets", "asset_count": "multivariate channels", "frequency": "benchmark-specific", "target": "long-horizon values", "model": "DLinear/NLinear versus Transformers", "lookback": "benchmark-specific", "split": "standard LTSF splits", "metric": "MSE/MAE", "identity_handling": "not an entity-prior study", "pooled_grouped_evaluation": "dataset aggregate", "static_prior_control": "linear temporal baselines, not label priors", "temporal_order_control": "architecture studies; not the full registered bundle", "direct_difference": "Closest challenge to Transformer temporal extraction, but no heterogeneous multi-asset ROC mechanism.", "source_url": "https://doi.org/10.1609/aaai.v37i9.26317", "novelty_relation": "close methodological precedent"},
        {"work_id": "A07", "type": "academic", "citation_or_repository": "Li et al. (2024), MASTER", "year": 2024, "dataset": "CSI300 and CSI800 stock universes", "asset_count": "about 300/800 per date", "frequency": "daily", "target": "stock-return ranking", "model": "market-guided stock Transformer", "lookback": "8 days", "split": "chronological train/validation/test", "metric": "IC, RankIC and portfolio metrics", "identity_handling": "cross-stock and market information; no explicit static-prior falsification", "pooled_grouped_evaluation": "cross-sectional per-date ranking", "static_prior_control": "no", "temporal_order_control": "no", "direct_difference": "Closest financial Transformer, but evaluates cross-sectional ranking rather than within-asset timing; repository discloses validation/test processing issues.", "source_url": "https://doi.org/10.1609/aaai.v38i1.27767", "novelty_relation": "closest financial architecture comparator"},
        {"work_id": "A08", "type": "academic", "citation_or_repository": "Shao et al. (2022), Spatial-Temporal Identity", "year": 2022, "dataset": "PEMS04/07/08, PEMS-BAY, Electricity", "asset_count": "170 to 883 sensors/336 variables", "frequency": "5-minute or hourly", "target": "future traffic/electricity values", "model": "MLP with spatial and temporal identity embeddings", "lookback": "dataset-specific", "split": "standard chronological benchmark splits", "metric": "MAE/RMSE/MAPE", "identity_handling": "identity is an intentional predictive input with ablation", "pooled_grouped_evaluation": "aggregate over sensors", "static_prior_control": "no label-prior comparator", "temporal_order_control": "no", "direct_difference": "Shows identity can be useful; dissertation asks when identity inflates pooled discrimination without timing skill.", "source_url": "https://arxiv.org/abs/2208.05233", "novelty_relation": "closest identity-input precedent"},
        {"work_id": "A09", "type": "academic", "citation_or_repository": "Sirignano and Cont (2019), Universal features of price formation", "year": 2019, "dataset": "billions of US equity order-book quotes/trades", "asset_count": "many US stocks", "frequency": "high frequency", "target": "subsequent price movement", "model": "pooled deep neural network", "lookback": "many recent order-book observations", "split": "out-of-sample across stocks/times", "metric": "prediction accuracy", "identity_handling": "pooled universal versus stock-specific models", "pooled_grouped_evaluation": "cross-stock transfer", "static_prior_control": "not the dissertation's label prior", "temporal_order_control": "history-length/path-dependence analysis, no registered destruction bundle", "direct_difference": "Positive microstructure evidence for pooled dynamics; different information set and much higher frequency.", "source_url": "https://doi.org/10.1080/14697688.2019.1622295", "novelty_relation": "important counterpoint"},
        {"work_id": "A10", "type": "academic", "citation_or_repository": "Zhang et al. (2019), DeepLOB", "year": 2019, "dataset": "FI-2010 and LSE limit-order-book data", "asset_count": "FI-2010 has 5 stocks", "frequency": "event/high frequency", "target": "future mid-price movement classes", "model": "CNN-Inception-LSTM", "lookback": "100 order-book updates in FI-2010 setup", "split": "chronological benchmark protocol", "metric": "accuracy/precision/recall/F1", "identity_handling": "pooled benchmark; no static asset-prior audit", "pooled_grouped_evaluation": "benchmark aggregate", "static_prior_control": "no", "temporal_order_control": "no", "direct_difference": "Financial sequence prediction at microstructure scale, not heterogeneous daily pooled ROC analysis.", "source_url": "https://arxiv.org/abs/1808.03668", "novelty_relation": "financial deep-learning comparator"},
        {"work_id": "A11", "type": "academic", "citation_or_repository": "Janes and Pepe (2009), Covariate-adjusted ROC", "year": 2009, "dataset": "biomarker illustration and simulations", "asset_count": "not applicable", "frequency": "not time series", "target": "binary classification", "model": "covariate-adjusted ROC estimators", "lookback": "not applicable", "split": "statistical estimation", "metric": "covariate-adjusted ROC", "identity_handling": "explicit adjustment for covariates affecting marker distributions", "pooled_grouped_evaluation": "pooled versus covariate-adjusted discrimination", "static_prior_control": "conceptually related", "temporal_order_control": "not applicable", "direct_difference": "Statistical precedent that pooled ROC can be misleading under covariate heterogeneity; dissertation applies an adversarial financial sequence framework.", "source_url": "https://doi.org/10.1093/biomet/asp002", "novelty_relation": "closest metric-theory precedent"},
        {"work_id": "A12", "type": "academic", "citation_or_repository": "Geirhos et al. (2020), Shortcut learning in deep neural networks", "year": 2020, "dataset": "cross-domain perspective", "asset_count": "not applicable", "frequency": "not applicable", "target": "general prediction", "model": "deep networks", "lookback": "not applicable", "split": "challenging-distribution evaluation", "metric": "task-specific", "identity_handling": "shortcut features broadly defined", "pooled_grouped_evaluation": "not financial", "static_prior_control": "conceptual", "temporal_order_control": "not financial", "direct_difference": "Provides shortcut-learning theory; dissertation operationalises it for pooled financial forecasting.", "source_url": "https://doi.org/10.1038/s42256-020-00257-z", "novelty_relation": "established conceptual foundation"},
        {"work_id": "A13", "type": "academic", "citation_or_repository": "Hewitt and Liang (2019), Designing and Interpreting Probes with Control Tasks", "year": 2019, "dataset": "NLP representations", "asset_count": "not applicable", "frequency": "not applicable", "target": "representation probe tasks", "model": "linear/MLP probes", "lookback": "not applicable", "split": "train/test probe evaluation", "metric": "accuracy and selectivity", "identity_handling": "control tasks distinguish representation content from probe memorisation", "pooled_grouped_evaluation": "not financial", "static_prior_control": "control tasks", "temporal_order_control": "no", "direct_difference": "Methodological basis for cautious identity/state probes, not a financial forecasting study.", "source_url": "https://doi.org/10.18653/v1/D19-1275", "novelty_relation": "probe methodology foundation"},
        {"work_id": "A14", "type": "academic", "citation_or_repository": "Jain and Wallace (2019), Attention is not Explanation", "year": 2019, "dataset": "NLP classification tasks", "asset_count": "not applicable", "frequency": "not applicable", "target": "text classification", "model": "attention-based neural models", "lookback": "not applicable", "split": "task-specific", "metric": "prediction and explanation diagnostics", "identity_handling": "not applicable", "pooled_grouped_evaluation": "not applicable", "static_prior_control": "no", "temporal_order_control": "alternative attention distributions", "direct_difference": "Supports treating financial attention only as a diagnostic, not an explanation.", "source_url": "https://doi.org/10.18653/v1/N19-1357", "novelty_relation": "interpretability caveat"},
        {"work_id": "A15", "type": "academic", "citation_or_repository": "Cerqueira, Torgo and Mozetic (2019), Evaluating time-series forecasting models", "year": 2019, "dataset": "62 real and 3 synthetic time series", "asset_count": "65 series", "frequency": "varied", "target": "forecast values", "model": "multiple performance-estimation procedures", "lookback": "model-specific", "split": "cross-validation versus temporally ordered out-of-sample", "metric": "forecast loss estimation", "identity_handling": "not central", "pooled_grouped_evaluation": "across series", "static_prior_control": "no", "temporal_order_control": "evaluation preserves order", "direct_difference": "Supports chronology-preserving validation but does not address overlapping multi-asset labels or identity shortcuts.", "source_url": "https://arxiv.org/abs/1905.11744", "novelty_relation": "validation methodology foundation"},
    ]


def _selection_rows() -> list[dict[str, Any]]:
    return [
        {"selection_id": "S01", "rank": 1, "category": "A_main", "module": "Corrected Transformer forecasting pipeline", "result": "Raw pooled ROC-AUC 0.789814 on 21,514 test windows; calibrated Brier 0.110920", "title_relevance": "high", "validity": "moderate", "robustness": "high arithmetic; adaptive test", "interpretability": "context", "replication": "limited", "novelty": "low alone", "clarity": "high", "space_efficiency": "high", "paper_use": "Establish the real model and apparent headline performance before falsification."},
        {"selection_id": "S02", "rank": 2, "category": "A_main", "module": "Static-prior, within-asset and temporal-order falsification", "result": "Asset prior 0.823906 > Transformer 0.789814; within AUC 0.491638; negligible order effects", "title_relevance": "very high", "validity": "high within adaptive evidence class", "robustness": "high", "interpretability": "high", "replication": "partial", "novelty": "apparently distinctive combination", "clarity": "very high", "space_efficiency": "very high", "paper_use": "Central methodological result."},
        {"selection_id": "S03", "rank": 3, "category": "A_main", "module": "Controlled simulation and fixed cross-model recurrence", "result": "1,040 simulations pass all mechanism gates; 0/5 empirical model gates pass", "title_relevance": "high", "validity": "high for mechanism; adaptive empirical panel", "robustness": "high", "interpretability": "high", "replication": "simulation plus model-family recurrence", "novelty": "moderate", "clarity": "high", "space_efficiency": "high", "paper_use": "Separates metric mechanism from one-model accident."},
        {"selection_id": "S04", "rank": 4, "category": "B_supporting", "module": "Identity removal, swap and representation probes", "result": "No-ID AUC 0.715477; identity accuracy 0.166930 versus chance about 0.0127", "title_relevance": "very high", "validity": "post-hoc diagnostic", "robustness": "multi-control", "interpretability": "high but non-causal", "replication": "within project", "novelty": "moderate", "clarity": "high", "space_efficiency": "medium", "paper_use": "Supports shortcut interpretation."},
        {"selection_id": "S05", "rank": 5, "category": "B_supporting", "module": "Experiment A static plus dynamic residual", "result": "Gate failed; selected test within AUC 0.525360", "title_relevance": "high", "validity": "valid negative", "robustness": "bounded controls", "interpretability": "high", "replication": "none", "novelty": "moderate", "clarity": "medium", "space_efficiency": "medium", "paper_use": "One sentence/table robustness."},
        {"selection_id": "S06", "rank": 6, "category": "B_supporting", "module": "Experiment B within-asset objective", "result": "Gate failed; selected test within AUC 0.561924 with CI crossing 0.5", "title_relevance": "high", "validity": "valid negative", "robustness": "three seeds and controls", "interpretability": "medium", "replication": "none", "novelty": "moderate", "clarity": "medium", "space_efficiency": "medium", "paper_use": "Shows a direct recovery attempt was inconclusive."},
        {"selection_id": "S07", "rank": 7, "category": "B_supporting", "module": "Experiment C continuous downside", "result": "Gate failed; Transformer MAE materially worse than simple controls", "title_relevance": "medium", "validity": "valid negative", "robustness": "block/subperiod/order checks", "interpretability": "medium", "replication": "none", "novelty": "low", "clarity": "high", "space_efficiency": "medium", "paper_use": "Rules out target-binarisation as the sole explanation."},
        {"selection_id": "S08", "rank": 8, "category": "C_one_line", "module": "Bounded regime analysis", "result": "0/14 regimes pass; latent analysis locked", "title_relevance": "high", "validity": "post-hoc", "robustness": "strict gate", "interpretability": "medium", "replication": "none", "novelty": "low", "clarity": "high", "space_efficiency": "very high", "paper_use": "One sentence negative."},
        {"selection_id": "S09", "rank": 9, "category": "D_viva", "module": "Interpretability attribution suite", "result": "Macro/context and asset identity perturbations largest; one randomisation sanity condition fails", "title_relevance": "very high", "validity": "limited diagnostic", "robustness": "multiple methods/seeds", "interpretability": "bounded", "replication": "none", "novelty": "low", "clarity": "medium", "space_efficiency": "low", "paper_use": "Viva/supplement; no causal ranking."},
        {"selection_id": "S10", "rank": 10, "category": "D_viva", "module": "External checks", "result": "Japan partial shortcut pattern; ECB/Bank of Canada temporal lifts inconclusive", "title_relevance": "medium", "validity": "mixed", "robustness": "adversarial", "interpretability": "medium", "replication": "partial/inconclusive", "novelty": "moderate", "clarity": "low", "space_efficiency": "low", "paper_use": "Viva and limitations."},
        {"selection_id": "S11", "rank": 11, "category": "E_excluded", "module": "Original Phase 2D stress headline", "result": "Superseded by static-prior and within-asset falsification", "title_relevance": "high", "validity": "insufficient for headline", "robustness": "superseded", "interpretability": "low", "replication": "none", "novelty": "low", "clarity": "misleading alone", "space_efficiency": "low", "paper_use": "Do not present as positive model superiority."},
        {"selection_id": "S12", "rank": 12, "category": "E_excluded", "module": "Attention as explanation", "result": "Diagnostic only", "title_relevance": "medium", "validity": "inadmissible explanation", "robustness": "fails explanatory standard", "interpretability": "unsafe", "replication": "none", "novelty": "none", "clarity": "misleading", "space_efficiency": "low", "paper_use": "Exclude explanatory claim."},
    ]


def _build_markdown_reports(state: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    conditioned = state["conditioned"]
    no_id = state["no_id"]
    manifest = state["manifest"]
    source_commit = state["source_commit"]
    split = manifest["split"]

    audit_summary = [
        {"check": "Prediction reconstruction", "finding": "Exact for 3 conditioned and 3 no-ID checkpoints", "status": "pass"},
        {"check": "Ranking metric basis", "finding": "Raw ensemble scores are authoritative", "status": "reconciled"},
        {"check": "Probability metric basis", "finding": "Validation-selected isotonic scores govern Brier/log loss and thresholds", "status": "reconciled"},
        {"check": "Stress target", "finding": "t+1 through t+10 OR of terminal return, minimum path loss and volatility spike", "status": "pass_with_comparability_limit"},
        {"check": "Split", "finding": "Fold 3, purge 18 global dates, embargo 1; no measured boundary crossing", "status": "pass"},
        {"check": "Scaling", "finding": "Per-asset feature scaling fit on training endpoints only", "status": "pass"},
        {"check": "Final recovery experiments", "finding": "A, B and C all failed preregistered promotion gates", "status": "valid_negative"},
        {"check": "Regime/latent gate", "finding": "0 of 14 regimes pass; latent change-point work locked", "status": "pass"},
    ]
    _write_markdown(
        "ifddrp_final_scientific_audit.md",
        f"""
# IFDDRP Final Scientific Audit

Freeze date: {FREEZE_DATE}. Evidence-producing source commit: `{source_commit}`.

## Verdict

The central numbers are reproducible from preserved predictions. One reporting convention required reconciliation: raw ensemble probabilities are authoritative for ROC-AUC, PR-AUC and within-asset ranking, while validation-selected calibrated probabilities are authoritative for Brier score, log loss and thresholded decisions. Isotonic ties explain the small historical differences (`0.7891/0.4982` versus raw `{conditioned['raw_roc_auc']:.6f}/{conditioned['pair_weighted_within_auc']:.6f}`). No model rerun is required.

{_markdown_table(audit_summary)}

## Authoritative Transformer contract

- Universe: 80 configured daily instruments across six families; 79 are represented in the corrected split windows.
- Data: observed OHLCV sessions, not a forward-filled union calendar. Adjusted Close is preferred for targets; all 80 final target series had complete adjusted prices, so row-level Close fallback was not exercised.
- Origin and horizon: 60 observed sessions through market close at t; `target_stress_10d` uses only t+1 through t+10.
- Target: one when terminal ten-session return is at most -5%, the minimum future path relative to origin is at most -7%, or future realised volatility is at least twice trailing 20-session volatility scaled to ten sessions.
- Features: 34 train-scaled inputs (27 technical and 7 FRED macro/context series) plus 12 learned asset-embedding channels repeated over time. Raw OHLCV, family identity and missingness indicators are not direct inputs.
- Model: hidden width 128, two encoder layers, four heads, feed-forward width 256, sinusoidal positions to 1,024, temporal-attention pooling, LayerNorm/dropout 0.3/linear logit; 272,449 parameters.
- Training: soft-F1 loss, AdamW, learning rate 3e-4, weight decay 1e-4, batch 1,024, at most 12 epochs, seeds 7/42/123, validation-only early stopping/calibration/threshold selection.
- Split: train `{split['train_start']}` to `{split['train_end']}`, validation `{split['validation_start']}` to `{split['validation_end']}`, test `{split['test_start']}` to `{split['test_end']}`; purge 18, embargo 1. Window counts are 245,055/20,494/21,514.

## Reconciled central metrics

- Conditioned Transformer raw ranking: pooled ROC-AUC `{conditioned['raw_roc_auc']:.6f}`, PR-AUC `{conditioned['raw_pr_auc']:.6f}`, pair-weighted within-asset ROC-AUC `{conditioned['pair_weighted_within_auc']:.6f}`.
- Conditioned calibrated probabilities: Brier `{conditioned['calibrated_brier']:.6f}`, log loss `{conditioned['calibrated_log_loss']:.6f}`.
- No-ID raw ranking: pooled ROC-AUC `{no_id['raw_roc_auc']:.6f}`, within-asset ROC-AUC `{no_id['pair_weighted_within_auc']:.6f}`.
- Training-only priors: asset ROC-AUC `0.823906`; family ROC-AUC `0.816549`. Both exceed the learned Transformer in pooled ranking.
- Fixed cross-model panel: best learned pooled model was the MLP at `0.796478`; best learned within-asset point estimate was the MLP at `0.556967`; zero of five strict temporal-skill gates passed.

## Final bounded experiments

Experiment A selected an MLP family-prior residual on validation. Its test within-asset AUC was `0.525360`, but the paired lift and chronology conditions failed. Experiment B selected the Transformer BCE-plus-pairwise objective. Its test within-asset AUC was `0.561924` with interval `[0.492527, 0.625022]`; paired lift and chronology again failed. Experiment C selected the no-ID Transformer, whose equal-asset MAE `0.037816` was materially worse than the training asset mean `0.020896` and ridge `0.020691`; order effects were not established. All three are valid negatives.

## Interpretation and title

Exact checkpoint reconstruction passed. Mean zero-occlusion response was largest for macro/context (`0.160730`), followed by asset-embedding removal (`0.136454`) and returns/momentum (`0.129481`). The largest lag-block response was distant days 41-60 (`0.141292`). These are sensitivity diagnostics, not causal importance. A trained-versus-randomised attribution rank control failed; attention is supplementary only. Asset identity is materially decodable, but the head does not demonstrate useful within-asset timing. No regime passed the dynamic gate.

The fixed title is defensible only as an adversarial discovery: the apparent emergent dynamics mainly reflected static shortcut structure. It is not evidence that predictive emergent temporal dynamics were recovered.

## Defects and rerun decision

The purge-10 and union-calendar defects are historically preserved and superseded by the corrected run. Macro/context attribution is provenance-limited because the final seven FRED inputs are not a complete point-in-time vintage reconstruction. The test is repeatedly opened and all final historical results remain adaptive. The initial final-interpretability grouping mismatch was corrected with an exhaustive fixed 34-feature partition and a checkpoint-only rerun. No material defect remains that requires model retraining. Independent or prospective confirmation, not another historical architecture search, is the scientifically justified next empirical action.
""",
    )

    reuse_rows = [
        {"evidence": "Corrected daily panel", "use": "Phase 6 and final A/B/C", "independence": "same opened development panel", "allowed interpretation": "developmental/adaptive"},
        {"evidence": "Phase 6 test endpoints", "use": "central falsification and final interpretation", "independence": "reused repeatedly", "allowed interpretation": "historical held-out but adaptive"},
        {"evidence": "Phase 6 predictions/checkpoints", "use": "arithmetic reconstruction, probes, occlusion", "independence": "same trained models", "allowed interpretation": "post-hoc robustness"},
        {"evidence": "Experiments A/B/C", "use": "bounded recovery attempts", "independence": "protocol frozen, but same opened test", "allowed interpretation": "valid negative/adaptive"},
        {"evidence": "Independent simulation", "use": "mechanism identification", "independence": "new simulated samples under registered design", "allowed interpretation": "simulation only"},
        {"evidence": "Japan Yahoo panel", "use": "external-market shortcut check", "independence": "new geography, overlapping provider", "allowed interpretation": "partial replication"},
        {"evidence": "ECB and Bank of Canada FX", "use": "external temporal-skill checks", "independence": "official external providers", "allowed interpretation": "inconclusive adaptive replication"},
        {"evidence": "FRED macro/context", "use": "seven model inputs and attribution", "independence": "current-vintage, not fully point-in-time", "allowed interpretation": "provenance-limited diagnostic"},
    ]
    _write_markdown(
        "ifddrp_final_data_reuse_audit.md",
        f"""
# IFDDRP Final Data-Reuse Audit

{_markdown_table(reuse_rows)}

The corrected Phase 6 panel is the authoritative implementation but not an independent confirmation set. Experiments A-C were legitimately protocol-bounded before their own results, yet their final test shares the already-opened historical period. This distinction prevents the words independent, fresh, confirmed or prospective from being attached to those findings.

Raw and processed panels, predictions and checkpoints remain local ignored artefacts. Compact derived metrics and reports may be committed. No later result retroactively changes the evidence class of an earlier result.
""",
    )

    _write_markdown(
        "ifddrp_final_evidence_freeze.md",
        f"""
# IFDDRP Final Evidence Freeze

Frozen on {FREEZE_DATE} from evidence-producing source commit `{source_commit}`. This freeze supersedes reporting ambiguities but does not overwrite historical files.

## Three principal contributions

1. **A real corrected Transformer forecasting pipeline.** A 272,449-parameter Transformer used 60 observed sessions, 34 features and a 12-channel asset embedding to forecast ten-session stress across a broad daily panel. Its raw pooled ROC-AUC was `{conditioned['raw_roc_auc']:.6f}`.
2. **An adversarial shortcut diagnosis.** The training-only asset prior (`0.823906`) and family prior (`0.816549`) exceeded the Transformer; the Transformer's within-asset AUC was `{conditioned['pair_weighted_within_auc']:.6f}`; identity was decodable; removing identity reduced pooled ranking; and registered order destruction barely changed pooled ranking.
3. **Mechanism and recurrence evidence.** A 1,040-run controlled simulation showed how target-prior heterogeneity can inflate pooled AUC while within-asset evaluation remains at chance in no-signal settings and recovers planted dynamics. Across logistic, MLP, LSTM, TCN and Transformer models, no strict empirical temporal-skill gate passed.

## Evidence classes

- Main paper: the three contributions above.
- Major supporting: identity ablation/swap/probes and bounded Experiments A-C as valid negatives.
- One-line robustness: zero of 14 regimes passed; latent-state analysis remained locked.
- Viva/supplement: exact model specification, target/purge history, endpoint manifests, feature and lag diagnostics, external checks and protocol deviations.
- Excluded/superseded: Transformer superiority, literal discovery of predictive emergent dynamics, attention as explanation, causal macro attribution, trading/monitoring claims, non-selected test winners and calibrated-score AUC as ranking authority.

## Scientific stopping rule

No historical model fairly beat the static asset prior, no model passed the chronology gate, and none of the three final bounded recovery experiments passed. Additional historical model search would increase adaptivity without supplying independent evidence. The empirical modelling programme is frozen; the next action is dissertation writing from this registry, followed only by independently preregistered replication if the research continues.
""",
    )

    _write_markdown(
        "ifddrp_final_main_paper_evidence.md",
        """
# IFDDRP Final Main-Paper Evidence

## Contribution 1: corrected Transformer pipeline

Report the exact observed-session pipeline, 60-session lookback, ten-session stress target, 34 inputs plus asset embedding, corrected purge/embargo, three-seed training and raw/calibrated metric distinction. The pooled ROC-AUC establishes why the model initially looked successful; it is not a superiority claim.

## Contribution 2: pooled performance versus genuine timing

Make the static asset/family priors, pair-weighted within-asset ROC-AUC, no-ID ablation and three temporal-order controls the central result. The clean claim is that the tested pooled metric mainly measured cross-sectional structure and did not establish useful within-asset chronology.

## Contribution 3: mechanism and cross-model recurrence

Use the registered simulation to demonstrate that heterogeneous label priors are sufficient to inflate pooled AUC and that within-asset/order-sensitive metrics recover planted dynamic signal. Pair it with the bounded five-model panel, where no strict temporal-skill gate passed. This is stronger than a Transformer-only negative.

Recommended compact presentation: one pipeline/target diagram, one pooled-versus-within/static table, one order-control figure, and one simulation mechanism figure. Experiments A-C belong in a compact robustness table.
""",
    )

    _write_markdown(
        "ifddrp_final_supporting_evidence.md",
        """
# IFDDRP Final Supporting Evidence

- **Experiment A:** explicit static-prior plus dynamic residual did not pass paired-lift or chronology gates.
- **Experiment B:** within-asset pairwise training produced a descriptive point improvement but its test interval crossed chance and the paired/chronology gates failed.
- **Experiment C:** changing to a continuous downside outcome did not rescue neural forecasting; ridge and training asset means had substantially lower MAE.
- **Identity diagnostics:** identity removal, swapping and representation probes consistently show that asset identity affects pooled outputs.
- **Interpretability:** macro/context, asset identity and returns/momentum perturbations move scores, but explanation sanity and provenance limits prohibit causal feature claims.
- **Regimes:** none of 14 fixed regimes passed the strict dynamic gate; deeper latent-state analysis was correctly stopped.
- **External checks:** Japan partly reproduced the shortcut pattern; official ECB and Bank of Canada studies did not establish incremental temporal lift. They do not amount to independent confirmation.
""",
    )

    _write_markdown(
        "ifddrp_final_viva_evidence.md",
        """
# IFDDRP Final Viva Evidence

Use the following evidence to answer implementation and scientific-governance questions:

1. The authoritative model specification and 34-row input registry, including the absence of direct family/missingness channels.
2. The Phase 6 run manifest and prediction reconstruction for all six final checkpoints.
3. The measured purge correction: purge 10 allowed 180 crossing labels; purge 18 removed measured crossings.
4. The observed-session eligibility correction and preserved historical union-calendar defect.
5. Raw-versus-calibrated metric reconciliation, including why isotonic ties alter AUC slightly.
6. Pair-support audit for within-asset objectives and explicit exclusion of four unsupported validation assets.
7. The exact continuous outcome definition and distinction from peak-to-trough drawdown.
8. Attribution sanity controls, parameter randomisation failure and attention caveat.
9. Protocol freeze, amendment, execution manifests and all negative stopping decisions.
10. External replication limitations and the prospective-confirmation path.
""",
    )

    _write_markdown(
        "ifddrp_final_excluded_evidence.md",
        """
# IFDDRP Final Excluded or Superseded Evidence

- Do not headline Phase 2D stress F1 as Transformer superiority; later static and grouped controls supersede that interpretation.
- Do not use validation-calibrated/isotonic probabilities as the authoritative source for ROC-AUC, PR-AUC or within-asset ranking. Retain them for Brier, log loss and thresholded decisions.
- Do not promote non-selected test rows from Experiments A-C.
- Do not claim that within-asset pairwise training recovered temporal skill; its promotion gate failed.
- Do not claim that continuous downside risk is practically forecastable by the tested neural models.
- Do not interpret attention weights, Integrated Gradients or occlusion as causal explanations.
- Do not promote macro/context sensitivity because the FRED inputs are current-vintage/provenance-limited.
- Do not claim a stable dynamic regime, latent transition or predictive emergent state; the gate remained locked.
- Do not claim trading profitability, market monitoring utility, causal lead-lag structure or independent confirmation.
- Do not use superseded purge-10 or union-calendar results as authoritative implementation evidence.
""",
    )

    title_rows = [
        {"title_component": "Interpretable", "support": "partial", "evidence": "bounded occlusion, lag analysis, probes, identity ablation and sanity controls", "limit": "one randomisation control failed; sensitivity is not causality"},
        {"title_component": "Transformer models", "support": "strong", "evidence": "real trained 272,449-parameter encoder reconstructed exactly across three seeds", "limit": "Transformer is not superior to static priors"},
        {"title_component": "Financial time-series forecasting", "support": "strong", "evidence": "80-instrument daily panel, leakage-corrected chronological design and future stress/downside outcomes", "limit": "historical test is adaptive and target comparability is imperfect"},
        {"title_component": "Discovering emergent market dynamics", "support": "adversarial/partial", "evidence": "the study discovered that apparent pooled dynamics can emerge from static cross-sectional priors", "limit": "no useful predictive emergent temporal dynamics were established"},
    ]
    _write_markdown(
        "ifddrp_final_title_support_assessment.md",
        f"""
# IFDDRP Final Title-Support Assessment

Fixed title: **Interpretable Transformer Models for Financial Time Series Forecasting: Discovering Emergent Market Dynamics**

{_markdown_table(title_rows)}

The defensible reading is interpretation 3: apparent dynamics mainly reflected static shortcut structure, with partial support for interpretation 4. The title should be explained as a research question tested adversarially, not as a presupposed positive finding.
""",
    )

    _write_markdown(
        "ifddrp_novelty_claims_register.md",
        """
# IFDDRP Novelty Claims Register

## Defensible

- The combination of training-only global/family/asset priors, pooled and within-asset metrics, identity removal/swapping/probing, endpoint-preserving order destruction, controlled target-heterogeneity simulation and bounded recovery experiments is **apparently distinctive and underexplored** in the closest reviewed work.
- The contribution is primarily an adversarial evaluation framework for heterogeneous multi-asset forecasting, not a new Transformer architecture.
- The simulation plus empirical falsification provides a clear mechanism for why pooled ROC-AUC can overstate temporal forecasting skill.

## Established components

Global forecasting, financial sequence models, entity embeddings, covariate-adjusted ROC analysis, shortcut learning, representation probes, attention caveats and chronology-preserving validation all have strong prior literature. Each must be cited as precedent.

## Unsafe

Do not write "first", "never studied", "unprecedented" or "proves". The literature review is bounded, and individual elements of the framework are established. The safe novelty statement is that no reviewed paper or implementation combined the complete falsification bundle for a heterogeneous daily financial panel.
""",
    )

    oss_rows = [
        {"implementation": "PatchTST", "scope": "patched channel-independent Transformer", "reproducibility": "official Apache-2.0 repository", "closest_use": "architecture comparator", "missing_from_dissertation_question": "static label priors, within-asset ROC, identity swap/probe and full order controls", "url": "https://github.com/yuqinie98/PatchTST"},
        {"implementation": "iTransformer", "scope": "variate-token Transformer", "reproducibility": "official MIT repository", "closest_use": "alternative multivariate tokenisation", "missing_from_dissertation_question": "financial target-prior and grouped-ranking falsification", "url": "https://github.com/thuml/iTransformer"},
        {"implementation": "DLinear", "scope": "simple linear LTSF baselines", "reproducibility": "official repository", "closest_use": "strong simplicity control", "missing_from_dissertation_question": "identity/static-prior mechanism and financial grouped evaluation", "url": "https://github.com/honeywell21/DLinear"},
        {"implementation": "MASTER", "scope": "market-guided stock Transformer", "reproducibility": "official code plus data caveats", "closest_use": "closest financial Transformer", "missing_from_dissertation_question": "within-asset timing and static-prior/order falsification; repo discloses validation/test processing issue", "url": "https://github.com/SJTU-DMTai/MASTER"},
        {"implementation": "STID", "scope": "MLP with spatial/temporal identity", "reproducibility": "official CIKM repository", "closest_use": "identity-input precedent", "missing_from_dissertation_question": "tests identity as beneficial signal rather than harmful pooled shortcut", "url": "https://github.com/GestaltCogTeam/STID"},
        {"implementation": "GluonTS", "scope": "probabilistic global forecasting library", "reproducibility": "mature Apache-2.0 package", "closest_use": "global model and probabilistic baseline framework", "missing_from_dissertation_question": "no turnkey financial shortcut audit", "url": "https://github.com/awslabs/gluonts"},
        {"implementation": "NeuralForecast", "scope": "30+ neural forecasting models", "reproducibility": "active Apache-2.0 package", "closest_use": "model implementations and fair baseline APIs", "missing_from_dissertation_question": "generic metrics do not supply the complete identity/order/static-prior bundle", "url": "https://github.com/Nixtla/neuralforecast"},
        {"implementation": "PyTorch Forecasting", "scope": "TFT and neural forecasting utilities", "reproducibility": "active open-source package", "closest_use": "TFT implementation", "missing_from_dissertation_question": "no bespoke multi-asset shortcut protocol", "url": "https://github.com/sktime/pytorch-forecasting"},
        {"implementation": "TimesFM", "scope": "pretrained time-series foundation model", "reproducibility": "official Google Research repository", "closest_use": "future zero-shot comparator", "missing_from_dissertation_question": "not justified on current opened panel without a new frozen protocol", "url": "https://github.com/google-research/timesfm"},
        {"implementation": "Time-Series-Library", "scope": "broad advanced-model benchmark library", "reproducibility": "official research library", "closest_use": "reference implementations", "missing_from_dissertation_question": "leaderboard breadth is not the scientific objective; project notes benchmark limitations", "url": "https://github.com/thuml/Time-Series-Library"},
    ]
    _write_markdown(
        "ifddrp_open_source_comparator_review.md",
        f"""
# IFDDRP Open-Source Comparator Review

{_markdown_table(oss_rows)}

These projects provide credible architecture and forecasting implementations. None supplies the complete dissertation protocol: training-only global/family/asset priors, raw pooled plus pair-weighted within-asset ranking, identity removal/swap/probes, three endpoint-preserving sequence controls, and a controlled target-heterogeneity simulation. That absence supports an "apparently distinctive combination" claim, not first-ever novelty.

MASTER is the closest financial Transformer implementation, but its repository documents validation/test processing and data-reproduction caveats. Foundation-model or broader architecture comparisons would require a new preregistered dataset and are not justified as additions to the frozen historical dissertation evidence.
""",
    )

    _write_markdown(
        "ifddrp_final_resume_state.md",
        f"""
# IFDDRP Final Resume State

- Freeze date: {FREEZE_DATE}.
- Evidence-producing source commit: `{source_commit}`.
- Final bounded Experiments A, B and C: completed; all promotion gates failed.
- Final Transformer interpretation: completed after explicit fixed-group correction; no model retraining.
- Central metrics: recomputed from preserved Phase 6 predictions and reconciled by score type.
- Evidence freeze and closest-work review: rebuild with `py scripts/build_ifddrp_final_evidence_freeze.py`.
- Empirical modelling status: scientifically frozen on the opened historical panel.
- Next highest-value action: write the dissertation from the three principal contributions and the claim register. Do not reopen modelling unless using a new, independently preregistered replication or prospective dataset.

Resume validation commands:

```text
py scripts/build_ifddrp_final_evidence_freeze.py
py -m pytest
py scripts/check_public_hygiene.py
```
""",
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    state = _source_state()
    metrics = _metric_rows(state)
    results = _result_rows()
    claims = _claim_rows()
    deviations = _protocol_deviations()
    closest = _closest_work_rows()
    selection = _selection_rows()

    _write_csv("ifddrp_authoritative_metric_registry.csv", metrics)
    _write_csv("ifddrp_final_result_registry.csv", results)
    _write_csv("ifddrp_claim_evidence_matrix.csv", claims)
    _write_csv("ifddrp_final_protocol_deviations.csv", deviations)
    _write_csv("ifddrp_closest_work_matrix.csv", closest)
    _write_csv("ifddrp_final_dissertation_result_selection.csv", selection)
    _build_markdown_reports(state, metrics)
    print(
        "IFDDRP final evidence freeze built: "
        f"metrics={len(metrics)}, results={len(results)}, claims={len(claims)}, "
        f"deviations={len(deviations)}, closest_work={len(closest)}, selections={len(selection)}"
    )


if __name__ == "__main__":
    main()
