"""Run the independently implemented PRP-1 shortcut simulation programme."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from market_dynamics.research.independent_shortcut_simulation import (
    IndependentSimulationConfig,
    fit_train_only_priors,
    simulate_independent_shortcut_panel,
)


def build_simulation_design(options: dict[str, Any]) -> list[dict[str, object]]:
    """Enumerate every frozen core and robustness replication without selection."""
    seeds = [int(options["base_seed"]) + index for index in range(int(options["replications_per_core_cell"]))]
    rows: list[dict[str, object]] = []
    for prior in options["prior_heterogeneity"]:
        for dynamic in options["dynamic_signal"]:
            for persistence in options["persistence"]:
                for replicate, seed in enumerate(seeds):
                    rows.append(
                        {
                            "scenario": "core",
                            "replicate": replicate,
                            "seed": seed,
                            "prior_heterogeneity": float(prior),
                            "dynamic_signal": float(dynamic),
                            "persistence": float(persistence),
                        }
                    )
    for scenario in options["robustness_scenarios"]:
        for anchor in options["robustness_anchor_cells"]:
            for replicate, seed in enumerate(seeds):
                rows.append(
                    {
                        "scenario": str(scenario["id"]),
                        "replicate": replicate,
                        "seed": seed,
                        **{key: float(value) for key, value in anchor.items()},
                        "scenario_parameters": json.dumps({key: value for key, value in scenario.items() if key != "id"}, sort_keys=True),
                    }
                )
    return rows


def simulation_config(options: dict[str, Any], design_row: dict[str, object]) -> IndependentSimulationConfig:
    """Translate one frozen design row into the independent DGP configuration."""
    config = IndependentSimulationConfig(
        n_assets=int(options["n_assets"]),
        n_families=int(options["n_families"]),
        sequence_length=int(options["sequence_length"]),
        train_periods=int(options["train_periods"]),
        purge_periods=int(options["purge_periods"]),
        validation_periods=int(options["validation_periods"]),
        seed=int(design_row["seed"]),
        prior_heterogeneity=float(design_row["prior_heterogeneity"]),
        family_prior_heterogeneity=0.0,
        dynamic_signal_strength=float(design_row["dynamic_signal"]),
        persistence=float(design_row["persistence"]),
        common_shock_strength=0.0,
        cross_sectional_dependence=0.0,
        event_onset_probability=0.0,
        event_effect=0.0,
        min_event_duration=1,
        max_event_duration=1,
        missing_rate=0.0,
    )
    parameters = json.loads(str(design_row.get("scenario_parameters", "{}")))
    replacements: dict[str, object] = {}
    if "family_prior_heterogeneity" in parameters:
        replacements["family_prior_heterogeneity"] = float(parameters["family_prior_heterogeneity"])
    if "common_shock" in parameters:
        replacements["common_shock_strength"] = float(parameters["common_shock"])
    if "cross_sectional_dependence" in parameters:
        replacements["cross_sectional_dependence"] = float(parameters["cross_sectional_dependence"])
    if "event_duration" in parameters:
        duration = int(parameters["event_duration"])
        replacements.update({"event_onset_probability": 0.03, "event_effect": 1.0, "min_event_duration": duration, "max_event_duration": duration})
    if "missingness" in parameters:
        replacements["missing_rate"] = float(parameters["missingness"])
    return replace(config, **replacements)


def evaluate_simulation_cell(config: IndependentSimulationConfig) -> dict[str, object]:
    """Fit train-only priors/classifier and score one complete validation cell."""
    data = simulate_independent_shortcut_panel(config)
    train = data.panel[data.panel["split"].eq("train")].copy()
    validation = data.panel[data.panel["split"].eq("validation")].copy()
    priors = fit_train_only_priors(train)
    numeric = _sequence_columns(config.sequence_length)
    categorical = ["asset_id", "family_id"]
    processor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("sequence", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
        ]
    )
    model = make_pipeline(processor, LogisticRegression(max_iter=500, solver="lbfgs", random_state=config.seed))
    model.fit(train[categorical + numeric], train["target"].to_numpy(dtype=int))
    ordered = model.predict_proba(validation[categorical + numeric])[:, 1]
    reversed_frame = perturb_sequences(validation, config.sequence_length, "reversed")
    permuted_frame = perturb_sequences(validation, config.sequence_length, "deterministic_permutation")
    reversed_score = model.predict_proba(reversed_frame[categorical + numeric])[:, 1]
    permuted_score = model.predict_proba(permuted_frame[categorical + numeric])[:, 1]
    y = validation["target"].to_numpy(dtype=int)
    global_score = priors.predict(validation, level="global")
    family_score = priors.predict(validation, level="family")
    asset_score = priors.predict(validation, level="asset")
    metrics: dict[str, object] = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_positives": int(train["target"].sum()),
        "validation_positives": int(validation["target"].sum()),
        "train_prevalence": float(train["target"].mean()),
        "validation_prevalence": float(validation["target"].mean()),
    }
    for name, score in [
        ("global_prior", global_score),
        ("family_prior", family_score),
        ("asset_prior", asset_score),
        ("pooled_classifier", ordered),
    ]:
        metrics.update(_score_metrics(validation, score, name))
    ordered_auc = _safe_auc(y, ordered)
    metrics.update(
        {
            "reversed_pooled_roc_auc": _safe_auc(y, reversed_score),
            "permuted_pooled_roc_auc": _safe_auc(y, permuted_score),
            "reversal_auc_drop": ordered_auc - _safe_auc(y, reversed_score),
            "permutation_auc_drop": ordered_auc - _safe_auc(y, permuted_score),
            "reversal_score_spearman": float(pd.Series(ordered).corr(pd.Series(reversed_score), method="spearman")),
            "permutation_score_spearman": float(pd.Series(ordered).corr(pd.Series(permuted_score), method="spearman")),
        }
    )
    return metrics


def perturb_sequences(frame: pd.DataFrame, sequence_length: int, method: str) -> pd.DataFrame:
    """Destroy order identically across every registered sequence channel."""
    if method == "reversed":
        order = np.arange(sequence_length)[::-1]
    elif method == "deterministic_permutation":
        indices = np.arange(sequence_length)
        order = np.concatenate((indices[::2], indices[1::2]))
    else:
        raise ValueError(f"Unknown simulation perturbation: {method}")
    result = frame.copy()
    for prefix in ["observed_dynamic_lag_", "observed_common_lag_", "event_lag_", "missing_lag_"]:
        columns = [f"{prefix}{lag}" for lag in range(sequence_length - 1, -1, -1)]
        result.loc[:, columns] = frame[columns].to_numpy()[:, order]
    return result


def run_simulation_programme(options: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    """Run/resume all cells and persist after every replication."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "simulation_rows.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    design = build_simulation_design(options)
    _bind_resume_state(run_dir, options, design, existing)
    completed = set(existing.loc[existing.get("status", pd.Series(dtype=str)).eq("completed"), "run_id"].astype(str)) if not existing.empty else set()
    rows = existing.to_dict(orient="records") if not existing.empty else []
    for design_row in design:
        run_id = _run_id(design_row)
        if run_id in completed:
            continue
        config = simulation_config(options, design_row)
        started = perf_counter()
        base = {"run_id": run_id, **design_row, **{f"config_{key}": value for key, value in asdict(config).items()}}
        try:
            result = evaluate_simulation_cell(config)
            row = {**base, "status": "completed", "failure_reason": "", "runtime_seconds": perf_counter() - started, **result}
        except Exception as exc:
            row = {**base, "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}", "runtime_seconds": perf_counter() - started}
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def summarize_simulation(rows: pd.DataFrame, options: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Summarize all completed cells and evaluate only frozen core gates."""
    completed = rows[rows["status"].eq("completed")].copy()
    group_columns = ["scenario", "prior_heterogeneity", "dynamic_signal", "persistence"]
    metric_columns = [
        "global_prior_pooled_roc_auc",
        "global_prior_pr_auc",
        "global_prior_brier",
        "global_prior_log_loss",
        "family_prior_pooled_roc_auc",
        "family_prior_pr_auc",
        "family_prior_brier",
        "family_prior_log_loss",
        "asset_prior_pooled_roc_auc",
        "asset_prior_pr_auc",
        "asset_prior_brier",
        "asset_prior_log_loss",
        "pooled_classifier_pooled_roc_auc",
        "pooled_classifier_pr_auc",
        "pooled_classifier_brier",
        "pooled_classifier_log_loss",
        "pooled_classifier_pair_weighted_within_asset_roc_auc",
        "pooled_classifier_asset_macro_roc_auc",
        "pooled_classifier_eligible_assets",
        "train_rows",
        "validation_rows",
        "train_positives",
        "validation_positives",
        "train_prevalence",
        "validation_prevalence",
        "reversal_auc_drop",
        "permutation_auc_drop",
    ]
    summary_rows: list[dict[str, object]] = []
    confidence_level = float(options.get("inference", {}).get("confidence_level", 0.95))
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    for keys, part in completed.groupby(group_columns, observed=True, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["replications"] = len(part)
        for metric in metric_columns:
            values = part[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            critical = (
                float(student_t.ppf((1.0 + confidence_level) / 2.0, len(values) - 1))
                if len(values) > 1
                else np.nan
            )
            margin = critical * float(np.std(values, ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            row[f"{metric}_ci_lower"] = float(np.mean(values) - margin) if len(values) > 1 else np.nan
            row[f"{metric}_ci_upper"] = float(np.mean(values) + margin) if len(values) > 1 else np.nan
        row["pooled_classifier_pooled_minus_within_auc_mean"] = (
            row["pooled_classifier_pooled_roc_auc_mean"]
            - row["pooled_classifier_pair_weighted_within_asset_roc_auc_mean"]
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    core = completed[completed["scenario"].eq("core")]
    gate_values = _seed_cluster_gate_values(core)
    estimates = {
        name: float(values.mean()) for name, values in gate_values.items()
    }
    gates_config = options["gates"]
    gates = {
        "static_prior_inflation": estimates["static_prior_auc_increase"] >= float(gates_config["minimum_static_prior_auc_increase"]),
        "no_signal_within_near_chance": float(gates_config["no_signal_within_auc_lower"]) <= estimates["no_signal_within_auc"] <= float(gates_config["no_signal_within_auc_upper"]),
        "strong_signal_within_recovery": estimates["strong_signal_within_auc"] >= float(gates_config["minimum_strong_signal_within_auc"]),
        "strong_signal_reversal_sensitivity": estimates["strong_signal_reversal_drop"] >= float(gates_config["minimum_strong_signal_reversal_drop"]),
        "strong_signal_permutation_sensitivity": estimates["strong_signal_permutation_drop"] >= float(gates_config["minimum_strong_signal_permutation_drop"]),
    }
    inference = _gate_inference(
        gate_values,
        gates_config,
        gates,
        options.get("inference", {}),
    )
    all_strong = core[core["dynamic_signal"].eq(1.5)]
    sensitivity_estimates = {
        "strong_signal_all_persistence_within_auc": float(
            all_strong["pooled_classifier_pair_weighted_within_asset_roc_auc"].mean()
        ),
        "strong_signal_all_persistence_reversal_drop": float(
            all_strong["reversal_auc_drop"].mean()
        ),
        "strong_signal_all_persistence_permutation_drop": float(
            all_strong["permutation_auc_drop"].mean()
        ),
    }
    return summary, {
        "estimates": estimates,
        "gates": gates,
        "gate_inference": inference,
        "sensitivity_estimates": sensitivity_estimates,
        "completed": len(completed),
        "failed": int(rows["status"].eq("failed").sum()),
    }


def _seed_cluster_gate_values(core: pd.DataFrame) -> dict[str, pd.Series]:
    """Collapse factorial repeats to the 20 independent common-seed clusters."""
    low = core[(core["prior_heterogeneity"].eq(0.0)) & core["dynamic_signal"].eq(0.0)]
    high = core[(core["prior_heterogeneity"].eq(2.5)) & core["dynamic_signal"].eq(0.0)]
    paired = low.merge(
        high,
        on=["replicate", "seed", "persistence"],
        suffixes=("_low", "_high"),
        validate="one_to_one",
    )
    paired["difference"] = (
        paired["asset_prior_pooled_roc_auc_high"]
        - paired["asset_prior_pooled_roc_auc_low"]
    )
    no_signal = core[core["dynamic_signal"].eq(0.0)]
    strong = core[(core["dynamic_signal"].eq(1.5)) & core["persistence"].eq(0.7)]
    metric = "pooled_classifier_pair_weighted_within_asset_roc_auc"
    return {
        "static_prior_auc_increase": paired.groupby("seed")["difference"].mean(),
        "no_signal_within_auc": no_signal.groupby("seed")[metric].mean(),
        "strong_signal_within_auc": strong.groupby("seed")[metric].mean(),
        "strong_signal_reversal_drop": strong.groupby("seed")["reversal_auc_drop"].mean(),
        "strong_signal_permutation_drop": strong.groupby("seed")["permutation_auc_drop"].mean(),
    }


def _gate_inference(
    gate_values: dict[str, pd.Series],
    gates_config: dict[str, Any],
    preregistered_gates: dict[str, bool],
    inference_config: dict[str, Any],
) -> list[dict[str, object]]:
    """Report seed-clustered intervals without redefining preregistered gates."""
    gate_names = {
        "static_prior_auc_increase": "static_prior_inflation",
        "no_signal_within_auc": "no_signal_within_near_chance",
        "strong_signal_within_auc": "strong_signal_within_recovery",
        "strong_signal_reversal_drop": "strong_signal_reversal_sensitivity",
        "strong_signal_permutation_drop": "strong_signal_permutation_sensitivity",
    }
    family_size = len(gate_names)
    confidence_level = float(inference_config.get("confidence_level", 0.95))
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    alpha = 1.0 - confidence_level
    rows: list[dict[str, object]] = []
    for estimate_name, gate_name in gate_names.items():
        values = gate_values[estimate_name].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            raise ValueError(f"Gate {gate_name} requires at least two seed clusters")
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        critical_95 = float(student_t.ppf(1.0 - alpha / 2.0, len(values) - 1))
        critical_simultaneous = float(
            student_t.ppf(1.0 - alpha / (2.0 * family_size), len(values) - 1)
        )
        lower_95 = mean - critical_95 * standard_error
        upper_95 = mean + critical_95 * standard_error
        lower_simultaneous = mean - critical_simultaneous * standard_error
        upper_simultaneous = mean + critical_simultaneous * standard_error
        if estimate_name == "no_signal_within_auc":
            threshold = (
                f"[{float(gates_config['no_signal_within_auc_lower']):.4f}, "
                f"{float(gates_config['no_signal_within_auc_upper']):.4f}]"
            )
            simultaneous_support = (
                lower_simultaneous >= float(gates_config["no_signal_within_auc_lower"])
                and upper_simultaneous <= float(gates_config["no_signal_within_auc_upper"])
            )
        else:
            threshold_key = {
                "static_prior_auc_increase": "minimum_static_prior_auc_increase",
                "strong_signal_within_auc": "minimum_strong_signal_within_auc",
                "strong_signal_reversal_drop": "minimum_strong_signal_reversal_drop",
                "strong_signal_permutation_drop": "minimum_strong_signal_permutation_drop",
            }[estimate_name]
            threshold_value = float(gates_config[threshold_key])
            threshold = f">= {threshold_value:.4f}"
            simultaneous_support = lower_simultaneous >= threshold_value
        rows.append(
            {
                "gate": gate_name,
                "estimate_name": estimate_name,
                "seed_clusters": len(values),
                "confidence_level": confidence_level,
                "estimate": mean,
                "standard_error": standard_error,
                "ci95_lower": lower_95,
                "ci95_upper": upper_95,
                "bonferroni95_lower": lower_simultaneous,
                "bonferroni95_upper": upper_simultaneous,
                "threshold": threshold,
                "preregistered_point_gate_passed": bool(preregistered_gates[gate_name]),
                "post_hoc_simultaneous_interval_support": bool(simultaneous_support),
            }
        )
    return rows


def _bind_resume_state(
    run_dir: Path,
    options: dict[str, Any],
    design: list[dict[str, object]],
    existing: pd.DataFrame,
) -> None:
    """Reject a run directory whose rows do not match the frozen configuration."""
    normalized = json.dumps(options, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    design_ids = [_run_id(row) for row in design]
    design_hash = hashlib.sha256("\n".join(design_ids).encode("utf-8")).hexdigest()
    expected = {
        "config_sha256": config_hash,
        "design_sha256": design_hash,
        "registered_runs": len(design_ids),
    }
    manifest_path = run_dir / "simulation_manifest.json"
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                "Simulation run directory is bound to a different configuration or design"
            )
        return
    if not existing.empty:
        unknown = sorted(set(existing["run_id"].astype(str)).difference(design_ids))
        if unknown:
            raise RuntimeError(f"Existing simulation ledger has unregistered run ids: {unknown[:3]}")
        completed = existing[existing["status"].eq("completed")]
        duplicates = completed["run_id"].duplicated(keep=False)
        if duplicates.any():
            raise RuntimeError("Existing simulation ledger has duplicate completed run ids")
        expected_design = {_run_id(row): row for row in design}
        for record in completed.to_dict(orient="records"):
            config = simulation_config(options, expected_design[str(record["run_id"])])
            for key, expected_value in asdict(config).items():
                column = f"config_{key}"
                if column not in record:
                    raise RuntimeError(f"Existing simulation ledger lacks {column}")
                observed_value = record[column]
                if isinstance(expected_value, float):
                    matches = bool(np.isclose(float(observed_value), expected_value))
                else:
                    matches = str(observed_value) == str(expected_value)
                if not matches:
                    raise RuntimeError(
                        f"Existing simulation row {record['run_id']} conflicts on {column}"
                    )
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def _score_metrics(frame: pd.DataFrame, score: np.ndarray, prefix: str) -> dict[str, float]:
    y = frame["target"].to_numpy(dtype=int)
    per_asset: list[tuple[float, int]] = []
    for _, part in frame.assign(_score=score).groupby("asset_id", observed=True):
        labels = part["target"].to_numpy(dtype=int)
        if len(np.unique(labels)) == 2:
            pairs = int(labels.sum() * (len(labels) - labels.sum()))
            per_asset.append((float(roc_auc_score(labels, part["_score"])), pairs))
    return {
        f"{prefix}_pooled_roc_auc": _safe_auc(y, score),
        f"{prefix}_pr_auc": float(average_precision_score(y, score)),
        f"{prefix}_brier": float(brier_score_loss(y, score)),
        f"{prefix}_log_loss": float(log_loss(y, np.clip(score, 1e-6, 1.0 - 1e-6), labels=[0, 1])),
        f"{prefix}_asset_macro_roc_auc": float(np.mean([value for value, _ in per_asset])) if per_asset else np.nan,
        f"{prefix}_pair_weighted_within_asset_roc_auc": float(np.average([value for value, _ in per_asset], weights=[pairs for _, pairs in per_asset])) if per_asset else np.nan,
        f"{prefix}_eligible_assets": len(per_asset),
    }


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def _sequence_columns(sequence_length: int) -> list[str]:
    return [f"{prefix}{lag}" for prefix in ["observed_dynamic_lag_", "observed_common_lag_", "event_lag_", "missing_lag_"] for lag in range(sequence_length - 1, -1, -1)]


def _run_id(row: dict[str, object]) -> str:
    return "|".join(str(row.get(key, "")) for key in ["scenario", "prior_heterogeneity", "dynamic_signal", "persistence", "replicate", "seed"])
