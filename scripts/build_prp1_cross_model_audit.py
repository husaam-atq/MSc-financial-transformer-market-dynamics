"""Build compact post-execution audit evidence for the fixed cross-model study."""

from __future__ import annotations

import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "results" / "runs" / "prp1_fixed_cross_model_20260715"
HISTORICAL_DIR = PROJECT_ROOT / "results" / "runs" / "phase6_transformer_falsification_20260712"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
MODELS = ("flattened_logistic", "mlp", "lstm", "tcn", "transformer_encoder")
VARIANTS = ("asset_conditioned", "no_explicit_asset_id")
SEEDS = (7, 42, 123)
HISTORICAL_VARIANTS = {
    "asset_conditioned": "corrected_asset_conditioned",
    "no_explicit_asset_id": "no_explicit_asset_id",
}


def _prediction_path(model: str, variant: str, seed: int) -> Path:
    if model == "transformer_encoder":
        historical = HISTORICAL_VARIANTS[variant]
        return HISTORICAL_DIR / "predictions" / f"{historical}_seed{seed}.parquet"
    return RUN_DIR / "predictions" / f"{model}_{variant}_seed{seed}.parquet"


def _nonoverlap_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for variant in VARIANTS:
            frames: list[pd.DataFrame] = []
            for seed in SEEDS:
                frame = pd.read_parquet(_prediction_path(model, variant, seed))
                test = frame[frame["split"].eq("test")].copy()
                test = test.sort_values(["asset_id", "Date", "source_index"]).reset_index(drop=True)
                selected = test.groupby("asset_id", observed=True).cumcount().mod(10).eq(0)
                sample = test[selected].copy()
                rows.append(_score_nonoverlap(sample, model, variant, str(seed)))
                frames.append(test)
            key = ["Date", "source_index", "asset_id", "asset_ticker", "y_true"]
            ensemble = frames[0][key].copy()
            probabilities = []
            for frame in frames:
                aligned = frame.sort_values(key).reset_index(drop=True)
                if not ensemble.sort_values(key).reset_index(drop=True)[key].equals(aligned[key]):
                    raise RuntimeError(f"Cross-model ensemble endpoint mismatch: {model}/{variant}")
                probabilities.append(aligned["raw_probability"].to_numpy(dtype=float))
            ensemble = ensemble.sort_values(key).reset_index(drop=True)
            ensemble["raw_probability"] = np.mean(np.vstack(probabilities), axis=0)
            selected = ensemble.groupby("asset_id", observed=True).cumcount().mod(10).eq(0)
            rows.append(_score_nonoverlap(ensemble[selected].copy(), model, variant, "ensemble"))
    return pd.DataFrame(rows)


def _score_nonoverlap(
    frame: pd.DataFrame, model: str, variant: str, seed: str
) -> dict[str, object]:
    y = frame["y_true"].to_numpy(dtype=int)
    probability = frame["raw_probability"].to_numpy(dtype=float)
    within: list[tuple[float, int]] = []
    for _, asset in frame.groupby("asset_id", observed=True):
        asset_y = asset["y_true"].to_numpy(dtype=int)
        if len(np.unique(asset_y)) < 2:
            continue
        positives = int(asset_y.sum())
        pairs = positives * (len(asset_y) - positives)
        within.append((float(roc_auc_score(asset_y, asset["raw_probability"])), pairs))
    return {
        "model": model,
        "identity_variant": variant,
        "seed": seed,
        "stride": 10,
        "selection_rule": "every_tenth_test_origin_within_asset_from_first_scored_origin",
        "n_obs": len(frame),
        "assets": int(frame["asset_id"].nunique()),
        "pooled_roc_auc": float(roc_auc_score(y, probability)),
        "pair_weighted_within_asset_roc_auc": float(
            np.average([value for value, _ in within], weights=[pairs for _, pairs in within])
        ),
        "eligible_assets": len(within),
        "status": "completed",
    }


def _artifact_manifest() -> pd.DataFrame:
    historical = [
        path
        for variant in HISTORICAL_VARIANTS.values()
        for seed in SEEDS
        for path in (
            HISTORICAL_DIR / "checkpoints" / f"{variant}_seed{seed}.pt",
            HISTORICAL_DIR / "predictions" / f"{variant}_seed{seed}.parquet",
        )
    ]
    paths = [
        *sorted((RUN_DIR / "checkpoints").glob("*")),
        *sorted((RUN_DIR / "predictions").glob("*")),
        *historical,
    ]
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "tracked": False,
                "artefact_class": "checkpoint" if "checkpoints" in path.parts else "prediction",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    temporal = TABLE_DIR / "prp1_fixed_cross_model_temporal_order.csv"
    local_temporal = RUN_DIR / "fixed_cross_model_temporal_order.csv"
    if len(pd.read_csv(temporal)) != 120:
        raise RuntimeError("Tracked temporal evidence is not the registered 120 rows")
    shutil.copyfile(temporal, local_temporal)
    nonoverlap = _nonoverlap_rows()
    nonoverlap.to_csv(TABLE_DIR / "prp1_fixed_cross_model_nonoverlap_sensitivity.csv", index=False)
    artefacts = _artifact_manifest()
    artefacts.to_csv(TABLE_DIR / "prp1_fixed_cross_model_artifact_manifest.csv", index=False)
    smoke = pd.read_csv(RUN_DIR / "manifests" / "smoke.csv")
    covered = set(zip(smoke["model"], ["asset_conditioned"] * len(smoke), strict=True))
    expected = set((model, variant) for model in MODELS for variant in VARIANTS)
    lines = [
        "# PRP-1 Fixed Cross-Model Execution Audit",
        "",
        "## Empirical completion",
        "",
        "The empirical study is cell-complete: 30 model/identity/seed cells, 163 evaluation rows, 120 temporal-order rows, 15 identity swaps and 48 identity probes. The strict temporal-skill recurrence gate remains failed for all five model families.",
        "",
        "## Protocol and provenance qualifications",
        "",
        f"- Smoke coverage was {len(covered)} of {len(expected)} model/identity combinations. Flattened logistic and all no-ID arms were not represented in the smoke manifest, despite the protocol stop rule.",
        "- The subsequent training and evaluation cells completed, but this does not retroactively satisfy the smoke gate.",
        "- The ignored local temporal file has been synchronized to the complete 120-row tracked evidence.",
        "- The registered stride-10 sensitivity is now reported from existing predictions; no model was retrained.",
        f"- The compact artefact manifest records SHA-256 hashes for {len(artefacts)} ignored checkpoint/prediction files.",
        "- The execution-time runner commit was not persisted inside the ignored run contract. Git history suggests the final execution/report commit is compatible, but the exact execution commit is unavailable and is not inferred as fact.",
        "",
        "## Scientific effect",
        "",
        "These corrections do not alter the negative scientific verdict. They narrow the completion claim from strict protocol completion to empirical cell completion with a documented preflight/smoke noncompliance.",
    ]
    (TABLE_DIR / "prp1_fixed_cross_model_execution_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
