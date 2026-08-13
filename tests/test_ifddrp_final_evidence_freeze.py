"""Consistency tests for the final IFDDRP evidence freeze."""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from market_dynamics.evaluation.post_freeze import aligned_probability_ensemble

TABLES = Path("reports/tables")
BUILDER = Path("scripts/build_ifddrp_final_evidence_freeze.py")


def _load_builder_module(path: Path = BUILDER):
    spec = importlib.util.spec_from_file_location("ifddrp_evidence_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_builder_imports_and_runs_arithmetic_without_git_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the builder from a directory with no Git metadata."""
    isolated_builder = tmp_path / BUILDER.name
    shutil.copyfile(BUILDER, isolated_builder)
    monkeypatch.chdir(tmp_path)
    module = _load_builder_module(isolated_builder)
    labels = np.asarray([0, 1, 0, 1], dtype=int)
    scores = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=float)
    assets = np.asarray([0, 0, 1, 1], dtype=int)

    within, macro, eligible = module._within_asset_auc(labels, scores, assets)

    assert not (tmp_path / ".git").exists()
    assert within == pytest.approx(1.0)
    assert macro == pytest.approx(1.0)
    assert eligible == 2


def test_markdown_writer_uses_canonical_lf_newlines(tmp_path: Path) -> None:
    """Generated evidence must stay byte-stable on Windows checkouts."""
    module = _load_builder_module()
    module.TABLES = tmp_path

    module._write_markdown("report.md", "line one\nline two")

    assert (tmp_path / "report.md").read_bytes() == b"line one\nline two\n"


def test_authoritative_metrics_separate_ranking_and_probability_scores() -> None:
    metrics = pd.read_csv(TABLES / "ifddrp_authoritative_metric_registry.csv")
    transformer = metrics.loc[
        metrics["model"].eq("transformer_encoder")
        & metrics["variant"].eq("asset_conditioned")
        & metrics["module"].eq("phase6")
    ]
    ranking = transformer.loc[transformer["metric"].isin(["roc_auc", "pr_auc", "pair_weighted_within_asset_roc_auc"])]
    proper = transformer.loc[transformer["metric"].isin(["brier_score", "log_loss"])]
    assert ranking["score_basis"].eq("raw ensemble probability").all()
    assert proper["score_basis"].eq("validation-selected isotonic probability").all()


def test_final_claims_do_not_promote_temporal_skill() -> None:
    claims = pd.read_csv(TABLES / "ifddrp_claim_evidence_matrix.csv")
    temporal = claims.loc[claims["claim_id"].isin(["C03", "C05", "C08", "C09"])]
    assert temporal["classification"].isin(["rejected", "unsafe"]).all()
    assert not claims["adjudication"].str.contains("confirmed temporal skill", case=False).any()


def test_final_selection_has_exactly_three_principal_contributions() -> None:
    selection = pd.read_csv(TABLES / "ifddrp_final_dissertation_result_selection.csv")
    main = selection.loc[selection["category"].eq("A_main")]
    assert len(main) == 3
    assert set(main["selection_id"]) == {"S01", "S02", "S03"}


def test_final_bounded_experiments_are_valid_negatives() -> None:
    results = pd.read_csv(TABLES / "ifddrp_final_result_registry.csv")
    bounded = results.loc[results["result_id"].isin(["R09", "R10", "R11"])]
    assert len(bounded) == 3
    assert bounded["status"].eq("completed_gate_failed").all()
    assert bounded["result_direction"].eq("valid_negative").all()


def test_closest_work_register_is_bounded_and_source_backed() -> None:
    closest = pd.read_csv(TABLES / "ifddrp_closest_work_matrix.csv")
    assert 10 <= len(closest) <= 15
    assert closest["source_url"].str.startswith("https://").all()
    assert closest["direct_difference"].str.len().gt(20).all()


def test_authoritative_metrics_reconstruct_from_private_prediction_artifacts() -> None:
    """Rebuild frozen ranking metrics when ignored authoritative artefacts exist."""
    phase6 = Path("results/runs/phase6_transformer_falsification_20260712/predictions")
    cross_model = Path("results/runs/prp1_fixed_cross_model_20260715/predictions")
    bounded = Path("results/runs/ifddrp_final_bounded_20260808/predictions")
    paths = {
        "transformer": phase6 / "corrected_asset_conditioned_ensemble.parquet",
        "no_id": phase6 / "no_explicit_asset_id_ensemble.parquet",
        "asset_prior": bounded / "static_dynamic__asset_prior_ensemble.parquet",
        "family_prior": bounded / "static_dynamic__family_prior_ensemble.parquet",
        **{
            f"mlp_seed_{seed}": cross_model / f"mlp_asset_conditioned_seed{seed}.parquet"
            for seed in (7, 42, 123)
        },
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        pytest.skip(
            "Private authoritative prediction artefacts are unavailable at their "
            f"documented ignored paths: {missing}"
        )

    module = _load_builder_module()

    def ranking_metrics(frame: pd.DataFrame, score_column: str) -> tuple[float, float]:
        test = frame.loc[frame["split"].eq("test")]
        labels = test["y_true"].to_numpy(dtype=int)
        scores = test[score_column].to_numpy(dtype=float)
        assets = test["asset_id"].to_numpy(dtype=int)
        within, _, _ = module._within_asset_auc(labels, scores, assets)
        return float(roc_auc_score(labels, scores)), within

    transformer = ranking_metrics(pd.read_parquet(paths["transformer"]), "ensemble_probability")
    no_id = ranking_metrics(pd.read_parquet(paths["no_id"]), "ensemble_probability")
    asset_prior = ranking_metrics(pd.read_parquet(paths["asset_prior"]), "raw_probability")
    family_prior = ranking_metrics(pd.read_parquet(paths["family_prior"]), "raw_probability")
    mlp_frames = [pd.read_parquet(paths[f"mlp_seed_{seed}"]) for seed in (7, 42, 123)]
    mlp = ranking_metrics(
        aligned_probability_ensemble(mlp_frames, "raw_probability"),
        "ensemble_probability",
    )

    assert transformer == pytest.approx((0.789813558, 0.491638470), abs=1e-6)
    assert no_id == pytest.approx((0.715476655, 0.472570086), abs=1e-6)
    assert asset_prior[0] == pytest.approx(0.823905856, abs=1e-6)
    assert asset_prior[1] == pytest.approx(0.5, abs=1e-12)
    assert family_prior[0] == pytest.approx(0.816548772, abs=1e-6)
    assert family_prior[1] == pytest.approx(0.5, abs=1e-12)
    assert mlp == pytest.approx((0.796477661, 0.556966922), abs=1e-6)
