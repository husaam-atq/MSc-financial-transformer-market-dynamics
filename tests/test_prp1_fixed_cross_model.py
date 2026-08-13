from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from market_dynamics.experiments.run_prp1_fixed_cross_model import (
    IDENTITY_VARIANTS,
    NEURAL_MODELS,
    FixedCrossModelContext,
    _load_logistic_model,
    _predict_with_represented_id_swap,
    _train_logistic_arm,
    _weighted_binary_metrics,
    balanced_dataset_indices,
    build_fixed_neural_model,
    date_block_bootstrap_within_auc,
    endpoint_fingerprint,
    load_prediction_frame,
    materialize_flattened_dataset,
    validate_prediction_frame,
)


class _TinyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        self.rows = [
            (torch.full((3, 2), float(index)), torch.tensor(index % 2, dtype=torch.float32), torch.tensor(index), torch.tensor(asset))
            for index, asset in enumerate([0, 0, 2, 2, 5, 5])
        ]
        self.targets = np.asarray([int(row[1]) for row in self.rows], dtype=np.float32)
        self.endpoints = np.arange(len(self.rows), dtype=int)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]

    def endpoint_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=len(self.rows)),
                "asset_id": [int(row[3]) for row in self.rows],
                "source_index": [int(row[2]) for row in self.rows],
            }
        )


def _context() -> FixedCrossModelContext:
    dataset = _TinyDataset()
    bundle = SimpleNamespace(
        train=dataset,
        validation=dataset,
        test=dataset,
        feature_columns=["a", "b"],
        asset_to_id={f"ASSET_{index}": index for index in range(6)},
    )
    options = {
        "lookback": 3,
        "model": {
            "dropout": 0.0,
            "tcn_kernel_size": 3,
            "transformer_heads": 2,
            "transformer_ff_multiplier": 2,
            "transformer_pooling": "temporal_attention",
            "asset_embedding_dim": 2,
            "max_length": 16,
            "hidden_sizes": {"mlp": 8, "lstm": 8, "tcn": 8, "transformer_encoder": 8},
            "layers": {"mlp": 2, "lstm": 2, "tcn": 2, "transformer_encoder": 2},
        },
    }
    return FixedCrossModelContext(
        phase6=SimpleNamespace(device=torch.device("cpu"), family_map={}),
        options=options,
        bundle=bundle,
        run_dir=Path("."),
        table_dir=Path("."),
        config_path=Path("config.yaml"),
        config_sha256="config",
        endpoint_sha256="endpoint",
        represented_asset_ids=(0, 2, 5),
    )


@pytest.mark.parametrize("model_name", NEURAL_MODELS)
@pytest.mark.parametrize("identity_variant", IDENTITY_VARIANTS)
def test_registered_models_are_real_and_shape_compatible(model_name: str, identity_variant: str) -> None:
    context = _context()
    model = build_fixed_neural_model(context, model_name, identity_variant)
    x = torch.randn(4, 3, 2)
    asset_ids = torch.tensor([0, 2, 5, 0])

    output = model(x, asset_ids)

    assert output.shape == (4,)
    assert type(model.base_model).__module__.startswith("market_dynamics.models.deep_learning")
    if identity_variant == "no_explicit_asset_id":
        conditioned = model.conditioned_input(x, asset_ids)
        assert conditioned.shape[-1] == 4
        assert torch.count_nonzero(conditioned[..., 2:]).item() == 0


def test_endpoint_fingerprint_changes_with_labels() -> None:
    context = _context()
    first = endpoint_fingerprint(context.bundle)
    context.bundle.test.targets[0] = 1.0
    second = endpoint_fingerprint(context.bundle)

    assert first != second


def test_balanced_probe_indices_are_bounded_per_sparse_asset_id() -> None:
    selected = balanced_dataset_indices(_TinyDataset(), per_asset=1)

    assert selected == [0, 2, 4]


class _IdentityEcho(torch.nn.Module):
    def forward(self, features: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        return asset_ids.float()


def test_identity_swap_cycles_only_represented_ids() -> None:
    dataset = _TinyDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False)

    _, probabilities, _ = _predict_with_represented_id_swap(
        _IdentityEcho(), loader, torch.device("cpu"), represented_ids=(0, 2, 5)
    )

    expected = torch.sigmoid(torch.tensor([2.0, 2.0, 5.0, 5.0, 0.0, 0.0])).numpy()
    np.testing.assert_allclose(probabilities, expected)


def test_ranking_metrics_use_raw_scores_not_calibrated_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [0, 1, 1, 0],
            "raw": [0.1, 0.4, 0.3, 0.2],
            "calibrated": [0.5, 0.5, 0.5, 0.5],
        }
    )

    metrics = _weighted_binary_metrics(frame, "raw", "calibrated", 0.5, None)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.25)


def test_prediction_validation_rejects_endpoint_label_misalignment() -> None:
    context = _context()
    parts = []
    reverse = {value: key for key, value in context.bundle.asset_to_id.items()}
    for split_name, dataset in (("validation", context.bundle.validation), ("test", context.bundle.test)):
        part = dataset.endpoint_metadata()
        part["asset_ticker"] = part["asset_id"].map(reverse)
        part["y_true"] = dataset.targets
        part["raw_probability"] = np.linspace(0.1, 0.9, len(part))
        part["split"] = split_name
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    validate_prediction_frame(context, frame)

    frame.loc[0, "y_true"] = 1 - frame.loc[0, "y_true"]

    with pytest.raises(RuntimeError, match="endpoint mismatch"):
        validate_prediction_frame(context, frame)


def test_prediction_validation_accepts_equivalent_binary_label_dtype() -> None:
    context = _context()
    parts = []
    reverse = {value: key for key, value in context.bundle.asset_to_id.items()}
    for split_name, dataset in (("validation", context.bundle.validation), ("test", context.bundle.test)):
        part = dataset.endpoint_metadata()
        part["asset_ticker"] = part["asset_id"].map(reverse)
        part["y_true"] = dataset.targets.astype(np.int64)
        part["raw_probability"] = np.linspace(0.1, 0.9, len(part))
        part["split"] = split_name
        parts.append(part)

    validate_prediction_frame(context, pd.concat(parts, ignore_index=True))


def test_date_block_bootstrap_within_auc_is_bounded() -> None:
    rows = []
    for asset_id in (0, 1):
        for position, date in enumerate(pd.date_range("2024-01-01", periods=20)):
            label = position % 2
            rows.append(
                {
                    "Date": date,
                    "asset_id": asset_id,
                    "y_true": label,
                    "raw_probability": 0.8 if label else 0.2,
                }
            )
    frame = pd.DataFrame(rows)

    result = date_block_bootstrap_within_auc(frame, "raw_probability", block_size=4, iterations=25, seed=7)

    assert result["estimate"] == pytest.approx(1.0)
    assert 0.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0
    assert result["valid_draw_fraction"] == pytest.approx(1.0)


def test_flattened_logistic_fits_scores_and_reloads_checkpoint(tmp_path: Path) -> None:
    for child in ("checkpoints", "predictions", "logs", "cache"):
        (tmp_path / child).mkdir()
    context = replace(_context(), run_dir=tmp_path, table_dir=tmp_path)
    context.options["logistic"] = {
        "solver": "liblinear",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 100,
        "tolerance": 1e-4,
    }

    row = _train_logistic_arm(context, "asset_conditioned", seed=7)
    first_reload = _load_logistic_model(context, "asset_conditioned", seed=7)
    second_reload = _load_logistic_model(context, "asset_conditioned", seed=7)
    predictions = load_prediction_frame(
        context,
        "flattened_logistic",
        "asset_conditioned",
        seed=7,
    )

    assert row["status"] == "completed"
    assert predictions is not None
    assert len(predictions) == len(context.bundle.validation) + len(context.bundle.test)
    scores = predictions["raw_probability"].to_numpy(dtype=float)
    assert scores.shape == (len(predictions),)
    assert np.isfinite(scores).all()
    assert np.logical_and(scores >= 0.0, scores <= 1.0).all()
    np.testing.assert_array_equal(first_reload.classes_, np.asarray([0, 1]))
    np.testing.assert_allclose(first_reload.coef_, second_reload.coef_)
    test_matrix, _, _ = materialize_flattened_dataset(
        context,
        context.bundle.test,
        "asset_conditioned",
        "test",
    )
    reloaded_scores = first_reload.predict_proba(test_matrix)[:, 1]
    saved_test_scores = predictions.loc[
        predictions["split"].eq("test"),
        "raw_probability",
    ].to_numpy(dtype=float)
    np.testing.assert_allclose(reloaded_scores, saved_test_scores)
