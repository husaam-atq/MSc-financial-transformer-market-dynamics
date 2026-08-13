from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from market_dynamics.datasets.window_dataset import WindowedTimeSeriesDataset
from market_dynamics.models.deep_learning import build_deep_model
from market_dynamics.training.losses import (
    GaussianNLLFromOutput,
    SigmoidFocalLoss,
    SoftF1Loss,
    build_loss,
)
from market_dynamics.training.sampling import (
    dataset_targets,
    make_weighted_binary_sampler,
)
from market_dynamics.training.train import fit_model, predict_loader


def test_training_smoke_on_synthetic_classification_data() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(96, 4)).astype(np.float32)
    targets = (features[:, 0] + 0.25 * features[:, 1] > 0.0).astype(np.float32)
    dataset = WindowedTimeSeriesDataset(
        features,
        targets,
        pd.date_range("2021-01-01", periods=len(features), freq="D"),
        lookback=12,
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    validation_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    config = {
        "phase2": {
            "model": {
                "hidden_size": 16,
                "dropout": 0.0,
                "num_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_multiplier": 2,
                "patch_length": 4,
                "patch_stride": 2,
            }
        }
    }
    training_config = {
        "epochs": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 2,
        "lr_scheduler_patience": 1,
        "mixed_precision": False,
        "use_pos_weight": True,
    }
    model = build_deep_model("mlp", input_size=4, lookback=12, config=config)
    device = torch.device("cpu")
    result = fit_model(
        model,
        train_loader,
        validation_loader,
        build_loss("classification", dataset.targets, training_config, device),
        training_config,
        device,
    )
    y_true, probabilities, indices = predict_loader(model, validation_loader, "classification", device)

    assert len(result.train_losses) >= 1
    assert result.best_epoch >= 0
    assert y_true.shape == probabilities.shape == indices.shape
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_heteroscedastic_model_and_loss_use_one_shared_two_output_head() -> None:
    config = {
        "phase2": {
            "model": {
                "hidden_size": 8,
                "dropout": 0.0,
                "num_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_multiplier": 2,
                "output_size": 2,
            }
        }
    }
    model = build_deep_model("mlp", input_size=3, lookback=12, config=config)
    output = model(torch.randn(4, 12, 3))

    loss = GaussianNLLFromOutput()(output, torch.randn(4))
    loss.backward()

    assert output.shape == (4, 2)
    assert torch.isfinite(loss)
    final_heads = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and module.out_features == 2
    ]
    assert len(final_heads) == 1
    assert final_heads[0].weight.grad is not None
    assert torch.isfinite(final_heads[0].weight.grad).all()


def test_phase2d_classification_losses_and_gradients_are_finite() -> None:
    logits = torch.tensor([-20.0, 0.0, 20.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0])
    losses = [
        SigmoidFocalLoss(alpha=2.0, gamma=2.0, label_smoothing=0.05)(
            logits,
            targets,
        ),
        SoftF1Loss(label_smoothing=0.05)(logits, targets),
        build_loss(
            "classification",
            targets.numpy(),
            {"classification_loss": "focal", "focal_gamma": 2.0},
            torch.device("cpu"),
        )(logits, targets),
    ]

    sum(losses).backward()

    assert all(torch.isfinite(loss) for loss in losses)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_weighted_sampler_uses_window_endpoint_targets_only() -> None:
    features = np.random.default_rng(7).normal(size=(20, 2)).astype(np.float32)
    targets = np.asarray([1.0] * 4 + [0.0] * 8 + [1.0] * 8, dtype=np.float32)
    dataset = WindowedTimeSeriesDataset(
        features,
        targets,
        pd.date_range("2022-01-01", periods=20),
        lookback=5,
    )

    endpoint_targets = dataset_targets(dataset)
    sampler = make_weighted_binary_sampler(dataset, seed=7, positive_target_rate=0.40)

    np.testing.assert_array_equal(endpoint_targets, targets[4:])
    assert sampler is not None
    positives = endpoint_targets == 1.0
    expected_weights = np.where(
        positives,
        0.40 / positives.sum(),
        0.60 / (~positives).sum(),
    )
    np.testing.assert_allclose(sampler.weights.numpy(), expected_weights)
    assert len(list(iter(sampler))) == len(dataset)


def test_weighted_sampler_rejects_continuous_endpoint_targets() -> None:
    features = np.random.default_rng(8).normal(size=(20, 2)).astype(np.float32)
    targets = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    dataset = WindowedTimeSeriesDataset(
        features,
        targets,
        pd.date_range("2022-01-01", periods=20),
        lookback=5,
    )

    with pytest.raises(ValueError, match="0/1"):
        make_weighted_binary_sampler(dataset, seed=7, positive_target_rate=0.40)
