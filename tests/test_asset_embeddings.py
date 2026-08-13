from __future__ import annotations

import torch

from market_dynamics.models.deep_learning import (
    AssetAgnosticModel,
    AssetConditionedModel,
    build_deep_model,
)
from market_dynamics.models.deep_learning.transformer_encoder import TransformerEncoderModel


def test_asset_embedding_wrapper_forward_and_backward() -> None:
    config = {"phase2": {"model": {"hidden_size": 16, "dropout": 0.0, "num_layers": 1, "transformer_heads": 4, "transformer_ff_multiplier": 2, "patch_length": 4, "patch_stride": 2}}}
    model = AssetConditionedModel(build_deep_model("lstm", 7, 12, config), num_assets=3, embedding_dim=2)
    output = model(torch.randn(4, 12, 5), torch.tensor([0, 1, 2, 1]))
    output.mean().backward()
    assert output.shape == (4,)
    assert model.asset_embedding.weight.grad is not None
    assert model.conditioned_input(torch.randn(2, 12, 5), torch.tensor([0, 1])).shape == (2, 12, 7)


def test_final_transformer_conditioning_contract_has_46_channels_and_no_id_control() -> None:
    numerical_channels = 34
    embedding_channels = 12
    conditioned_base = TransformerEncoderModel(
        input_size=numerical_channels + embedding_channels,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_length=60,
        pooling="temporal_attention",
    )
    no_id_base = TransformerEncoderModel(
        input_size=numerical_channels + embedding_channels,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_length=60,
        pooling="temporal_attention",
    )
    conditioned = AssetConditionedModel(
        conditioned_base,
        num_assets=3,
        embedding_dim=embedding_channels,
    )
    no_id = AssetAgnosticModel(no_id_base, zero_channels=embedding_channels)
    features = torch.randn(2, 60, numerical_channels)
    asset_ids = torch.tensor([0, 2])

    conditioned_input = conditioned.conditioned_input(features, asset_ids)
    no_id_input = no_id.conditioned_input(features, asset_ids)
    conditioned_output = conditioned(features, asset_ids)
    no_id_output = no_id(features, asset_ids)

    assert conditioned.asset_embedding.embedding_dim == embedding_channels
    assert conditioned_input.shape == no_id_input.shape == (2, 60, 46)
    assert torch.count_nonzero(no_id_input[..., numerical_channels:]).item() == 0
    assert conditioned_output.shape == no_id_output.shape == (2,)
    assert torch.isfinite(conditioned_output).all()
    assert torch.isfinite(no_id_output).all()
