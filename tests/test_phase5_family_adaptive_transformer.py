from __future__ import annotations

import torch

from market_dynamics.models.deep_learning import FamilyAdaptiveTransformer


def _model() -> FamilyAdaptiveTransformer:
    return FamilyAdaptiveTransformer(
        input_size=4,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        num_assets=4,
        asset_to_family=torch.tensor([0, 0, 1, 1]),
        num_families=2,
        asset_embedding_dim=3,
        family_embedding_dim=2,
        pooling="temporal_attention",
    )


def test_family_adaptive_transformer_shape_and_temporal_attention() -> None:
    model = _model()
    x = torch.randn(5, 12, 4)
    asset_ids = torch.tensor([0, 1, 2, 3, 0])
    output = model(x, asset_ids)
    _, _, families, attention = model.encode(x, asset_ids)
    assert output.shape == (5,)
    assert families.tolist() == [0, 0, 1, 1, 0]
    assert attention is not None
    assert torch.allclose(attention.sum(dim=1), torch.ones(5), atol=1e-6)


def test_family_specific_residual_head_is_selected_from_asset_family() -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.shared_head.parameters():
            parameter.zero_()
        for head in model.family_heads:
            head.weight.zero_()
        model.family_heads[0].bias.fill_(1.0)
        model.family_heads[1].bias.fill_(3.0)
    x = torch.zeros(2, 12, 4)
    output = model(x, torch.tensor([0, 2]))
    assert torch.allclose(output, torch.tensor([1.0, 3.0]), atol=1e-6)


def test_family_adaptive_transformer_rejects_invalid_asset_id() -> None:
    model = _model()
    try:
        model(torch.randn(1, 12, 4), torch.tensor([4]))
    except IndexError:
        pass
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("Out-of-range pooled asset ids must fail")
