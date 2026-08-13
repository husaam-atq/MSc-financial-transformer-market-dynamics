from __future__ import annotations

import pytest
import torch

from market_dynamics.models.deep_learning import build_deep_model


@pytest.mark.parametrize(
    "model_name",
    ["mlp", "lstm", "gru", "tcn", "transformer_encoder", "patchtst", "itransformer"],
)
@pytest.mark.parametrize(
    ("output_size", "expected_shape"),
    [(1, (4,)), (2, (4, 2))],
    ids=["scalar_head", "distributional_head"],
)
def test_deep_model_forward_shapes_for_retained_heads(
    model_name: str,
    output_size: int,
    expected_shape: tuple[int, ...],
) -> None:
    config = {
        "phase2": {
            "model": {
                "hidden_size": 32,
                "dropout": 0.1,
                "num_layers": 2,
                "tcn_kernel_size": 3,
                "transformer_heads": 4,
                "transformer_ff_multiplier": 2,
                "patch_length": 10,
                "patch_stride": 5,
                "output_size": output_size,
            }
        }
    }
    model = build_deep_model(model_name, input_size=6, lookback=60, config=config)
    output = model(torch.randn(4, 60, 6))

    assert output.shape == expected_shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("lookback", [60, 720], ids=["registered_60", "above_512"])
def test_transformer_supports_registered_and_long_lookbacks(lookback: int) -> None:
    config = {
        "phase2": {
            "model": {
                "hidden_size": 16,
                "dropout": 0.0,
                "num_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_multiplier": 2,
                "max_length": 512,
            }
        }
    }
    model = build_deep_model(
        "transformer_encoder",
        input_size=5,
        lookback=lookback,
        config=config,
    )

    output = model(torch.randn(1, lookback, 5))

    assert output.shape == (1,)
    assert torch.isfinite(output).all()
    assert model.position.encoding.shape[1] >= lookback


def test_transformer_temporal_attention_pooling_has_scalar_output() -> None:
    config = {
        "phase2": {
            "model": {
                "hidden_size": 32,
                "dropout": 0.0,
                "num_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_multiplier": 2,
                "transformer_pooling": "temporal_attention",
            }
        }
    }
    model = build_deep_model(
        "transformer_encoder",
        input_size=5,
        lookback=90,
        config=config,
    )
    inputs = torch.randn(3, 90, 5)

    output = model(inputs)
    states, representation, attention = model.encode(inputs)

    assert output.shape == (3,)
    assert states.shape == (3, 90, 32)
    assert representation.shape == (3, 32)
    assert torch.isfinite(output).all()
    assert attention is not None
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0.0)
