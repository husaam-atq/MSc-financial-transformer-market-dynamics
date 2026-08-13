from __future__ import annotations

import pytest
import torch

from market_dynamics.models.deep_learning import build_deep_model


@pytest.mark.parametrize("model_name", ["bilstm", "tft", "timesnet", "informer", "autoformer", "fedformer", "nbeats", "nhits", "dlinear", "nlinear"])
def test_extended_models_have_finite_scalar_output_and_backward(model_name: str) -> None:
    config = {"phase2": {"model": {"hidden_size": 32, "dropout": 0.1, "num_layers": 2, "transformer_heads": 4, "transformer_ff_multiplier": 2, "patch_length": 10, "patch_stride": 5}}}
    output = build_deep_model(model_name, 6, 60, config)(torch.randn(3, 60, 6))
    output.mean().backward()
    assert output.shape == (3,)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression test requires a GPU")
@pytest.mark.parametrize("model_name", ["timesnet", "autoformer", "fedformer"])
def test_fft_models_support_cuda_amp_with_60_step_windows(model_name: str) -> None:
    config = {"phase2": {"model": {"hidden_size": 32, "dropout": 0.1, "num_layers": 2, "transformer_heads": 4, "transformer_ff_multiplier": 2, "patch_length": 10, "patch_stride": 5}}}
    model = build_deep_model(model_name, 6, 60, config).cuda()
    with torch.autocast(device_type="cuda", enabled=True):
        output = model(torch.randn(2, 60, 6, device="cuda"))
        output.mean().backward()
    assert torch.isfinite(output).all()
