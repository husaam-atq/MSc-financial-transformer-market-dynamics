"""Factory for Phase 2 deep sequence models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from market_dynamics.models.deep_learning.autoformer import AutoformerModel
from market_dynamics.models.deep_learning.bilstm import BiLSTMModel
from market_dynamics.models.deep_learning.dlinear import DLinearModel
from market_dynamics.models.deep_learning.fedformer import FEDformerModel
from market_dynamics.models.deep_learning.gru import GRUModel
from market_dynamics.models.deep_learning.informer import InformerModel
from market_dynamics.models.deep_learning.itransformer import ITransformerModel
from market_dynamics.models.deep_learning.lstm import LSTMModel
from market_dynamics.models.deep_learning.mlp import MLPSequenceModel
from market_dynamics.models.deep_learning.nbeats import NBeatsModel
from market_dynamics.models.deep_learning.nhits import NHiTSModel
from market_dynamics.models.deep_learning.nlinear import NLinearModel
from market_dynamics.models.deep_learning.patchtst import PatchTSTModel
from market_dynamics.models.deep_learning.tcn import TCNModel
from market_dynamics.models.deep_learning.tft import TemporalFusionTransformerModel
from market_dynamics.models.deep_learning.timesnet import TimesNetModel
from market_dynamics.models.deep_learning.transformer_encoder import TransformerEncoderModel


def build_deep_model(
    model_name: str,
    input_size: int,
    lookback: int,
    config: dict[str, Any],
) -> nn.Module:
    """Build a configured scalar-output model for classification or regression."""
    model_config = config.get("phase2", {}).get("model", config.get("model", {}))
    output_size = int(model_config.get("output_size", 1))
    if output_size < 1:
        raise ValueError("Deep model output_size must be positive")
    hidden_size = int(model_config.get("hidden_size", 64))
    dropout = float(model_config.get("dropout", 0.15))
    num_layers = int(model_config.get("num_layers", 2))
    heads = int(model_config.get("transformer_heads", 4))
    ff_multiplier = int(model_config.get("transformer_ff_multiplier", 2))
    max_length = max(int(model_config.get("max_length", 1024)), int(lookback))

    name = model_name.lower()
    if name == "mlp":
        model = MLPSequenceModel(input_size, lookback, hidden_size, dropout)
    elif name == "lstm":
        model = LSTMModel(input_size, hidden_size, num_layers, dropout)
    elif name == "bilstm":
        model = BiLSTMModel(input_size, hidden_size, num_layers, dropout)
    elif name == "gru":
        model = GRUModel(input_size, hidden_size, num_layers, dropout)
    elif name == "tcn":
        model = TCNModel(
            input_size,
            hidden_size,
            num_layers,
            int(model_config.get("tcn_kernel_size", 3)),
            dropout,
        )
    elif name == "transformer_encoder":
        model = TransformerEncoderModel(
            input_size,
            hidden_size,
            num_layers,
            heads,
            dropout,
            ff_multiplier,
            max_length=max_length,
            pooling=str(model_config.get("transformer_pooling", "last")),
        )
    elif name == "patchtst":
        model = PatchTSTModel(
            input_size,
            hidden_size,
            num_layers,
            heads,
            int(model_config.get("patch_length", 10)),
            int(model_config.get("patch_stride", 5)),
            dropout,
            ff_multiplier,
        )
    elif name == "itransformer":
        model = ITransformerModel(
            input_size, lookback, hidden_size, num_layers, heads, dropout, ff_multiplier
        )
    elif name == "tft":
        model = TemporalFusionTransformerModel(input_size, hidden_size, num_layers, heads, dropout, max_length=max_length)
    elif name == "timesnet":
        model = TimesNetModel(input_size, hidden_size, dropout)
    elif name == "informer":
        model = InformerModel(input_size, hidden_size, num_layers, heads, dropout, max_length=max_length)
    elif name == "autoformer":
        model = AutoformerModel(input_size, hidden_size, num_layers, dropout)
    elif name == "fedformer":
        model = FEDformerModel(input_size, hidden_size, num_layers, heads, dropout, max_length=max_length)
    elif name == "nbeats":
        model = NBeatsModel(input_size, lookback, hidden_size, num_layers, dropout)
    elif name == "nhits":
        model = NHiTSModel(input_size, lookback, hidden_size, dropout)
    elif name == "dlinear":
        model = DLinearModel(input_size, lookback)
    elif name == "nlinear":
        model = NLinearModel(input_size, lookback)
    else:
        raise ValueError(f"Unknown deep model: {model_name}")
    if output_size != 1:
        _replace_final_scalar_head(model, output_size)
    return model


def _replace_final_scalar_head(model: nn.Module, output_size: int) -> None:
    """Replace one final scalar linear head while preserving the shared body."""
    candidates: list[tuple[str, nn.Linear]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and int(module.out_features) == 1
    ]
    if not candidates:
        raise ValueError(f"Could not locate a scalar output head for {type(model).__name__}")
    name, current = candidates[-1]
    parent = _module_parent(model, name)
    attribute = name.rsplit(".", 1)[-1]
    replacement = nn.Linear(current.in_features, output_size, bias=current.bias is not None)
    with torch.no_grad():
        replacement.weight[0].copy_(current.weight[0])
        if current.bias is not None:
            replacement.bias[0].copy_(current.bias[0])
    setattr(parent, attribute, replacement)


def _module_parent(model: nn.Module, module_name: str) -> nn.Module:
    parent = model
    parts = module_name.split(".")[:-1]
    for part in parts:
        parent = parent[int(part)] if part.isdigit() and isinstance(parent, nn.Sequential) else getattr(parent, part)
    return parent
