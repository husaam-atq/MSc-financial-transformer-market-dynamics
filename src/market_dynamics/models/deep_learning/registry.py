"""Explicit provenance for the supported deep-model implementations."""

from __future__ import annotations

import pandas as pd

DEEP_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    name: {"implementation": "project_native", "notes": notes}
    for name, notes in {
        "mlp": "Flattened observed-window baseline.",
        "lstm": "Native PyTorch sequence encoder.",
        "gru": "Native PyTorch sequence encoder.",
        "bilstm": "Bidirectional only within the observed historical input window.",
        "tcn": "Causal dilated temporal convolution.",
        "transformer_encoder": "Native PyTorch time-token encoder.",
        "patchtst": "Channel-independent temporal patch encoder.",
        "itransformer": "Inverted-variate token encoder.",
        "tft": "Compact TFT-style variable selection, LSTM and attention encoder.",
        "timesnet": "Compact FFT-period convolutional encoder.",
        "informer": "Query-sampled ProbSparse-style attention encoder.",
        "autoformer": "Decomposition and auto-correlation aggregation encoder.",
        "fedformer": "Fourier-domain decomposed transformer encoder.",
        "nbeats": "Generic residual N-BEATS block stack.",
        "nhits": "Multi-rate N-HiTS-inspired hierarchy.",
        "dlinear": "Trend/seasonal decomposition linear model.",
        "nlinear": "Last-value normalized linear model.",
    }.items()
}


def deep_model_registry_frame() -> pd.DataFrame:
    """Return a durable model provenance table for results directories."""
    return pd.DataFrame([{"model": name, **metadata} for name, metadata in DEEP_MODEL_REGISTRY.items()])
