"""PyTorch reproducibility and device helpers."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def set_torch_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available, otherwise return CPU with a clear warning."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def mixed_precision_enabled(requested: bool, device: torch.device) -> bool:
    """Enable AMP only for a CUDA device."""
    return bool(requested and device.type == "cuda")
