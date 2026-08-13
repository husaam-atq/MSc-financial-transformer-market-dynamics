"""Lightweight GPU and process memory telemetry for run manifests."""

from __future__ import annotations


def runtime_memory_summary() -> dict[str, float | str | None]:
    """Return process and CUDA memory measurements without requiring CUDA."""
    result: dict[str, float | str | None] = {"process_rss_mb": None, "cuda_allocated_mb": None}
    try:
        import psutil

        result["process_rss_mb"] = round(psutil.Process().memory_info().rss / 1024**2, 2)
    except ImportError:
        result["process_rss_mb"] = "psutil-not-installed"
    try:
        import torch

        if torch.cuda.is_available():
            result["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 2)
            result["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["cuda_allocated_mb"] = "torch-not-installed"
    return result
