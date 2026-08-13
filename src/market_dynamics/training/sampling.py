"""Train-only sampling helpers for Phase 2D class-imbalance experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler


def dataset_targets(dataset: Dataset[Any]) -> np.ndarray:
    """Return endpoint targets for supported window datasets or deterministic subsets."""
    if isinstance(dataset, Subset):
        base_targets = dataset_targets(dataset.dataset)
        return base_targets[np.asarray(dataset.indices, dtype=int)]
    if hasattr(dataset, "targets") and hasattr(dataset, "endpoints"):
        return np.asarray(dataset.targets)[np.asarray(dataset.endpoints, dtype=int)]
    if hasattr(dataset, "references") and hasattr(dataset, "assets"):
        values = []
        for asset_index, endpoint in dataset.references:
            asset = dataset.assets[int(asset_index)]
            values.append(float(asset.target[int(endpoint)]))
        return np.asarray(values, dtype=np.float32)
    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)!r}")


def dataset_asset_ids(dataset: Dataset[Any]) -> np.ndarray:
    """Return one pooled asset identifier for each window without loading features."""
    if isinstance(dataset, Subset):
        base_asset_ids = dataset_asset_ids(dataset.dataset)
        return base_asset_ids[np.asarray(dataset.indices, dtype=int)]
    if hasattr(dataset, "references") and hasattr(dataset, "assets"):
        return np.asarray(
            [int(dataset.assets[int(asset_index)].asset_id) for asset_index, _ in dataset.references],
            dtype=np.int64,
        )
    raise TypeError(f"Unsupported dataset type for asset-id extraction: {type(dataset)!r}")


def make_weighted_binary_sampler(
    dataset: Dataset[Any],
    seed: int,
    positive_target_rate: float = 0.30,
) -> WeightedRandomSampler[float] | None:
    """Build a train-only sampler that oversamples positive labels when both classes exist."""
    targets = dataset_targets(dataset)
    finite_targets = targets[np.isfinite(targets)]
    if len(finite_targets) != len(targets):
        raise ValueError("Weighted binary sampling requires finite targets")
    if not np.all(np.isin(np.unique(finite_targets), [0.0, 1.0])):
        raise ValueError("Weighted binary sampling requires targets encoded as 0/1")
    positives = targets > 0.5
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    rate = min(max(float(positive_target_rate), 1e-3), 1.0 - 1e-3)
    weights = np.where(positives, rate / positive_count, (1.0 - rate) / negative_count)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def make_balanced_group_sampler(
    dataset: Dataset[Any],
    asset_to_group: dict[int, int] | None,
    seed: int,
    mode: str,
) -> WeightedRandomSampler[float] | None:
    """Balance pooled windows by asset or a caller-supplied family grouping.

    All group assignments are fixed from training-universe metadata. The sampler
    touches no validation or test labels, outcomes, or timestamps.
    """
    asset_ids = dataset_asset_ids(dataset)
    if len(asset_ids) == 0:
        return None
    normalized_mode = str(mode).lower()
    if normalized_mode == "equal_asset":
        groups = asset_ids
    elif normalized_mode == "equal_family":
        if asset_to_group is None:
            raise ValueError("equal_family sampling requires an asset_to_group mapping")
        missing = sorted(set(int(asset) for asset in asset_ids).difference(asset_to_group))
        if missing:
            raise ValueError(f"Family sampler has no group assignment for asset ids: {missing}")
        groups = np.asarray([int(asset_to_group[int(asset)]) for asset in asset_ids], dtype=np.int64)
    else:
        raise ValueError("Balanced sampler mode must be 'equal_asset' or 'equal_family'")
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < 2:
        return None
    weight_by_group = {int(group): 1.0 / float(count) for group, count in zip(unique, counts, strict=True)}
    weights = np.asarray([weight_by_group[int(group)] for group in groups], dtype=np.float64)
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
