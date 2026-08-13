"""Deterministic, outcome-disjoint within-asset ranking pairs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from market_dynamics.training.sampling import dataset_targets


@dataclass(frozen=True)
class PairRegistry:
    """Selected item pairs, support diagnostics and a content hash."""

    pairs: pd.DataFrame
    audit: pd.DataFrame
    sha256: str


class WithinAssetPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Present fixed positive-negative pairs from one pooled split."""

    def __init__(self, base: Dataset[Any], registry: PairRegistry) -> None:
        self.base = base
        self.registry = registry
        rows = registry.pairs.reset_index(drop=True)
        if rows.empty:
            raise ValueError("Pair registry contains no selected pairs")
        self._positive_items = rows["positive_item"].to_numpy(dtype=np.int64)
        self._negative_items = rows["negative_item"].to_numpy(dtype=np.int64)
        self._weights = rows["pair_weight"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self._positive_items)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positive = self.base[int(self._positive_items[item])]
        negative = self.base[int(self._negative_items[item])]
        positive_asset = positive[3]
        negative_asset = negative[3]
        if int(positive_asset) != int(negative_asset):
            raise RuntimeError("Pair registry crossed asset identities")
        return positive[0], negative[0], positive_asset, torch.tensor(self._weights[item], dtype=torch.float32)


def build_outcome_disjoint_pair_registry(
    dataset: Dataset[Any],
    *,
    horizon: int,
    maximum_pairs_per_asset: int,
    seed: int,
    split: str,
) -> PairRegistry:
    """Select deterministic same-asset class-opposite pairs.

    Origins must be separated by at least ``horizon`` observed sessions. This
    makes their forward ``horizon``-session outcome windows non-overlapping.
    Selection is approximately pair-uniform and bounded per asset without
    materialising the full Cartesian product.
    """
    if horizon < 1 or maximum_pairs_per_asset < 1:
        raise ValueError("horizon and maximum_pairs_per_asset must be positive")
    metadata = dataset.endpoint_metadata().reset_index(drop=True).copy()
    metadata["item_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["y_true"] = dataset_targets(dataset).astype(int)
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for asset_id, raw_group in metadata.groupby("asset_id", observed=True, sort=True):
        group = raw_group.sort_values("Date").reset_index(drop=True)
        group["asset_ordinal"] = np.arange(len(group), dtype=np.int64)
        positive = group[group["y_true"].eq(1)].copy()
        negative = group[group["y_true"].eq(0)].copy()
        positive_ordinals = positive["asset_ordinal"].to_numpy(dtype=np.int64)
        negative_ordinals = negative["asset_ordinal"].to_numpy(dtype=np.int64)
        left_counts = np.searchsorted(negative_ordinals, positive_ordinals - horizon, side="right")
        right_starts = np.searchsorted(negative_ordinals, positive_ordinals + horizon, side="left")
        eligible_counts = left_counts + (len(negative_ordinals) - right_starts)
        valid_pairs = int(eligible_counts.sum())
        selected = _select_asset_pairs(
            positive,
            negative,
            eligible_counts,
            horizon=horizon,
            maximum_pairs=min(maximum_pairs_per_asset, valid_pairs),
            seed=int(seed) + 1009 * int(asset_id),
        )
        pair_count = len(selected)
        pair_weight = 1.0 / pair_count if pair_count else np.nan
        distances: list[int] = []
        endpoint_items: set[int] = set()
        for positive_row, negative_row in selected:
            distance = abs(int(positive_row["asset_ordinal"]) - int(negative_row["asset_ordinal"]))
            distances.append(distance)
            positive_item = int(positive_row["item_index"])
            negative_item = int(negative_row["item_index"])
            endpoint_items.update([positive_item, negative_item])
            rows.append(
                {
                    "split": split,
                    "asset_id": int(asset_id),
                    "positive_item": positive_item,
                    "negative_item": negative_item,
                    "positive_date": pd.Timestamp(positive_row["Date"]),
                    "negative_date": pd.Timestamp(negative_row["Date"]),
                    "origin_distance_sessions": distance,
                    "pair_weight": pair_weight,
                }
            )
        audits.append(
            {
                "split": split,
                "asset_id": int(asset_id),
                "endpoints": len(group),
                "positives": len(positive),
                "negatives": len(negative),
                "prevalence": float(group["y_true"].mean()),
                "raw_comparable_pairs": int(len(positive) * len(negative)),
                "outcome_disjoint_pairs": valid_pairs,
                "selected_pairs": pair_count,
                "unique_endpoints": len(endpoint_items),
                "minimum_origin_distance": min(distances) if distances else np.nan,
                "median_origin_distance": float(np.median(distances)) if distances else np.nan,
                "status": "selected" if pair_count else "insufficient_pair_support",
            }
        )
    pairs = pd.DataFrame(rows).sort_values(
        ["asset_id", "positive_date", "negative_date"], ignore_index=True
    ) if rows else pd.DataFrame()
    digest_columns = [
        "split", "asset_id", "positive_item", "negative_item",
        "positive_date", "negative_date", "origin_distance_sessions",
    ]
    payload = pairs[digest_columns].to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S").encode("utf-8") if len(pairs) else b""
    return PairRegistry(pairs=pairs, audit=pd.DataFrame(audits), sha256=hashlib.sha256(payload).hexdigest())


def _select_asset_pairs(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    eligible_counts: np.ndarray,
    *,
    horizon: int,
    maximum_pairs: int,
    seed: int,
) -> list[tuple[pd.Series, pd.Series]]:
    if maximum_pairs == 0:
        return []
    rng = np.random.default_rng(seed)
    positive_records = [row for _, row in positive.iterrows()]
    negative_records = [row for _, row in negative.iterrows()]
    positive_ordinals = positive["asset_ordinal"].to_numpy(dtype=np.int64)
    negative_ordinals = negative["asset_ordinal"].to_numpy(dtype=np.int64)
    left_counts = np.searchsorted(negative_ordinals, positive_ordinals - horizon, side="right")
    right_starts = np.searchsorted(negative_ordinals, positive_ordinals + horizon, side="left")
    probabilities = eligible_counts.astype(float) / eligible_counts.sum()
    selected: dict[tuple[int, int], tuple[pd.Series, pd.Series]] = {}
    attempts = 0
    limit = max(10_000, maximum_pairs * 30)
    while len(selected) < maximum_pairs and attempts < limit:
        positive_position = int(rng.choice(len(positive_records), p=probabilities))
        positive_row = positive_records[positive_position]
        left_count = int(left_counts[positive_position])
        right_start = int(right_starts[positive_position])
        eligible_count = int(eligible_counts[positive_position])
        draw = int(rng.integers(0, eligible_count))
        negative_position = draw if draw < left_count else right_start + draw - left_count
        negative_row = negative_records[negative_position]
        key = (int(positive_row["item_index"]), int(negative_row["item_index"]))
        selected[key] = (positive_row, negative_row)
        attempts += 1
    if len(selected) < maximum_pairs:
        for positive_row in positive_records:
            for negative_row in negative_records:
                if abs(int(positive_row["asset_ordinal"]) - int(negative_row["asset_ordinal"])) < horizon:
                    continue
                key = (int(positive_row["item_index"]), int(negative_row["item_index"]))
                selected[key] = (positive_row, negative_row)
                if len(selected) == maximum_pairs:
                    break
            if len(selected) == maximum_pairs:
                break
    return [selected[key] for key in sorted(selected)]
