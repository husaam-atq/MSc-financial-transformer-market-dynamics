"""Memory-efficient, leakage-safe pooled multi-asset sequence datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from market_dynamics.preprocessing.scaling import TrainOnlyPreprocessor
from market_dynamics.splits.temporal import GlobalDateSplit


@dataclass
class _AssetSplitData:
    """Scaled one-asset rows and valid strict window endpoints for one segment."""

    features: np.ndarray
    target: np.ndarray
    dates: pd.DatetimeIndex
    asset_id: int
    endpoints: np.ndarray
    source_indices: np.ndarray


class PooledWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """A lazy pooled dataset; each window remains within one asset and one split."""

    def __init__(self, assets: list[_AssetSplitData], lookback: int) -> None:
        self.assets = assets
        self.lookback = lookback
        self.references = [(asset_index, int(endpoint)) for asset_index, part in enumerate(assets) for endpoint in part.endpoints]
        if not self.references:
            raise ValueError("Pooled split has no valid per-asset windows")

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        asset_index, endpoint = self.references[item]
        asset = self.assets[asset_index]
        start = endpoint - self.lookback + 1
        return (
            torch.from_numpy(asset.features[start : endpoint + 1]),
            torch.tensor(asset.target[endpoint], dtype=torch.float32),
            torch.tensor(asset.source_indices[endpoint], dtype=torch.long),
            torch.tensor(asset.asset_id, dtype=torch.long),
        )

    def endpoint_metadata(self) -> pd.DataFrame:
        """Return test endpoint date and asset id in loader ordering."""
        rows = []
        for asset_index, endpoint in self.references:
            asset = self.assets[asset_index]
            rows.append({"Date": asset.dates[endpoint], "asset_id": asset.asset_id, "source_index": asset.source_indices[endpoint]})
        return pd.DataFrame(rows)


@dataclass
class PooledWindowDataBundle:
    """Pooled split datasets, train-only per-asset scalers and id mappings."""

    train: PooledWindowDataset
    validation: PooledWindowDataset
    test: PooledWindowDataset
    preprocessors: dict[str, TrainOnlyPreprocessor]
    asset_to_id: dict[str, int]
    feature_columns: list[str]
    target_column: str
    split: GlobalDateSplit
    skipped_assets: dict[str, str]


def build_pooled_window_datasets(
    panel: pd.DataFrame,
    identifier: str,
    feature_columns: list[str],
    target_column: str,
    split: GlobalDateSplit,
    lookback: int,
) -> PooledWindowDataBundle:
    """Fit one scaler per asset on train dates, then create strictly local windows."""
    if target_column not in panel.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if identifier not in panel.columns:
        raise KeyError(f"Panel identifier not found: {identifier}")
    if lookback < 1:
        raise ValueError("lookback must be positive")
    asset_to_id = {name: index for index, name in enumerate(sorted(panel[identifier].dropna().unique()))}
    preprocessors: dict[str, TrainOnlyPreprocessor] = {}
    components: dict[str, list[_AssetSplitData]] = {"train": [], "validation": [], "test": []}
    skipped: dict[str, str] = {}
    source_offset = 0
    for asset_name, raw_asset in panel.groupby(identifier, observed=True, sort=True):
        asset = raw_asset.sort_index().dropna(subset=[target_column]).copy()
        train_part = asset[asset.index.isin(split.train_dates)]
        if len(train_part) < lookback:
            skipped[str(asset_name)] = f"train rows below lookback ({len(train_part)} < {lookback})"
            continue
        preprocessor = TrainOnlyPreprocessor(scale=True)
        preprocessor.fit(train_part[feature_columns])
        complete = _asset_split_data(asset, split.train_dates, preprocessor, feature_columns, target_column, asset_to_id[str(asset_name)], lookback, source_offset)
        validation = _asset_split_data(asset, split.val_dates, preprocessor, feature_columns, target_column, asset_to_id[str(asset_name)], lookback, source_offset)
        test = _asset_split_data(asset, split.test_dates, preprocessor, feature_columns, target_column, asset_to_id[str(asset_name)], lookback, source_offset)
        if not all(part is not None for part in [complete, validation, test]):
            skipped[str(asset_name)] = "one or more split segments has fewer rows than lookback"
            source_offset += len(asset)
            continue
        preprocessors[str(asset_name)] = preprocessor
        components["train"].append(complete)
        components["validation"].append(validation)
        components["test"].append(test)
        source_offset += len(asset)
    return PooledWindowDataBundle(
        train=PooledWindowDataset(components["train"], lookback),
        validation=PooledWindowDataset(components["validation"], lookback),
        test=PooledWindowDataset(components["test"], lookback),
        preprocessors=preprocessors,
        asset_to_id=asset_to_id,
        feature_columns=feature_columns,
        target_column=target_column,
        split=split,
        skipped_assets=skipped,
    )


def _asset_split_data(
    asset: pd.DataFrame,
    dates: pd.DatetimeIndex,
    preprocessor: TrainOnlyPreprocessor,
    feature_columns: list[str],
    target_column: str,
    asset_id: int,
    lookback: int,
    source_offset: int,
) -> _AssetSplitData | None:
    subset = asset[asset.index.isin(dates)]
    if len(subset) < lookback:
        return None
    features = preprocessor.transform(subset[feature_columns]).astype(np.float32)
    endpoints = np.arange(lookback - 1, len(subset), dtype=np.int64)
    return _AssetSplitData(
        features=features,
        target=subset[target_column].to_numpy(dtype=np.float32),
        dates=pd.DatetimeIndex(subset.index),
        asset_id=asset_id,
        endpoints=endpoints,
        source_indices=np.arange(source_offset, source_offset + len(subset), dtype=np.int64),
    )
