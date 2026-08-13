from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from market_dynamics.training.sampling import dataset_asset_ids, make_balanced_group_sampler


@dataclass
class _Asset:
    asset_id: int


class _PooledLikeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        self.assets = [_Asset(0), _Asset(1), _Asset(2)]
        self.references = [(0, 0)] * 8 + [(1, 0)] * 2 + [(2, 0)] * 2

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        asset_position, _ = self.references[index]
        return torch.zeros(1), torch.tensor(0.0), torch.tensor(index), torch.tensor(self.assets[asset_position].asset_id)


def test_equal_asset_sampler_uses_inverse_window_counts() -> None:
    dataset = _PooledLikeDataset()
    sampler = make_balanced_group_sampler(dataset, asset_to_group=None, seed=7, mode="equal_asset")
    assert sampler is not None
    weights = sampler.weights.detach().cpu().numpy()
    assert np.allclose(weights[:8], np.full(8, 1.0 / 8.0))
    assert np.allclose(weights[8:10], np.full(2, 1.0 / 2.0))


def test_equal_family_sampler_uses_fixed_asset_family_mapping() -> None:
    dataset = _PooledLikeDataset()
    sampler = make_balanced_group_sampler(dataset, asset_to_group={0: 0, 1: 1, 2: 1}, seed=7, mode="equal_family")
    assert sampler is not None
    weights = sampler.weights.detach().cpu().numpy()
    assert np.allclose(weights[:8], np.full(8, 1.0 / 8.0))
    assert np.allclose(weights[8:], np.full(4, 1.0 / 4.0))
    assert dataset_asset_ids(dataset).tolist() == [0] * 8 + [1] * 2 + [2] * 2
