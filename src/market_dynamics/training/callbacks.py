"""Small callback-style utilities for custom PyTorch training."""

from __future__ import annotations


class EarlyStopping:
    """Stop after validation loss fails to improve for a fixed patience."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.wait = 0

    def step(self, value: float, epoch: int) -> bool:
        """Update state and return whether training should stop."""
        if value < self.best_loss - self.min_delta:
            self.best_loss = value
            self.best_epoch = epoch
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience
