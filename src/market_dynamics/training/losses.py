"""Task-aware Phase 2 loss construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SigmoidFocalLoss(nn.Module):
    """Binary focal loss for imbalanced stress classification."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _smooth_binary_targets(targets, self.label_smoothing)
        probabilities = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - targets)
        return (alpha_t * ((1.0 - p_t).clamp_min(1e-8) ** self.gamma) * ce).mean()


class SoftF1Loss(nn.Module):
    """Differentiable binary soft-F1 surrogate for minority-class stress targets."""

    def __init__(self, eps: float = 1e-7, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _smooth_binary_targets(targets, self.label_smoothing)
        probabilities = torch.sigmoid(logits)
        true_positive = (probabilities * targets).sum()
        false_positive = (probabilities * (1.0 - targets)).sum()
        false_negative = ((1.0 - probabilities) * targets).sum()
        soft_f1 = (2.0 * true_positive + self.eps) / (
            2.0 * true_positive + false_positive + false_negative + self.eps
        )
        return 1.0 - soft_f1


class PairwiseLogisticRankingLoss(nn.Module):
    """Logistic loss for positive-minus-negative scores within one asset."""

    def forward(self, positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
        if positive_logits.shape != negative_logits.shape:
            raise ValueError("positive and negative logits must have the same shape")
        return F.softplus(-(positive_logits - negative_logits)).mean()


class SmoothedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with symmetric binary label smoothing."""

    def __init__(self, pos_weight: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            _smooth_binary_targets(targets, self.label_smoothing),
            pos_weight=self.pos_weight,
        )


class GaussianNLLFromOutput(nn.Module):
    """Gaussian NLL for models that output [mean, log_variance]."""

    def __init__(self, min_log_variance: float = -10.0, max_log_variance: float = 5.0) -> None:
        super().__init__()
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if output.ndim != 2 or output.size(-1) != 2:
            raise ValueError("GaussianNLLFromOutput expects model output shape [batch, 2]")
        mean = output[:, 0]
        log_variance = output[:, 1].clamp(self.min_log_variance, self.max_log_variance)
        return 0.5 * (log_variance + (targets - mean).pow(2) / torch.exp(log_variance)).mean()


def _smooth_binary_targets(targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    if label_smoothing <= 0.0:
        return targets.float()
    epsilon = min(max(float(label_smoothing), 0.0), 0.499)
    return targets.float() * (1.0 - epsilon) + 0.5 * epsilon


def task_from_target(target_column: str) -> str:
    """Infer model task from the project target naming convention."""
    if target_column.startswith("target_direction_") or target_column.startswith("target_stress_"):
        return "classification"
    if "realized_vol" in target_column:
        return "regression"
    raise ValueError(f"Cannot infer task from target column: {target_column}")


def build_loss(
    task: str,
    train_targets: np.ndarray,
    training_config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Build a leakage-safe loss using class weights from training labels only."""
    if task == "classification":
        loss_name = str(training_config.get("classification_loss", "bce")).lower()
        label_smoothing = float(training_config.get("label_smoothing", 0.0))
        positives = float(np.sum(train_targets > 0.5))
        negatives = float(np.sum(train_targets <= 0.5))
        alpha = float(training_config.get("focal_alpha", 1.0))
        if training_config.get("use_pos_weight", True) and positives > 0.0 and negatives > 0.0:
            alpha = float(training_config.get("focal_alpha", negatives / positives))
        if loss_name == "focal":
            return SigmoidFocalLoss(
                alpha=alpha,
                gamma=float(training_config.get("focal_gamma", 2.0)),
                label_smoothing=label_smoothing,
            )
        if loss_name in {"soft_f1", "softf1"}:
            return SoftF1Loss(label_smoothing=label_smoothing)
        if training_config.get("use_pos_weight", True):
            if positives > 0.0 and negatives > 0.0:
                pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
                return SmoothedBCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=label_smoothing)
        return SmoothedBCEWithLogitsLoss(label_smoothing=label_smoothing)
    if task == "regression":
        regression_loss = str(training_config.get("regression_loss", "huber")).lower()
        if regression_loss in {"gaussian_nll", "heteroscedastic_gaussian_nll"}:
            return GaussianNLLFromOutput(
                min_log_variance=float(training_config.get("min_log_variance", -10.0)),
                max_log_variance=float(training_config.get("max_log_variance", 5.0)),
            )
        if regression_loss == "mse":
            return nn.MSELoss()
        return nn.HuberLoss(delta=float(training_config.get("huber_delta", 1.0)))
    raise ValueError(f"Unsupported task: {task}")
