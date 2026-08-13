"""Custom PyTorch training and inference loops for Phase 2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from market_dynamics.training.callbacks import EarlyStopping
from market_dynamics.utils.torch_utils import mixed_precision_enabled


@dataclass
class TrainingResult:
    """Best validation state and full loss history for one model run."""

    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_loss: float
    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    training_config: dict[str, Any],
    device: torch.device,
) -> TrainingResult:
    """Train one model with validation early stopping and best-state restoration."""
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_scheduler_factor", 0.5)),
        patience=int(training_config.get("lr_scheduler_patience", 2)),
    )
    amp_enabled = mixed_precision_enabled(bool(training_config.get("mixed_precision", False)), device)
    scaler = _make_grad_scaler(device, amp_enabled)
    stopping = EarlyStopping(patience=int(training_config.get("early_stopping_patience", 5)))
    best_state = deepcopy(model.state_dict())
    result = TrainingResult(best_state, best_epoch=-1, best_validation_loss=float("inf"))

    for epoch in range(int(training_config.get("epochs", 20))):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip_norm=float(training_config.get("gradient_clip_norm", 1.0)),
        )
        validation_loss = _run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            gradient_clip_norm=None,
        )
        scheduler.step(validation_loss)
        result.train_losses.append(train_loss)
        result.validation_losses.append(validation_loss)
        result.learning_rates.append(float(optimizer.param_groups[0]["lr"]))

        if validation_loss < result.best_validation_loss:
            result.best_validation_loss = validation_loss
            result.best_epoch = epoch
            result.best_state_dict = deepcopy(model.state_dict())

        if stopping.step(validation_loss, epoch):
            break

    model.load_state_dict(result.best_state_dict)
    return result


def fit_model_with_pairwise_ranking(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    training_config: dict[str, Any],
    device: torch.device,
    pairwise_coefficient: float = 1.0,
    pointwise_coefficient: float = 1.0,
    maximum_pairs_per_asset: int | None = None,
) -> TrainingResult:
    """Fit a fixed pointwise/pairwise within-asset objective."""
    if pointwise_coefficient < 0.0 or pairwise_coefficient < 0.0:
        raise ValueError("Objective coefficients must be non-negative")
    if pointwise_coefficient == 0.0 and pairwise_coefficient == 0.0:
        raise ValueError("At least one objective coefficient must be positive")
    if maximum_pairs_per_asset is not None and maximum_pairs_per_asset < 1:
        raise ValueError("maximum_pairs_per_asset must be positive when supplied")
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_scheduler_factor", 0.5)),
        patience=int(training_config.get("lr_scheduler_patience", 2)),
    )
    amp_enabled = mixed_precision_enabled(bool(training_config.get("mixed_precision", False)), device)
    scaler = _make_grad_scaler(device, amp_enabled)
    stopping = EarlyStopping(patience=int(training_config.get("early_stopping_patience", 5)))
    result = TrainingResult(deepcopy(model.state_dict()), best_epoch=-1, best_validation_loss=float("inf"))
    for epoch in range(int(training_config.get("epochs", 20))):
        train_loss = _run_pairwise_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled,
            float(training_config.get("gradient_clip_norm", 1.0)), pairwise_coefficient,
            pointwise_coefficient, maximum_pairs_per_asset,
        )
        validation_loss = _run_pairwise_epoch(
            model, validation_loader, criterion, device, None, None, amp_enabled, None,
            pairwise_coefficient, pointwise_coefficient, maximum_pairs_per_asset,
        )
        scheduler.step(validation_loss)
        result.train_losses.append(train_loss)
        result.validation_losses.append(validation_loss)
        result.learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        if validation_loss < result.best_validation_loss:
            result.best_validation_loss = validation_loss
            result.best_epoch = epoch
            result.best_state_dict = deepcopy(model.state_dict())
        if stopping.step(validation_loss, epoch):
            break
    model.load_state_dict(result.best_state_dict)
    return result


def fit_model_with_explicit_pairs(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    training_config: dict[str, Any],
    device: torch.device,
    *,
    pointwise_coefficient: float,
    pairwise_coefficient: float,
) -> TrainingResult:
    """Fit on a frozen registry of same-asset positive-negative pairs."""
    if min(pointwise_coefficient, pairwise_coefficient) < 0.0:
        raise ValueError("Objective coefficients must be non-negative")
    if pointwise_coefficient == 0.0 and pairwise_coefficient == 0.0:
        raise ValueError("At least one objective coefficient must be positive")
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_scheduler_factor", 0.5)),
        patience=int(training_config.get("lr_scheduler_patience", 2)),
    )
    amp_enabled = mixed_precision_enabled(bool(training_config.get("mixed_precision", False)), device)
    scaler = _make_grad_scaler(device, amp_enabled)
    stopping = EarlyStopping(patience=int(training_config.get("early_stopping_patience", 5)))
    result = TrainingResult(deepcopy(model.state_dict()), best_epoch=-1, best_validation_loss=float("inf"))
    for epoch in range(int(training_config.get("epochs", 20))):
        train_loss = _run_explicit_pair_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            amp_enabled,
            float(training_config.get("gradient_clip_norm", 1.0)),
            pointwise_coefficient,
            pairwise_coefficient,
        )
        validation_loss = _run_explicit_pair_epoch(
            model,
            validation_loader,
            device,
            None,
            None,
            amp_enabled,
            None,
            pointwise_coefficient,
            pairwise_coefficient,
        )
        scheduler.step(validation_loss)
        result.train_losses.append(train_loss)
        result.validation_losses.append(validation_loss)
        result.learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        if validation_loss < result.best_validation_loss:
            result.best_validation_loss = validation_loss
            result.best_epoch = epoch
            result.best_state_dict = deepcopy(model.state_dict())
        if stopping.step(validation_loss, epoch):
            break
    model.load_state_dict(result.best_state_dict)
    return result


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true targets, predictions/probabilities and source indices."""
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in loader:
        features, targets, source_indices, asset_ids = _unpack_batch(batch)
        logits_or_values = _forward_model(model, features.to(device, non_blocking=True), asset_ids, device)
        if task == "classification":
            values = torch.sigmoid(logits_or_values)
        elif logits_or_values.ndim == 2 and logits_or_values.size(-1) == 2:
            values = logits_or_values[:, 0]
        else:
            values = logits_or_values
        y_true.append(targets.detach().cpu().numpy())
        y_pred.append(values.detach().cpu().numpy())
        indices.append(source_indices.detach().cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(indices)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: Any | None,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        features, targets, _, asset_ids = _unpack_batch(batch)
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = _forward_model(model, features, asset_ids, device)
            loss = criterion(outputs, targets)

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().item()) * len(features)
        total_samples += len(features)
    if total_samples == 0:
        raise RuntimeError("DataLoader produced no samples")
    return total_loss / total_samples


def _run_pairwise_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: Any | None,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
    pairwise_coefficient: float,
    pointwise_coefficient: float = 1.0,
    maximum_pairs_per_asset: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        features, targets, _, asset_ids = _unpack_batch(batch)
        if asset_ids is None:
            raise ValueError("Within-asset ranking requires asset identifiers")
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        asset_ids = asset_ids.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(features, asset_ids)
            ranking_logits = model.residual_logits(features) if hasattr(model, "residual_logits") else outputs
            pointwise = criterion(outputs, targets)
            pair_losses: list[torch.Tensor] = []
            for asset_id in torch.unique(asset_ids):
                mask = asset_ids.eq(asset_id)
                positive = ranking_logits[mask & targets.gt(0.5)]
                negative = ranking_logits[mask & targets.le(0.5)]
                if positive.numel() and negative.numel():
                    pair_losses.append(
                        _bounded_pairwise_logistic_loss(
                            positive,
                            negative,
                            maximum_pairs=maximum_pairs_per_asset,
                        )
                    )
            pairwise = torch.stack(pair_losses).mean() if pair_losses else outputs.sum() * 0.0
            loss = float(pointwise_coefficient) * pointwise + float(pairwise_coefficient) * pairwise
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().item()) * len(features)
        total_samples += len(features)
    if total_samples == 0:
        raise RuntimeError("DataLoader produced no samples")
    return total_loss / total_samples


def _run_explicit_pair_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: Any | None,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
    pointwise_coefficient: float,
    pairwise_coefficient: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_pairs = 0
    for positive, negative, asset_ids, pair_weights in loader:
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)
        asset_ids = asset_ids.to(device, non_blocking=True)
        pair_weights = pair_weights.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            positive_scores = _forward_model(model, positive, asset_ids, device)
            negative_scores = _forward_model(model, negative, asset_ids, device)
            pointwise = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    positive_scores,
                    torch.ones_like(positive_scores),
                    reduction="none",
                )
                + F.binary_cross_entropy_with_logits(
                    negative_scores,
                    torch.zeros_like(negative_scores),
                    reduction="none",
                )
            )
            pairwise = F.softplus(-(positive_scores - negative_scores))
            per_pair = float(pointwise_coefficient) * pointwise + float(pairwise_coefficient) * pairwise
            loss = torch.sum(per_pair * pair_weights) / torch.clamp(pair_weights.sum(), min=1e-12)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().item()) * len(positive)
        total_pairs += len(positive)
    if total_pairs == 0:
        raise RuntimeError("Pair DataLoader produced no pairs")
    return total_loss / total_pairs


def _bounded_pairwise_logistic_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    maximum_pairs: int | None,
) -> torch.Tensor:
    """Return deterministic positive-negative logistic loss with an optional cap."""
    differences = (positive[:, None] - negative[None, :]).reshape(-1)
    if maximum_pairs is not None and differences.numel() > maximum_pairs:
        positions = torch.linspace(
            0,
            differences.numel() - 1,
            steps=maximum_pairs,
            device=differences.device,
        ).round().long()
        differences = differences[positions]
    return F.softplus(-differences).mean()


def _make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - legacy PyTorch fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _unpack_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Accept local three-tensor datasets and pooled four-tensor datasets."""
    if len(batch) == 3:
        features, targets, source_indices = batch
        return features, targets, source_indices, None
    if len(batch) == 4:
        features, targets, source_indices, asset_ids = batch
        return features, targets, source_indices, asset_ids
    raise ValueError(f"Unexpected dataset batch size: {len(batch)}")


def _forward_model(
    model: nn.Module,
    features: torch.Tensor,
    asset_ids: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Pass asset ids only to pooled conditional models."""
    if asset_ids is None:
        return model(features)
    return model(features, asset_ids.to(device, non_blocking=True))
