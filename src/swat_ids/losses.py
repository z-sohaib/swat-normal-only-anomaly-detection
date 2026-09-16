from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class EQLv2Loss(nn.Module):
    """Practical EQL v2-style loss for long-tailed multiclass IDS experiments.

    The original EQL v2 was proposed for long-tailed object detection. This
    implementation keeps the core thesis idea: rare classes receive larger loss
    weights so majority classes do not dominate the gradient signal.
    """

    def __init__(self, labels: np.ndarray, num_classes: int, gamma: float = 2.0) -> None:
        super().__init__()
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0.0] = 1.0
        frequency = counts / counts.sum()
        inverse = (1.0 - frequency) ** gamma
        weights = inverse / inverse.mean()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights.to(logits.device))


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, labels: np.ndarray, num_classes: int, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weights", class_weights(labels, num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        alpha = self.weights.to(logits.device).gather(0, targets)
        return (-alpha * ((1.0 - pt) ** self.gamma) * log_pt).mean()


def build_loss(
    name: str,
    labels: np.ndarray,
    num_classes: int,
    focal_gamma: float = 2.0,
) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights(labels, num_classes))
    if name == "eql_v2":
        return EQLv2Loss(labels, num_classes)
    if name == "class_balanced_focal":
        return ClassBalancedFocalLoss(labels, num_classes, gamma=focal_gamma)
    raise ValueError(f"Unsupported loss: {name}")
