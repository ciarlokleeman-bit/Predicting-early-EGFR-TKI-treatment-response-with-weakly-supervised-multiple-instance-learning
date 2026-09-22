"""Class-weighted binary cross-entropy.

    L = -[ w_R * y * log(sigma(y_hat)) + w_S * (1 - y) * log(1 - sigma(y_hat)) ]

with w_R = N / (2 N_R) for the early poor-response class and w_S = N / (2 N_S)
for the favourable class, both computed on the training partition.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogits(nn.Module):
    def __init__(self, class_weights: Optional[Tuple[float, float]] = None):
        super().__init__()
        w_pos, w_neg = class_weights if class_weights is not None else (1.0, 1.0)
        self.register_buffer("w_pos", torch.tensor(float(w_pos)))
        self.register_buffer("w_neg", torch.tensor(float(w_neg)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        weights = targets * self.w_pos + (1.0 - targets) * self.w_neg
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)

    def extra_repr(self) -> str:
        return f"w_R={float(self.w_pos):.4f}, w_S={float(self.w_neg):.4f}"


def combine_losses(bag_loss: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Add the CLAM instance-clustering term when the model produced one."""
    inst = outputs.get("instance_loss")
    if inst is None:
        return bag_loss
    bag_weight = outputs.get("bag_weight")
    bw = float(bag_weight) if bag_weight is not None else 0.7
    return bw * bag_loss + (1.0 - bw) * inst
