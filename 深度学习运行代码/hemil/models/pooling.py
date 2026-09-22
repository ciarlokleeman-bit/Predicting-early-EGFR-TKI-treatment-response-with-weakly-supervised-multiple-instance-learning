"""Non-learned bag pooling used as baselines."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """Global average of patch features (masked)."""

    def __init__(self, in_dim: int = 2048):
        super().__init__()
        self.in_dim = int(in_dim)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        m = mask.to(h.dtype)
        counts = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        z = (h * m.unsqueeze(-1)).sum(dim=1) / counts
        a = m / counts
        return z, a


class MaxPooling(nn.Module):
    """Element-wise maximum over patch features (masked)."""

    def __init__(self, in_dim: int = 2048):
        super().__init__()
        self.in_dim = int(in_dim)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        neg = torch.finfo(h.dtype).min
        z = h.masked_fill(~mask.unsqueeze(-1), neg).max(dim=1).values
        # for reporting only: fraction of feature dimensions each patch contributed
        argmax = h.masked_fill(~mask.unsqueeze(-1), neg).argmax(dim=1)            # [B, D]
        a = torch.zeros(h.shape[:2], dtype=h.dtype, device=h.device)
        a.scatter_add_(1, argmax, torch.ones_like(argmax, dtype=h.dtype))
        a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)
        return z, a
