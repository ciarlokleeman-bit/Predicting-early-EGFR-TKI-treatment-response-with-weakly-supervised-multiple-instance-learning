"""Attention pooling operators.

Gated attention (Ilse et al., ICML 2018):

    a_i = exp{ w^T (tanh(V h_i) * sigm(U h_i)) } / sum_k exp{ w^T (tanh(V h_k) * sigm(U h_k)) }
    z   = sum_i a_i h_i

with V, U in R^{256 x 2048} and w in R^{256} for ResNet-50 features.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Softmax over ``dim`` that assigns exactly zero weight to padded positions."""
    if mask is None:
        return torch.softmax(logits, dim=dim)
    neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg)
    weights = torch.softmax(logits, dim=dim)
    return weights * mask.to(weights.dtype)


class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 256, bias: bool = False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.V = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.U = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def attention_logits(self, h: torch.Tensor) -> torch.Tensor:
        """``h``: [B, N, D] -> un-normalised attention scores [B, N]."""
        gate = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        return self.w(gate).squeeze(-1)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z, a)`` with ``z``: [B, D] slide representation and ``a``: [B, N] attention weights."""
        scores = self.attention_logits(h)
        a = masked_softmax(scores, mask, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return z, a


class AttentionPooling(nn.Module):
    """Attention-based MIL without the sigmoid gate: a_i ∝ exp{ w^T tanh(V h_i) }."""

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 256, bias: bool = False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.V = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def attention_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.w(torch.tanh(self.V(h))).squeeze(-1)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_logits(h)
        a = masked_softmax(scores, mask, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return z, a
