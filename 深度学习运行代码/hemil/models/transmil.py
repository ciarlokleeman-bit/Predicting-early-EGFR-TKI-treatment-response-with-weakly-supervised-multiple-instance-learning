"""TransMIL (Shao et al., NeurIPS 2021).

Patch embeddings are projected to 512-d, padded to a square token grid,
prefixed with a class token and passed through two Nystrom self-attention
layers with a pyramid position encoding generator (PPEG) in between. The class
token after the final LayerNorm is the slide representation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Iterative Moore-Penrose pseudo-inverse used by Nystrom attention."""
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))
    m = x.shape[-1]
    eye = torch.eye(m, device=x.device, dtype=x.dtype).view(*([1] * (x.ndim - 2)), m, m)
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * eye - (xz @ (15 * eye - (xz @ (7 * eye - xz)))))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 33,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (residual_conv_kernel, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h, m, iters = self.heads, self.num_landmarks, self.pinv_iterations
        remainder = n % m
        if remainder > 0:
            x = F.pad(x, (0, 0, m - remainder, 0), value=0.0)
        n_pad = x.shape[1]

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n_pad, h, self.dim_head).transpose(1, 2) for t in qkv]   # [b, h, n_pad, d]
        q = q * self.scale

        l = n_pad // m
        q_land = q.reshape(b, h, m, l, self.dim_head).sum(dim=3) / l
        k_land = k.reshape(b, h, m, l, self.dim_head).sum(dim=3) / l

        sim1 = q @ k_land.transpose(-1, -2)          # [b, h, n_pad, m]
        sim2 = q_land @ k_land.transpose(-1, -2)     # [b, h, m, m]
        sim3 = q_land @ k.transpose(-1, -2)          # [b, h, m, n_pad]
        attn1, attn2, attn3 = sim1.softmax(-1), sim2.softmax(-1), sim3.softmax(-1)
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)      # [b, h, n_pad, d]
        if self.residual:
            out = out + self.res_conv(v)
        out = out.transpose(1, 2).reshape(b, n_pad, h * self.dim_head)
        out = self.to_out(out)
        return out[:, -n:]


class TransLayer(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, num_landmarks: int = 256, pinv_iterations: int = 6, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    """Pyramid position encoding generator: depthwise 7x7, 5x5 and 3x3 convolutions on the token grid."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, _, c = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        grid = feat_token.transpose(1, 2).reshape(b, c, height, width)
        grid = self.proj(grid) + grid + self.proj1(grid) + self.proj2(grid)
        tokens = grid.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), tokens), dim=1)


class TransMIL(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        embed_dim: int = 512,
        n_heads: int = 8,
        n_landmarks: int = 256,
        pinv_iterations: int = 6,
        attn_dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layer1 = TransLayer(embed_dim, n_heads, n_landmarks, pinv_iterations, attn_dropout)
        self.pos_layer = PPEG(embed_dim)
        self.layer2 = TransLayer(embed_dim, n_heads, n_landmarks, pinv_iterations, attn_dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self._fc2 = nn.Linear(embed_dim, n_outputs)
        self.n_outputs = int(n_outputs)

    def forward_bag(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``features``: [n, in_dim] for one slide."""
        h = self._fc1(features.unsqueeze(0))                     # [1, n, 512]
        n = h.shape[1]
        side = int(math.ceil(math.sqrt(n)))
        add = side * side - n
        if add > 0:
            h = torch.cat([h, h[:, :add, :]], dim=1)
        h = torch.cat((self.cls_token, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        cls = self.norm(h)[:, 0]                                # [1, 512]
        logits = self._fc2(cls)
        return {"logits": logits.squeeze(0), "z": cls.squeeze(0)}

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: object) -> Dict[str, torch.Tensor]:
        # the pseudo-inverse iteration is run in fp32 regardless of autocast
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            logits, zs = [], []
            for b in range(features.shape[0]):
                keep = mask[b] if mask is not None else torch.ones(features.shape[1], dtype=torch.bool, device=features.device)
                out = self.forward_bag(features[b][keep])
                logits.append(out["logits"])
                zs.append(out["z"])
            logits_t = torch.stack(logits)
            if self.n_outputs == 1:
                logits_t = logits_t.squeeze(-1)
            attention = (mask.float() / mask.float().sum(1, keepdim=True).clamp_min(1.0)) if mask is not None else None
            result: Dict[str, torch.Tensor] = {"logits": logits_t, "z": torch.stack(zs)}
            if attention is not None:
                result["attention"] = attention
            return result
