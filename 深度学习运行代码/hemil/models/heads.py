"""Slide-level classification head.

    z_out = W2 . ReLU(W1 z + b1) + b2         W1 in R^{256 x 2048}, W2 in R^{256 x 256}
    y_hat = w_c^T Dropout(z_out) + b_c        p = 0.5, scalar logit -> sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 256, dropout: float = 0.5, n_outputs: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, n_outputs)
        self.n_outputs = int(n_outputs)

    def mlp(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(z)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_out = self.dropout(self.mlp(z))
        logits = self.classifier(z_out)
        if self.n_outputs == 1:
            logits = logits.squeeze(-1)
        return logits
