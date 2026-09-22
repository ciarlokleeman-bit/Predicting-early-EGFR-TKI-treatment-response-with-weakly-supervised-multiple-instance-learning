"""CLAM single-branch (Lu et al., Nature Biomedical Engineering 2021).

Frozen ResNet-50 features are projected to 512-d, pooled with gated attention
and classified. An instance-level clustering branch pulls the k highest
attention patches of a bag towards its class and pushes the k lowest away,
which is the "clustering constraint" of the original method.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GatedAttentionPooling, masked_softmax


class CLAM_SB(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        embed_dim: int = 512,
        attention_hidden: int = 256,
        dropout: float = 0.25,
        k_sample: int = 8,
        instance_loss: str = "ce",
        bag_weight: float = 0.7,
        subtyping: bool = False,
        n_outputs: int = 1,
        n_instance_classes: int = 2,
    ):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention = GatedAttentionPooling(embed_dim, attention_hidden, bias=True)
        self.classifier = nn.Linear(embed_dim, n_outputs)
        self.instance_classifiers = nn.ModuleList([nn.Linear(embed_dim, 2) for _ in range(n_instance_classes)])
        self.k_sample = int(k_sample)
        self.instance_loss_type = instance_loss
        self.bag_weight = float(bag_weight)
        self.subtyping = bool(subtyping)
        self.n_outputs = int(n_outputs)

    # ------------------------------------------------------------ instance
    def _instance_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.instance_loss_type == "svm":
            return F.multi_margin_loss(logits, targets)
        return F.cross_entropy(logits, targets)

    def _topk(self, a: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
        return torch.topk(a, k, dim=0, largest=largest).indices

    def inst_eval(self, a: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> torch.Tensor:
        """In-class bag: top-k patches labelled 1, bottom-k labelled 0."""
        k = min(self.k_sample, max(1, h.shape[0] // 2))
        top = h[self._topk(a, k, largest=True)]
        bottom = h[self._topk(a, k, largest=False)]
        logits = classifier(torch.cat([top, bottom], dim=0))
        targets = torch.cat([torch.ones(k), torch.zeros(k)]).long().to(h.device)
        return self._instance_loss(logits, targets)

    def inst_eval_out(self, a: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> torch.Tensor:
        """Out-of-class bag (subtyping only): top-k patches labelled 0."""
        k = min(self.k_sample, h.shape[0])
        top = h[self._topk(a, k, largest=True)]
        logits = classifier(top)
        targets = torch.zeros(k, dtype=torch.long, device=h.device)
        return self._instance_loss(logits, targets)

    # ------------------------------------------------------------- forward
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        instance_eval: bool = True,
    ) -> Dict[str, torch.Tensor]:
        h = self.fc(features)                                     # [B, N, 512]
        scores = self.attention.attention_logits(h)               # [B, N]
        a = masked_softmax(scores, mask, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)               # [B, 512]
        logits = self.classifier(z)
        if self.n_outputs == 1:
            logits = logits.squeeze(-1)

        out: Dict[str, torch.Tensor] = {"logits": logits, "attention": a, "z": z}
        if instance_eval and label is not None and self.training:
            losses = []
            for b in range(h.shape[0]):
                keep = mask[b] if mask is not None else torch.ones(h.shape[1], dtype=torch.bool, device=h.device)
                hb, ab = h[b][keep], a[b][keep]
                if hb.shape[0] < 2:
                    continue
                y = int(label[b].item())
                for c, clf in enumerate(self.instance_classifiers):
                    if c == y:
                        losses.append(self.inst_eval(ab, hb, clf))
                    elif self.subtyping:
                        losses.append(self.inst_eval_out(ab, hb, clf))
            if losses:
                inst = torch.stack(losses).mean()
                out["instance_loss"] = inst
                out["bag_weight"] = torch.tensor(self.bag_weight, device=h.device)
        return out
