"""Single-stream attention MIL: (frozen encoder) -> pooling -> MLP head -> logit."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..encoders.factory import FrozenEncoder
from .heads import ClassifierHead


class AttentionMIL(nn.Module):
    """Pooling operator followed by the two-layer MLP classifier.

    ``forward`` returns a dict with

        logits     [B]      poor-response logit
        attention  [B, N]   patch weights (zero on padding)
        z          [B, D]   slide-level representation used for t-SNE
    """

    def __init__(self, pooling: nn.Module, head: ClassifierHead):
        super().__init__()
        self.pooling = pooling
        self.head = head

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: object) -> Dict[str, torch.Tensor]:
        z, a = self.pooling(features, mask)
        logits = self.head(z)
        return {"logits": logits, "attention": a, "z": z}

    def attention_weights(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            _, a = self.pooling(features, mask)
        return a


class EncoderMIL(nn.Module):
    """Runs the frozen encoder on raw tiles before the MIL model (``data.input: tiles``).

    Only the MIL model's parameters are trainable. The encoder forward is
    gradient-free, so Grad-CAM uses ``hemil.interpretability.gradcam`` which
    re-enables activation gradients explicitly.
    """

    def __init__(self, encoder: FrozenEncoder, mil: nn.Module, encode_chunk: int = 256, amp: bool = True):
        super().__init__()
        self.encoder = encoder
        self.mil = mil
        self.encode_chunk = int(encode_chunk)
        self.amp = bool(amp)

    def train(self, mode: bool = True) -> "EncoderMIL":  # noqa: D401
        super().train(mode)
        self.encoder.train(False)
        return self

    def encode(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_bags(images, mask, chunk=self.encode_chunk, amp=self.amp)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Dict[str, torch.Tensor]:
        if features is None:
            if images is None:
                raise ValueError("either images or features must be given")
            if mask is None:
                mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
            features = self.encode(images, mask)
        return self.mil(features, mask, **kwargs)

    @property
    def trainable(self) -> nn.Module:
        return self.mil
