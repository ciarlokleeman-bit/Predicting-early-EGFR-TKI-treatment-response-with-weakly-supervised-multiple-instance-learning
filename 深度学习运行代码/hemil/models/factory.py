from __future__ import annotations

from typing import Mapping, Optional

import torch.nn as nn

from ..encoders.factory import FrozenEncoder
from .attention import AttentionPooling, GatedAttentionPooling
from .clam import CLAM_SB
from .heads import ClassifierHead
from .mil import AttentionMIL, EncoderMIL
from .pooling import MaxPooling, MeanPooling
from .transmil import TransMIL

POOLING_MODELS = {"gated_attention", "abmil", "mean_pool", "max_pool"}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mil(model_cfg: Mapping) -> nn.Module:
    name = str(model_cfg["name"]).lower()
    in_dim = int(model_cfg.get("in_dim", 2048))
    n_outputs = int(model_cfg.get("n_outputs", 1))

    if name in POOLING_MODELS:
        if name == "gated_attention":
            pooling: nn.Module = GatedAttentionPooling(
                in_dim, int(model_cfg.get("attention_hidden", 256)), bias=bool(model_cfg.get("attention_bias", False))
            )
        elif name == "abmil":
            pooling = AttentionPooling(
                in_dim, int(model_cfg.get("attention_hidden", 256)), bias=bool(model_cfg.get("attention_bias", False))
            )
        elif name == "mean_pool":
            pooling = MeanPooling(in_dim)
        else:
            pooling = MaxPooling(in_dim)
        head = ClassifierHead(
            in_dim=in_dim,
            hidden_dim=int(model_cfg.get("mlp_hidden", 256)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            n_outputs=n_outputs,
        )
        return AttentionMIL(pooling, head)

    if name == "clam_sb":
        return CLAM_SB(
            in_dim=in_dim,
            embed_dim=int(model_cfg.get("embed_dim", 512)),
            attention_hidden=int(model_cfg.get("attention_hidden", 256)),
            dropout=float(model_cfg.get("dropout", 0.25)),
            k_sample=int(model_cfg.get("k_sample", 8)),
            instance_loss=str(model_cfg.get("instance_loss", "ce")),
            bag_weight=float(model_cfg.get("bag_weight", 0.7)),
            subtyping=bool(model_cfg.get("subtyping", False)),
            n_outputs=n_outputs,
        )

    if name == "transmil":
        return TransMIL(
            in_dim=in_dim,
            embed_dim=int(model_cfg.get("embed_dim", 512)),
            n_heads=int(model_cfg.get("n_heads", 8)),
            n_landmarks=int(model_cfg.get("n_landmarks", 256)),
            pinv_iterations=int(model_cfg.get("pinv_iterations", 6)),
            attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
            n_outputs=n_outputs,
        )

    raise ValueError(f"unknown model {name!r}")


def build_model(
    model_cfg: Mapping,
    encoder: Optional[FrozenEncoder] = None,
    encode_chunk: int = 256,
    amp: bool = True,
) -> nn.Module:
    """Build the MIL model; wrap it with the frozen encoder when training from tiles."""
    mil = build_mil(model_cfg)
    if encoder is None:
        return mil
    return EncoderMIL(encoder, mil, encode_chunk=encode_chunk, amp=amp)
