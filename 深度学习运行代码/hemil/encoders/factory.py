"""Frozen patch encoders.

All encoders are ImageNet (or, for UNI, histology) pretrained networks with
the classification layer removed. Weights are frozen: gradients only reach
the attention parameters and the classification head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from ..data.transforms import Normalize
from ..utils.logging import get_logger

log = get_logger(__name__)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


class FrozenEncoder(nn.Module):
    """Wraps a backbone and keeps it in eval mode with ``requires_grad = False``."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        normalize: Normalize,
        input_size: int,
        name: str,
        gradcam_layer: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.normalize = normalize
        self.input_size = int(input_size)
        self.name = name
        self.gradcam_layer = gradcam_layer
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenEncoder":  # noqa: D401
        # the encoder never leaves eval mode (BatchNorm statistics stay frozen)
        super().train(False)
        return self

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: uint8 or float [n, 3, H, W] -> features [n, feature_dim]."""
        return self.backbone(self.prepare(x))

    @torch.no_grad()
    def encode(self, x: torch.Tensor, chunk: int = 256, amp: bool = True) -> torch.Tensor:
        """Chunked, gradient-free encoding of a stack of tiles."""
        outs = []
        device_type = "cuda" if x.is_cuda else "cpu"
        for start in range(0, x.shape[0], max(1, chunk)):
            piece = x[start : start + chunk]
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and x.is_cuda):
                outs.append(self.forward(piece).float())
        if not outs:
            return x.new_zeros((0, self.feature_dim), dtype=torch.float32)
        return torch.cat(outs, dim=0)

    def encode_bags(self, images: torch.Tensor, mask: torch.Tensor, chunk: int = 256, amp: bool = True) -> torch.Tensor:
        """``images``: [B, N, 3, H, W] padded bags -> features [B, N, D] (padding rows are zero)."""
        b, n = images.shape[:2]
        flat = images.reshape(b * n, *images.shape[2:])
        keep = mask.reshape(-1)
        feats = flat.new_zeros((b * n, self.feature_dim), dtype=torch.float32)
        if keep.any():
            feats[keep] = self.encode(flat[keep], chunk=chunk, amp=amp)
        return feats.view(b, n, self.feature_dim)

    def target_layer(self) -> nn.Module:
        if not self.gradcam_layer:
            raise ValueError(f"encoder {self.name} has no convolutional Grad-CAM layer configured")
        module: nn.Module = self.backbone
        for part in self.gradcam_layer.split("."):
            module = getattr(module, part)
        return module

    def extra_repr(self) -> str:
        return f"name={self.name}, feature_dim={self.feature_dim}, input_size={self.input_size}, frozen=True"


# ------------------------------------------------------------------ builders
def _torchvision_backbone(name: str, weights: Optional[str]) -> Tuple[nn.Module, int]:
    if name == "resnet50":
        w = tvm.ResNet50_Weights[weights] if weights else None
        net = tvm.resnet50(weights=w)
        net.fc = nn.Identity()            # global average pooled 2048-d features
        return net, 2048
    if name == "densenet121":
        w = tvm.DenseNet121_Weights[weights] if weights else None
        net = tvm.densenet121(weights=w)
        net.classifier = nn.Identity()    # relu + global average pool + flatten -> 1024-d
        return net, 1024
    raise ValueError(f"unsupported torchvision backbone {name!r}")


def _timm_backbone(timm_name: str, pretrained: bool = True, **kwargs) -> Tuple[nn.Module, int]:
    import timm

    net = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, **kwargs)
    return net, int(net.num_features)


def _local_vit_backbone(timm_name: str, weights: str, vit_kwargs: Mapping) -> Tuple[nn.Module, int]:
    import timm

    net = timm.create_model(timm_name, pretrained=False, **dict(vit_kwargs))
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the UNI weights from the Hugging Face hub "
            "(MahmoodLab/UNI, licence acceptance required) and place pytorch_model.bin there."
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.startswith("head")]
    if missing or unexpected:
        raise RuntimeError(f"UNI weights do not match the ViT-L/16 definition: missing={missing[:5]} unexpected={unexpected[:5]}")
    return net, int(net.num_features)


def build_encoder(enc_cfg: Mapping) -> FrozenEncoder:
    source = str(enc_cfg.get("source", "torchvision"))
    name = str(enc_cfg["name"])
    if source == "torchvision":
        backbone, dim = _torchvision_backbone(name, enc_cfg.get("weights"))
    elif source == "timm":
        backbone, dim = _timm_backbone(str(enc_cfg["timm_name"]))
    elif source == "local_vit":
        backbone, dim = _local_vit_backbone(str(enc_cfg["timm_name"]), str(enc_cfg["weights"]), enc_cfg.get("vit_kwargs", {}))
    else:
        raise ValueError(f"unknown encoder source {source!r}")

    declared = int(enc_cfg.get("feature_dim", dim))
    if declared != dim:
        raise ValueError(f"encoder {name}: config says feature_dim={declared} but the network produces {dim}")
    norm_cfg = enc_cfg.get("normalize", {})
    normalize = Normalize(norm_cfg.get("mean", (0.485, 0.456, 0.406)), norm_cfg.get("std", (0.229, 0.224, 0.225)))
    enc = FrozenEncoder(
        backbone,
        feature_dim=dim,
        normalize=normalize,
        input_size=int(enc_cfg.get("input_size", 512)),
        name=name,
        gradcam_layer=enc_cfg.get("gradcam_layer"),
    )
    log.info("encoder %s: %.1fM frozen parameters, %d-d features", name, count_parameters(enc) / 1e6, dim)
    return enc
