"""AdamW (lr 1e-4, weight decay 1e-4) with cosine annealing and warm restarts."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LRScheduler


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def build_optimizer(model: torch.nn.Module, train_cfg: Mapping) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    betas = tuple(train_cfg.get("betas", (0.9, 0.999)))
    params = list(trainable_parameters(model))
    if not params:
        raise ValueError("model has no trainable parameters")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=float(train_cfg.get("momentum", 0.9)))
    raise ValueError(f"unknown optimizer {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Mapping) -> Optional[LRScheduler]:
    sched = train_cfg.get("scheduler") or {}
    name = str(sched.get("name", "cosine_warm_restarts")).lower()
    epochs = int(train_cfg.get("epochs", 80))
    if name in {"none", "constant"}:
        return None
    if name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(sched.get("T_0", 20)),
            T_mult=int(sched.get("T_mult", 1)),
            eta_min=float(sched.get("eta_min", 1e-6)),
        )
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(sched.get("eta_min", 1e-6)))
    raise ValueError(f"unknown scheduler {name!r}")
