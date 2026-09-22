"""Training / evaluation loop for one outer fold.

Each epoch draws a fresh random subset of 100 patches per training patient,
optimises the attention parameters and the classification head with AdamW,
scores the inner validation set on all of its patches, steps the cosine
schedule and applies early stopping on the inner-validation AUC.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..evaluation.metrics import roc_auc
from ..models.mil import EncoderMIL
from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .early_stopping import EarlyStopping
from .losses import WeightedBCEWithLogits, combine_losses
from .optim import build_optimizer, build_scheduler


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


class FoldTrainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        device: torch.device,
        out_dir: Union[str, Path],
        fold: int,
        class_weights: Optional[Tuple[float, float]] = None,
    ):
        self.cfg = cfg
        self.train_cfg = cfg.training
        self.device = device
        self.model = model.to(device)
        self.out_dir = ensure_dir(out_dir)
        self.fold = int(fold)
        self.input_key = "features" if str(cfg.data.get("input", "features")) == "features" else "images"
        use_weights = bool(self.train_cfg.get("class_weighting", True))
        self.criterion = WeightedBCEWithLogits(class_weights if use_weights else None).to(device)
        self.amp = bool(self.train_cfg.get("amp", True)) and device.type == "cuda"
        self.scaler = _grad_scaler(self.amp)
        self.grad_clip = self.train_cfg.get("grad_clip")
        self.log = get_logger(f"hemil.fold{self.fold}")
        self.log.info("loss: %s", self.criterion)

    # ------------------------------------------------------------ helpers
    def _state_module(self) -> nn.Module:
        return self.model.trainable if isinstance(self.model, EncoderMIL) else self.model

    def _forward(self, batch: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        x = batch[self.input_key].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            outputs = self.model(**{self.input_key: x}, mask=mask, label=labels)
        logits = outputs["logits"].float()
        bag_loss = self.criterion(logits, labels)
        loss = combine_losses(bag_loss, {k: (v.float() if torch.is_tensor(v) else v) for k, v in outputs.items()})
        return loss, outputs, labels

    # -------------------------------------------------------------- epochs
    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, labels = self._forward(batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._state_module().parameters(), float(self.grad_clip))
            self.scaler.step(optimizer)
            self.scaler.update()
            total += float(loss.item()) * int(labels.numel())
            count += int(labels.numel())
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, collect: bool = False) -> Dict[str, object]:
        """Score every patient in ``loader`` on all of its patches."""
        self.model.eval()
        rows: List[dict] = []
        zs: List[torch.Tensor] = []
        attention: Dict[str, dict] = {}
        total, count = 0.0, 0
        for batch in loader:
            loss, outputs, labels = self._forward(batch)
            logits = outputs["logits"].float().detach().cpu().numpy().reshape(-1)
            scores = 1.0 / (1.0 + np.exp(-logits))
            n_tiles = batch["n_tiles"].numpy()
            for i, pid in enumerate(batch["patient_id"]):
                rows.append(
                    {
                        "patient_id": pid,
                        "slide_id": batch["slide_id"][i],
                        "y_true": int(labels[i].item()),
                        "logit": float(logits[i]),
                        "score": float(scores[i]),
                        "n_tiles": int(n_tiles[i]),
                    }
                )
                if collect and "attention" in outputs:
                    attention[pid] = {
                        "attention": outputs["attention"][i, : int(n_tiles[i])].float().detach().cpu(),
                        "tile_idx": batch["tile_idx"][i],
                        "coords": batch["coords"][i],
                    }
            if collect and "z" in outputs:
                zs.append(outputs["z"].float().detach().cpu())
            total += float(loss.item()) * int(labels.numel())
            count += int(labels.numel())
        df = pd.DataFrame(rows)
        result: Dict[str, object] = {
            "loss": total / max(count, 1),
            "auc": roc_auc(df["y_true"].to_numpy(), df["score"].to_numpy()) if len(df) else float("nan"),
            "predictions": df,
        }
        if collect:
            result["z"] = torch.cat(zs, dim=0) if zs else None
            result["attention"] = attention
        return result

    # ----------------------------------------------------------------- fit
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, object]:
        epochs = int(self.train_cfg.get("epochs", 80))
        optimizer = build_optimizer(self.model, self.train_cfg)
        scheduler = build_scheduler(optimizer, self.train_cfg)
        es_cfg = self.train_cfg.get("early_stopping", {}) or {}
        stopper = EarlyStopping(
            patience=int(es_cfg.get("patience", 15)),
            mode=str(es_cfg.get("mode", "max")),
            min_epochs=int(es_cfg.get("min_epochs", 0)),
        )
        history: List[dict] = []
        log_path = self.out_dir / "train_log.csv"
        self.log.info(
            "training %d epochs, batch %d patients, %d train / %d val patients",
            epochs,
            train_loader.batch_size,
            len(train_loader.dataset),
            len(val_loader.dataset),
        )
        t_start = time.time()
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, optimizer)
            val = self.evaluate(val_loader)
            lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()
            improved = stopper.step(float(val["auc"]), epoch)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_auc": val["auc"],
                "lr": lr,
                "improved": int(improved),
                "epoch_time_s": time.time() - t0,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(log_path, index=False)
            self.save_checkpoint("last.pt", epoch, float(val["auc"]))
            if improved:
                self.save_checkpoint("best.pt", epoch, float(val["auc"]))
            self.log.info(
                "epoch %3d/%d | train_loss %.4f | val_loss %.4f | val_auc %.4f%s | lr %.2e | %.1fs",
                epoch + 1,
                epochs,
                train_loss,
                val["loss"],
                val["auc"],
                " *" if improved else "",
                lr,
                row["epoch_time_s"],
            )
            if stopper.stopped:
                self.log.info("early stopping at epoch %d (best epoch %d)", epoch + 1, stopper.best_epoch + 1)
                break
        self.load_checkpoint(self.out_dir / "best.pt")
        return {
            "best_epoch": stopper.best_epoch + 1,
            "best_val_auc": stopper.best,
            "epochs_run": len(history),
            "train_time_s": time.time() - t_start,
            "history": history,
        }

    # --------------------------------------------------------- checkpoints
    def save_checkpoint(self, name: str, epoch: int, val_auc: float) -> Path:
        path = self.out_dir / name
        module = self._state_module()
        torch.save(
            {
                "model_state": module.state_dict(),
                "epoch": int(epoch + 1),
                "val_auc": float(val_auc),
                "fold": self.fold,
                "model": self.cfg.model.to_dict(),
                "encoder": str(self.cfg.data.get("encoder", "")),
                "label_scheme": str(self.cfg.data.get("label_scheme", "primary")),
                "trainable_parameters": sum(p.numel() for p in module.parameters()),
                "created": datetime.now().isoformat(timespec="seconds"),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> dict:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._state_module().load_state_dict(ckpt["model_state"], strict=True)
        return ckpt
