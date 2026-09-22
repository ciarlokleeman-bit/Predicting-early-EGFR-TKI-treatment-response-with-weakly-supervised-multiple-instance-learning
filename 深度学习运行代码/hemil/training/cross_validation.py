"""Five-fold outer cross-validation with inner early stopping.

For every outer fold the model is trained on 64 patients, early-stopped on 16
inner-validation patients, the operating point (Youden's J) is chosen on the
same inner-validation predictions, and the 20 held-out patients are scored on
all of their patches. Out-of-fold predictions from the five folds are
concatenated for the pooled metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..config import Config, resolve_encoder_config, save_config
from ..data.clinical import class_weights, label_column, load_manifest
from ..data.datasets import FeatureBagDataset, TileBagDataset, build_dataloader
from ..data.splits import build_splits, load_splits, save_splits
from ..data.transforms import build_tile_transform
from ..encoders.factory import build_encoder
from ..evaluation.aggregate import summarise_cv
from ..evaluation.metrics import binary_metrics, youden_threshold
from ..models.factory import build_model, trainable_parameter_count
from ..utils.device import device_summary, resolve_device
from ..utils.io import ensure_dir, save_json
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import seed_everything
from .trainer import FoldTrainer

log = get_logger(__name__)


def _splits_for(cfg: Config, manifest: pd.DataFrame, label_col: str) -> List[dict]:
    scheme = str(cfg.data.get("label_scheme", "primary"))
    root = Path(cfg.paths.splits)
    split_dir = root if scheme == "primary" else root / scheme
    if split_dir.exists() and any(split_dir.glob("fold*.json")):
        splits = load_splits(split_dir)
        log.info("loaded %d folds from %s", len(splits), split_dir)
        return splits
    splits = build_splits(
        manifest,
        label_col=label_col,
        n_folds=int(cfg.cv.n_outer_folds),
        inner_val_size=int(cfg.cv.inner_val_size),
        seed=int(cfg.seed),
        shuffle=bool(cfg.cv.get("shuffle", True)),
    )
    save_splits(splits, split_dir)
    log.info("built %d stratified folds -> %s", len(splits), split_dir)
    return splits


def _make_dataset(cfg: Config, manifest: pd.DataFrame, ids: Sequence[str], label_col: str, train: bool, encoder_input_size: int):
    data_cfg = cfg.data
    patches = data_cfg.get("train_patches_per_bag", 100) if train else data_cfg.get("eval_patches_per_bag", "all")
    if str(data_cfg.get("input", "features")) == "features":
        return FeatureBagDataset(
            manifest,
            ids,
            features_root=cfg.paths.features,
            encoder_name=str(data_cfg.encoder),
            label_col=label_col,
            train=train,
            patches_per_bag=patches,
            augmentation=bool(data_cfg.get("augmentation", True)),
            seed=int(cfg.seed),
        )
    transform = build_tile_transform(
        train=train, input_size=encoder_input_size, augmentation=bool(data_cfg.get("augmentation", True))
    )
    return TileBagDataset(
        manifest,
        ids,
        tiles_root=cfg.paths.tiles,
        transform=transform,
        label_col=label_col,
        train=train,
        patches_per_bag=patches,
        seed=int(cfg.seed),
    )


def run_cross_validation(
    cfg: Config,
    configs_dir: str = "configs",
    folds: Optional[Sequence[int]] = None,
    resume: bool = False,
) -> Dict[str, object]:
    out_dir = ensure_dir(Path(cfg.output_root) / str(cfg.experiment_name))
    setup_logging(out_dir / "train.log")
    save_config(cfg, out_dir / "config.yaml")
    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.get("device", "auto")))
    log.info("experiment %s | device %s", cfg.experiment_name, device_summary(device))

    manifest = load_manifest(cfg.paths.manifest)
    label_col = label_column(str(cfg.data.get("label_scheme", "primary")))
    splits = _splits_for(cfg, manifest, label_col)
    fold_ids = list(folds) if folds is not None else [s["fold"] for s in splits]

    use_tiles = str(cfg.data.get("input", "features")) == "tiles"
    encoder = None
    input_size = 512
    if use_tiles:
        enc_cfg = resolve_encoder_config(cfg, configs_dir)
        encoder = build_encoder(enc_cfg)
        input_size = int(enc_cfg.get("input_size", 512))
        if int(cfg.model.in_dim) != int(enc_cfg.feature_dim):
            raise ValueError(f"model.in_dim={cfg.model.in_dim} but encoder {enc_cfg.name} gives {enc_cfg.feature_dim}-d features")

    batch_size = int(cfg.training.batch_size)
    eval_batch = int(cfg.training.get("eval_batch_size", 1 if use_tiles else batch_size))
    workers = int(cfg.data.get("num_workers", 0))
    pin = bool(cfg.data.get("pin_memory", True)) and device.type == "cuda"

    fold_rows: List[dict] = []
    oof_frames: List[pd.DataFrame] = []
    for split in splits:
        k = int(split["fold"])
        fold_dir = ensure_dir(out_dir / f"fold{k}")
        if k not in fold_ids:
            continue
        pred_path = fold_dir / "test_predictions.csv"
        if resume and pred_path.exists():
            log.info("fold %d already finished, reusing %s", k, pred_path)
            test_df = pd.read_csv(pred_path)
            thr = float(pd.read_json(fold_dir / "threshold.json", typ="series")["threshold"])
            fold_rows.append({"fold": k, **binary_metrics(test_df["y_true"], test_df["score"], thr)})
            oof_frames.append(test_df.assign(fold=k))
            continue

        seed_everything(int(cfg.seed) + k)
        train_ds = _make_dataset(cfg, manifest, split["train"], label_col, True, input_size)
        val_ds = _make_dataset(cfg, manifest, split["val"], label_col, False, input_size)
        test_ds = _make_dataset(cfg, manifest, split["test"], label_col, False, input_size)
        train_loader = build_dataloader(train_ds, batch_size, True, workers, pin, seed=int(cfg.seed) + k)
        val_loader = build_dataloader(val_ds, eval_batch, False, workers, pin, seed=int(cfg.seed))
        test_loader = build_dataloader(test_ds, eval_batch, False, workers, pin, seed=int(cfg.seed))

        w_pos, w_neg = class_weights(train_ds.labels())
        model = build_model(
            cfg.model,
            encoder=encoder,
            encode_chunk=int(cfg.training.get("encode_chunk", 256)),
            amp=bool(cfg.training.get("amp", True)),
        )
        log.info(
            "fold %d | %s | trainable parameters %s | class weights w_R=%.3f w_S=%.3f",
            k,
            cfg.model.name,
            f"{trainable_parameter_count(model):,}",
            w_pos,
            w_neg,
        )
        trainer = FoldTrainer(model, cfg, device, fold_dir, k, class_weights=(w_pos, w_neg))
        fit = trainer.fit(train_loader, val_loader)

        val = trainer.evaluate(val_loader)
        val_df: pd.DataFrame = val["predictions"]
        thr = youden_threshold(val_df["y_true"].to_numpy(), val_df["score"].to_numpy())
        val_df.to_csv(fold_dir / "val_predictions.csv", index=False)
        save_json(
            {"threshold": thr, "selection": str(cfg.training.get("threshold_selection", "youden")), "val_auc": val["auc"]},
            fold_dir / "threshold.json",
        )

        test = trainer.evaluate(test_loader, collect=True)
        test_df: pd.DataFrame = test["predictions"]
        test_df["pred"] = (test_df["score"] >= thr).astype(int)
        test_df["threshold"] = thr
        test_df["fold"] = k
        test_df.to_csv(pred_path, index=False)
        if test.get("z") is not None:
            torch.save({"patient_id": test_df["patient_id"].tolist(), "z": test["z"]}, fold_dir / "z_test.pt")
        if bool(cfg.training.get("save_attention", True)) and test.get("attention"):
            att_dir = ensure_dir(fold_dir / "attention")
            for pid, payload in test["attention"].items():
                torch.save(payload, att_dir / f"{pid}.pt")

        metrics = binary_metrics(test_df["y_true"].to_numpy(), test_df["score"].to_numpy(), thr)
        fold_rows.append(
            {
                "fold": k,
                **metrics,
                "best_epoch": fit["best_epoch"],
                "best_val_auc": fit["best_val_auc"],
                "epochs_run": fit["epochs_run"],
                "train_time_s": fit["train_time_s"],
            }
        )
        save_json(fold_rows[-1], fold_dir / "metrics.json")
        oof_frames.append(test_df)
        log.info("fold %d | test AUC %.3f | sens %.3f | spec %.3f | best epoch %d", k, metrics["auc"], metrics["sensitivity"], metrics["specificity"], fit["best_epoch"])

    oof = pd.concat(oof_frames, ignore_index=True).sort_values(["fold", "patient_id"])
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold")
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary = summarise_cv(
        fold_metrics,
        oof,
        n_bootstrap=int(cfg.evaluation.get("bootstrap_replicates", 1000)),
        ci_level=float(cfg.evaluation.get("ci_level", 0.95)),
        ece_bins=int(cfg.evaluation.get("ece_bins", 10)),
        high_specificity=float(cfg.evaluation.get("high_specificity", 0.95)),
        calibration_fit=str(cfg.evaluation.get("calibration_fit", "linear")),
        seed=int(cfg.seed),
    )
    summary["experiment_name"] = str(cfg.experiment_name)
    summary["model"] = str(cfg.model.name)
    summary["encoder"] = str(cfg.data.encoder)
    summary["label_scheme"] = str(cfg.data.get("label_scheme", "primary"))
    save_json(summary, out_dir / "cv_summary.json")
    log.info(
        "%s | fold-mean AUC %.3f +- %.3f | pooled AUC %.3f (%.3f-%.3f)",
        cfg.experiment_name,
        summary["fold_mean"]["auc"],
        summary["fold_sd"]["auc"],
        summary["pooled"]["auc"],
        summary["pooled"]["auc_ci"][0],
        summary["pooled"]["auc_ci"][1],
    )
    return summary
