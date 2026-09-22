from __future__ import annotations

import math


class EarlyStopping:
    """Stop when the monitored inner-validation metric has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 15, mode: str = "max", min_epochs: int = 0, min_delta: float = 0.0):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = int(patience)
        self.mode = mode
        self.min_epochs = int(min_epochs)
        self.min_delta = float(min_delta)
        self.best = -math.inf if mode == "max" else math.inf
        self.best_epoch = -1
        self.bad_epochs = 0
        self.stopped = False

    def improved(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record ``value`` for ``epoch``; return True when it is a new best."""
        if self.improved(value):
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        if epoch + 1 >= self.min_epochs and self.bad_epochs >= self.patience:
            self.stopped = True
        return False

    def state(self) -> dict:
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "patience": self.patience,
            "stopped": self.stopped,
        }
