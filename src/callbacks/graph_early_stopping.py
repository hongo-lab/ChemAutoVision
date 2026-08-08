"""Early stopping state management for graph-model training."""

from math import isfinite
from typing import Literal, Optional


class GraphEarlyStopping:
    """Track a validation metric and decide when training should stop.

    This class only manages early-stopping state. Saving and restoring the
    best Chemprop checkpoint remains the responsibility of the training loop.

    Args:
        patience: Number of consecutive non-improving epochs to tolerate.
        mode: ``"min"`` for metrics such as RMSE, or ``"max"`` for metrics
            such as ROC-AUC.
        min_delta: Minimum absolute change required to count as improvement.
            As in Keras, a negative value is treated as its absolute value.
    """

    def __init__(
        self,
        patience: int = 30,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TypeError("patience must be an integer")
        if patience < 0:
            raise ValueError("patience must be greater than or equal to 0")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")

        self.patience = patience
        self.mode = mode
        self.min_delta = abs(float(min_delta))
        self.reset()

    def reset(self) -> None:
        """Reset all state so the callback can be reused for another model."""
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.wait = 0
        self.stopped_epoch: Optional[int] = None
        self.should_stop = False

    def is_improvement(self, current: float) -> bool:
        """Return whether ``current`` improves on the best finite score."""
        current = float(current)
        if not isfinite(current):
            return False
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def update(self, current: float, epoch: int) -> bool:
        """Update state for an epoch and return whether training should stop."""
        if self.should_stop:
            return True

        current = float(current)
        if self.is_improvement(current):
            self.best_score = current
            self.best_epoch = epoch
            self.wait = 0
            return False

        self.wait += 1
        if self.wait >= self.patience:
            self.should_stop = True
            self.stopped_epoch = epoch

        return self.should_stop

