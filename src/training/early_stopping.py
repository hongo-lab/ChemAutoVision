"""Helpers connecting early-stopping decisions to checkpoint persistence."""

from collections.abc import Callable
from typing import TypeVar

from callbacks import GraphEarlyStopping


Checkpoint = TypeVar("Checkpoint")


def update_early_stopping_and_save(
    early_stopping: GraphEarlyStopping,
    current_score: float,
    epoch: int,
    save_best: Callable[[], None],
) -> bool:
    """Update early stopping and save only when ``current_score`` is best."""
    improved = early_stopping.is_improvement(current_score)
    should_stop = early_stopping.update(current_score, epoch)

    if improved:
        save_best()

    return should_stop


def get_best_checkpoint_path(
    early_stopping_results: list[dict], model_idx: int = 0
) -> str:
    """Return the checkpoint restored for the requested ensemble member."""
    for result in early_stopping_results:
        if result["model_idx"] == model_idx:
            return result["checkpoint_path"]

    raise ValueError(f"No early-stopping result found for model {model_idx}")


def restore_best_checkpoint(
    checkpoint_path: str, load_best: Callable[[str], Checkpoint]
) -> Checkpoint:
    """Load the same checkpoint path retained by early stopping."""
    return load_best(checkpoint_path)
