"""Compatibility helpers for trusted Chemprop 1.x checkpoints."""

from collections.abc import Iterator
from contextlib import contextmanager
import os


_FORCE_NO_WEIGHTS_ONLY_LOAD = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


@contextmanager
def trusted_chemprop_checkpoint_loading() -> Iterator[None]:
    """Allow Chemprop 1.x to load its full, locally generated checkpoints.

    Chemprop 1.x stores an ``argparse.Namespace`` alongside the model state and
    calls ``torch.load`` without a ``weights_only`` argument. PyTorch 2.6+
    defaults such calls to ``weights_only=True``. The documented environment
    override makes omitted arguments behave as ``weights_only=False`` while
    this context is active.

    Only use this context for checkpoints produced by this application or
    obtained from another trusted source: full pickle loading can execute code.
    """
    previous_value = os.environ.get(_FORCE_NO_WEIGHTS_ONLY_LOAD)
    os.environ[_FORCE_NO_WEIGHTS_ONLY_LOAD] = "1"
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop(_FORCE_NO_WEIGHTS_ONLY_LOAD, None)
        else:
            os.environ[_FORCE_NO_WEIGHTS_ONLY_LOAD] = previous_value
