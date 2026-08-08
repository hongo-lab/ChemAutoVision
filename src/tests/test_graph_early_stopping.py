import math

import pytest

from callbacks.graph_early_stopping import GraphEarlyStopping


def test_max_mode_stops_after_consecutive_non_improvements():
    callback = GraphEarlyStopping(patience=2, mode="max")

    assert callback.update(0.70, epoch=0) is False
    assert callback.update(0.72, epoch=1) is False
    assert callback.update(0.71, epoch=2) is False
    assert callback.update(0.72, epoch=3) is True

    assert callback.best_score == pytest.approx(0.72)
    assert callback.best_epoch == 1
    assert callback.stopped_epoch == 3


def test_min_mode_resets_wait_after_improvement():
    callback = GraphEarlyStopping(patience=2, mode="min")

    assert callback.update(1.0, epoch=0) is False
    assert callback.update(1.1, epoch=1) is False
    assert callback.wait == 1
    assert callback.update(0.9, epoch=2) is False

    assert callback.wait == 0
    assert callback.best_score == pytest.approx(0.9)
    assert callback.best_epoch == 2


def test_min_delta_must_be_exceeded():
    callback = GraphEarlyStopping(patience=2, mode="max", min_delta=0.005)

    assert callback.update(0.80, epoch=0) is False
    assert callback.update(0.805, epoch=1) is False
    assert callback.update(0.804, epoch=2) is True
    assert callback.best_score == pytest.approx(0.80)


def test_non_finite_score_is_not_an_improvement():
    callback = GraphEarlyStopping(patience=1, mode="min")

    assert callback.update(1.0, epoch=0) is False
    assert callback.update(math.nan, epoch=1) is True
    assert callback.best_score == pytest.approx(1.0)


def test_reset_clears_state():
    callback = GraphEarlyStopping(patience=1, mode="min")
    callback.update(1.0, epoch=0)
    callback.update(1.1, epoch=1)

    callback.reset()

    assert callback.best_score is None
    assert callback.best_epoch is None
    assert callback.wait == 0
    assert callback.stopped_epoch is None
    assert callback.should_stop is False


@pytest.mark.parametrize("patience", [-1, -10])
def test_negative_patience_is_rejected(patience):
    with pytest.raises(ValueError):
        GraphEarlyStopping(patience=patience)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        GraphEarlyStopping(mode="auto")
