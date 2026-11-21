import numpy as np
import pytest

from model_explanation.cam_faithfulness import (
    _top_n_mask,
    evaluate_faithfulness,
)


class ConstantModel:
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, inputs, training=False):
        return np.full((len(inputs), 1), self.probability, dtype=np.float32)


def test_top_n_mask_selects_exact_count_even_with_ties():
    heatmap = np.ones((10, 10), dtype=np.float32)
    assert _top_n_mask(heatmap, 0.1).sum() == 10


def test_faithfulness_uses_selected_negative_class_score():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    heatmap = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    result = evaluate_faithfulness(
        ConstantModel(0.2), image, heatmap, steps=2, target_class=0
    )
    assert np.allclose(result["deletion_scores"], 0.8)
    assert np.allclose(result["insertion_scores"], 0.8)


def test_faithfulness_rejects_zero_steps():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    heatmap = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    with pytest.raises(ValueError, match="steps"):
        evaluate_faithfulness(
            ConstantModel(0.2), image, heatmap, steps=0, target_class=1
        )
