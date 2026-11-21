from pathlib import Path

import numpy as np
import pytest

from model_explanation.cam_iou import binarize_top_k, compute_iou, evaluate_iou


def test_binarize_top_k_selects_exact_pixel_count():
    heatmap = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
    assert binarize_top_k(heatmap, 10).sum() == 10


def test_compute_iou_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        compute_iou(np.ones((2, 2), bool), np.ones((3, 3), bool))


def test_evaluate_iou_rejects_missing_heatmap(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    heatmap = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    np.save(left / "42.npy", heatmap)
    with pytest.raises(FileNotFoundError, match="missing heatmaps"):
        evaluate_iou([left, right], ["left", "right"], [10.0], ["42"])
