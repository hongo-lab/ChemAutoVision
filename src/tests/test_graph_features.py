from pathlib import Path

import numpy as np
import pytest

from data.graph_features import validate_heatmap_paths


def test_missing_heatmap_is_always_an_error(tmp_path: Path):
    with pytest.raises(ValueError, match="missing"):
        validate_heatmap_paths(
            {"gradcam": [tmp_path / "missing.npy"]},
            expected_count=1,
            expected_shape=(4, 4),
        )


def test_invalid_heatmap_is_always_an_error(tmp_path: Path):
    path = tmp_path / "invalid.npy"
    np.save(path, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="constant"):
        validate_heatmap_paths(
            {"gradcam": [path]}, expected_count=1, expected_shape=(4, 4)
        )
