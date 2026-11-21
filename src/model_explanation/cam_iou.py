"""Top-k Intersection-over-Union evaluation for CAM heatmaps."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from model_explanation.cam_generation import validate_heatmap


def binarize_top_k(heatmap: np.ndarray, k: float) -> np.ndarray:
    if not 0.0 < k <= 100.0:
        raise ValueError(f"k must be in (0, 100], got {k}")
    heatmap = validate_heatmap(heatmap)
    count = max(1, int(round(heatmap.size * k / 100.0)))
    order = np.argsort(-heatmap.reshape(-1), kind="stable")
    mask = np.zeros(heatmap.size, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(heatmap.shape)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"mask shapes differ: {mask_a.shape} and {mask_b.shape}")
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        raise ValueError("IoU is undefined for two empty masks")
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


def compute_pairwise_iou(heatmaps: list[np.ndarray], k: float) -> list[float]:
    masks = [binarize_top_k(heatmap, k) for heatmap in heatmaps]
    return [
        compute_iou(masks[left], masks[right])
        for left, right in combinations(range(len(masks)), 2)
    ]


def evaluate_iou(
    heatmap_dirs: list[Path],
    dir_labels: list[str],
    k_values: list[float],
    molecule_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(heatmap_dirs) < 2:
        raise ValueError("IoU evaluation requires at least two heatmap directories")
    if len(dir_labels) != len(heatmap_dirs):
        raise ValueError("dir_labels must match heatmap_dirs")

    records: list[dict[str, object]] = []
    for molecule_key in molecule_keys:
        paths = [directory / f"{molecule_key}.npy" for directory in heatmap_dirs]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing heatmaps for molecule {molecule_key}: {missing}"
            )
        heatmaps = [validate_heatmap(np.load(path)) for path in paths]
        shapes = {heatmap.shape for heatmap in heatmaps}
        if len(shapes) != 1:
            raise ValueError(
                f"heatmap shapes differ for molecule {molecule_key}: {sorted(shapes)}"
            )
        pairs = list(combinations(range(len(heatmaps)), 2))
        for k in k_values:
            values = compute_pairwise_iou(heatmaps, k)
            for (left, right), value in zip(pairs, values):
                records.append(
                    {
                        "molecule_key": molecule_key,
                        "k": k,
                        "pair_label": f"{dir_labels[left]}_vs_{dir_labels[right]}",
                        "iou": value,
                    }
                )

    detail = pd.DataFrame(records, columns=["molecule_key", "k", "pair_label", "iou"])
    summary = (
        detail.groupby(["k", "pair_label"])["iou"]
        .agg(mean_iou="mean", std_iou="std", n_molecules="count")
        .reset_index()
    )
    return detail, summary
