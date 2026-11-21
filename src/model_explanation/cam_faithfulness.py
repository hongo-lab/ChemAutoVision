"""Deletion and insertion faithfulness metrics for CAM heatmaps."""

from __future__ import annotations

import numpy as np

from model_explanation.cam_generation import binary_target_score, validate_heatmap


def _top_n_mask(heatmap: np.ndarray, ratio: float) -> np.ndarray:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    count = int(round(heatmap.size * ratio))
    mask = np.zeros(heatmap.size, dtype=bool)
    if count:
        # Stable ordering makes ties deterministic while selecting exactly N pixels.
        order = np.argsort(-heatmap.reshape(-1), kind="stable")
        mask[order[:count]] = True
    return mask.reshape(heatmap.shape)


def mask_top_ratio(image: np.ndarray, heatmap: np.ndarray, ratio: float) -> np.ndarray:
    heatmap = validate_heatmap(heatmap, image.shape[:2])
    masked = image.copy()
    masked[_top_n_mask(heatmap, ratio)] = 255
    return masked


def reveal_top_ratio(
    image: np.ndarray, heatmap: np.ndarray, ratio: float
) -> np.ndarray:
    heatmap = validate_heatmap(heatmap, image.shape[:2])
    revealed = np.full_like(image, 255)
    mask = _top_n_mask(heatmap, ratio)
    revealed[mask] = image[mask]
    return revealed


def _predict(
    model, image: np.ndarray, task_type: str, target_class: int | None
) -> float:
    inputs = image.astype(np.float32)[None] / 255.0
    predictions = model(inputs, training=False)
    if task_type == "regression":
        return float(np.asarray(predictions).reshape(-1)[0])
    if target_class is None:
        raise ValueError("classification faithfulness requires target_class")
    return binary_target_score(predictions, target_class)


def compute_deletion_curve(
    model,
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    steps: int = 10,
    task_type: str = "classification",
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    ratios = np.linspace(0.0, 1.0, steps + 1)
    scores = [
        _predict(
            model, mask_top_ratio(image, heatmap, float(ratio)), task_type, target_class
        )
        for ratio in ratios
    ]
    return ratios, np.asarray(scores, dtype=np.float32)


def compute_insertion_curve(
    model,
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    steps: int = 10,
    task_type: str = "classification",
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    ratios = np.linspace(0.0, 1.0, steps + 1)
    scores = [
        _predict(
            model,
            reveal_top_ratio(image, heatmap, float(ratio)),
            task_type,
            target_class,
        )
        for ratio in ratios
    ]
    return ratios, np.asarray(scores, dtype=np.float32)


def auc_from_curve(ratios: np.ndarray, scores: np.ndarray) -> float:
    return float(np.trapz(scores, ratios))


def evaluate_faithfulness(
    model,
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    steps: int = 10,
    task_type: str = "classification",
    target_class: int | None = None,
) -> dict[str, object]:
    deletion_x, deletion_y = compute_deletion_curve(
        model,
        image,
        heatmap,
        steps=steps,
        task_type=task_type,
        target_class=target_class,
    )
    insertion_x, insertion_y = compute_insertion_curve(
        model,
        image,
        heatmap,
        steps=steps,
        task_type=task_type,
        target_class=target_class,
    )
    return {
        "deletion_auc": auc_from_curve(deletion_x, deletion_y),
        "insertion_auc": auc_from_curve(insertion_x, insertion_y),
        "target_class": target_class,
        "deletion_ks": deletion_x,
        "deletion_scores": deletion_y,
        "insertion_ks": insertion_x,
        "insertion_scores": insertion_y,
    }
