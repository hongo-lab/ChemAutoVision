"""Shared filesystem helpers for CAM command-line workflows."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from settings import IMG_SIZE


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_image_path(raw_value: object, source_csv: Path) -> Path:
    raw_path = Path(str(raw_value))
    attempts = (
        [raw_path]
        if raw_path.is_absolute()
        else [
            Path.cwd() / raw_path,
            REPO_ROOT / "src" / raw_path,
            REPO_ROOT / raw_path,
            source_csv.parent / raw_path,
        ]
    )
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    raise FileNotFoundError(
        f"image not found: {raw_value!r}; tried {[str(path) for path in attempts]}"
    )


def load_image_uint8(path: Path) -> np.ndarray:
    raw = tf.io.read_file(str(path))
    tensor = tf.image.decode_png(raw, channels=3)
    tensor = tf.image.resize(tensor, [IMG_SIZE[0], IMG_SIZE[1]])
    array = np.asarray(tensor.numpy())
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def preprocess_image(path: Path) -> np.ndarray:
    return load_image_uint8(path).astype(np.float32)[None] / 255.0


def heatmap_path(
    image_path: Path,
    source_run_id: str,
    method: str,
    target_policy: str,
) -> Path:
    return (
        image_path.parent
        / "cams"
        / source_run_id
        / method
        / target_policy
        / image_path.with_suffix(".npy").name
    )


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)
