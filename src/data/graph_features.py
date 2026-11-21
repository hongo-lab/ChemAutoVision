"""Build Chemprop per-atom feature arrays from CAM heatmaps."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from model_explanation.cam_atom_mapping import cam_to_atom_scores
from model_explanation.cam_generation import validate_heatmap


def load_heatmap(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"heatmap not found: {path}")
    try:
        heatmap = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid heatmap file: {path}") from exc
    return validate_heatmap(heatmap)


def validate_heatmap_paths(
    paths_by_method: dict[str, list[Path]],
    *,
    expected_count: int,
    expected_shape: tuple[int, int],
) -> None:
    """Fail before descriptor generation if any CAM file is absent or invalid."""
    errors: list[str] = []
    for method, paths in paths_by_method.items():
        if len(paths) != expected_count:
            errors.append(
                f"{method}: expected {expected_count} paths, received {len(paths)}"
            )
            continue
        for index, path in enumerate(paths):
            if not path.is_file():
                errors.append(f"{method}[{index}]: missing {path}")
                continue
            try:
                validate_heatmap(
                    np.load(path, allow_pickle=False), expected_shape=expected_shape
                )
            except Exception as exc:
                errors.append(f"{method}[{index}]: {path}: {exc}")
    if errors:
        preview = "\n  ".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise ValueError(f"CAM validation failed:\n  {preview}{suffix}")


def cam_to_atom_feature_vector(
    smiles: str,
    heatmap: np.ndarray,
    *,
    radius: int = 8,
    aggregation: str = "mean",
    include_bonds: bool = False,
    img_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    scores = cam_to_atom_scores(
        smiles,
        heatmap,
        radius=radius,
        aggregation=aggregation,
        include_bonds=include_bonds,
        img_size=img_size,
    )
    expected_indices = list(range(len(scores)))
    if sorted(scores) != expected_indices:
        raise ValueError("CAM atom scores do not contain contiguous RDKit atom indices")
    return np.asarray([scores[index] for index in expected_indices], dtype=np.float32)


def build_atom_descriptors(
    smiles_list: list[str],
    heatmap_paths_per_method: dict[str, list[Path]],
    *,
    radius: int = 8,
    aggregation: str = "mean",
    include_bonds: bool = False,
    img_size: tuple[int, int] = (256, 256),
) -> list[np.ndarray]:
    """Return one ``(n_atoms, n_methods)`` array per molecule.

    Missing or invalid heatmaps are always errors.  There is intentionally no
    zero-filling policy because zero is a valid CAM value.
    """
    if not heatmap_paths_per_method:
        raise ValueError("at least one CAM method is required")
    validate_heatmap_paths(
        heatmap_paths_per_method,
        expected_count=len(smiles_list),
        expected_shape=(img_size[1], img_size[0]),
    )
    methods = list(heatmap_paths_per_method)
    descriptors: list[np.ndarray] = []
    for molecule_index, smiles in enumerate(smiles_list):
        columns = [
            cam_to_atom_feature_vector(
                smiles,
                load_heatmap(heatmap_paths_per_method[method][molecule_index]),
                radius=radius,
                aggregation=aggregation,
                include_bonds=include_bonds,
                img_size=img_size,
            )
            for method in methods
        ]
        atom_counts = {len(column) for column in columns}
        if len(atom_counts) != 1:
            raise ValueError(
                f"CAM methods produced different atom counts for molecule "
                f"{molecule_index}: {sorted(atom_counts)}"
            )
        descriptors.append(np.stack(columns, axis=1))
    return descriptors


def heatmap_set_sha256(paths: Iterable[Path]) -> str:
    """Hash ordered heatmap names and bytes for descriptor provenance."""
    digest = hashlib.sha256(b"ChemAutoVision CAM heatmap set v1\0")
    for path in paths:
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def get_atom_feature_names() -> list[str]:
    try:
        from chemprop.features.featurization import PARAMS

        atom_features = PARAMS.ATOM_FEATURES
    except (ImportError, AttributeError):
        try:
            from chemprop.features.featurization import Featurization_parameters

            atom_features = Featurization_parameters().ATOM_FEATURES
        except (ImportError, AttributeError):
            atom_features = None
    if atom_features is not None:
        names: list[str] = []
        for feature_name, choices in atom_features.items():
            names.extend(f"{feature_name}={choice}" for choice in choices)
            names.append(f"{feature_name}=unknown")
        return [*names, "aromaticity", "mass"]
    try:
        from chemprop.features.featurization import get_atom_fdim

        return [f"atom_feat_{index}" for index in range(get_atom_fdim())]
    except (ImportError, AttributeError):
        return []


def get_full_feature_names(cam_methods: list[str]) -> list[str]:
    return [
        *get_atom_feature_names(),
        *(f"cam_score_{method}" for method in cam_methods),
    ]
