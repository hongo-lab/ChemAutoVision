"""Map pixel CAM heatmaps to RDKit atom indices."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from rdkit import Chem

from data.images import render_molecule_with_coords
from model_explanation.cam_generation import validate_heatmap
from settings import IMG_SIZE, IS_DOWNSCALE_BEFORE_DAUG


def render_mol_with_coords(
    smiles: str,
    img_size: tuple[int, int] = (IMG_SIZE[0], IMG_SIZE[1]),
) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    """Render with the same downscale-and-pad policy used by data generation."""
    width, height = img_size
    ratio = 1 / 1.5 if IS_DOWNSCALE_BEFORE_DAUG else 1.0
    mol_width = math.floor(width * ratio)
    mol_height = math.floor(height * ratio)
    pad_x = (width - mol_width) // 2
    pad_y = (height - mol_height) // 2
    molecule_image, local_coords = render_molecule_with_coords(
        smiles, (mol_width, mol_height)
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(molecule_image, (pad_x, pad_y))
    coords = {index: (x + pad_x, y + pad_y) for index, (x, y) in local_coords.items()}
    return np.asarray(canvas, dtype=np.uint8), coords


def aggregate_cam_at_atoms(
    heatmap: np.ndarray,
    atom_coords: dict[int, tuple[float, float]],
    *,
    radius: int = 8,
    method: str = "mean",
) -> dict[int, float]:
    """Aggregate CAM values in a circular neighbourhood around each atom."""
    if radius < 1:
        raise ValueError("radius must be at least 1")
    heatmap = validate_heatmap(heatmap)
    height, width = heatmap.shape
    ys, xs = np.mgrid[0:height, 0:width]
    scores: dict[int, float] = {}
    for atom_index, (center_x, center_y) in atom_coords.items():
        squared_distance = (xs - center_x) ** 2 + (ys - center_y) ** 2
        mask = squared_distance <= radius**2
        if not mask.any():
            raise ValueError(
                f"no pixels selected for atom {atom_index} at {(center_x, center_y)}"
            )
        if method == "mean":
            score = heatmap[mask].mean()
        elif method == "max":
            score = heatmap[mask].max()
        elif method == "gaussian":
            sigma = radius / 2.0
            weights = np.exp(-squared_distance / (2.0 * sigma**2))
            weights[~mask] = 0.0
            score = (heatmap * weights).sum() / weights.sum()
        else:
            raise ValueError(f"unknown aggregation method: {method!r}")
        scores[atom_index] = float(score)
    return scores


def attribute_bond_cam_to_atoms(
    mol: Chem.Mol,
    atom_scores: dict[int, float],
    heatmap: np.ndarray,
    atom_coords: dict[int, tuple[float, float]],
    *,
    bond_radius: int = 4,
) -> dict[int, float]:
    """Add half of each bond-midpoint CAM value to its endpoint atoms.

    Values are deliberately not re-normalized per molecule: such normalization
    would destroy absolute CAM magnitude before graph-model training.
    """
    heatmap = validate_heatmap(heatmap)
    height, width = heatmap.shape
    ys, xs = np.mgrid[0:height, 0:width]
    updated = dict(atom_scores)
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        left_x, left_y = atom_coords[left]
        right_x, right_y = atom_coords[right]
        midpoint_x = (left_x + right_x) / 2.0
        midpoint_y = (left_y + right_y) / 2.0
        mask = (xs - midpoint_x) ** 2 + (ys - midpoint_y) ** 2 <= bond_radius**2
        bond_score = float(heatmap[mask].mean())
        updated[left] += bond_score / 2.0
        updated[right] += bond_score / 2.0
    return updated


def cam_to_atom_scores(
    smiles: str,
    heatmap: np.ndarray,
    *,
    radius: int = 8,
    aggregation: str = "mean",
    include_bonds: bool = False,
    img_size: tuple[int, int] = (IMG_SIZE[0], IMG_SIZE[1]),
) -> dict[int, float]:
    """Return CAM scores keyed by RDKit atom index."""
    heatmap = validate_heatmap(heatmap, (img_size[1], img_size[0]))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    _, coords = render_mol_with_coords(smiles, img_size)
    if len(coords) != mol.GetNumAtoms():
        raise ValueError(
            f"atom-coordinate count mismatch: {len(coords)} != {mol.GetNumAtoms()}"
        )
    scores = aggregate_cam_at_atoms(heatmap, coords, radius=radius, method=aggregation)
    if include_bonds:
        scores = attribute_bond_cam_to_atoms(mol, scores, heatmap, coords)
    return scores
