"""Load the shared QM9 SMILES/target table through DeepChem."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem

from data.dataset_registry import QM9_TARGETS


def _as_smiles(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Chem.Mol):
        return Chem.MolToSmiles(value)
    return None


def load_qm9_dataframe(
    loader: Callable[..., tuple[list[str], tuple[Any, ...], list[Any]]] | None = None,
) -> pd.DataFrame:
    """Return one frame containing SMILES plus HOMO, LUMO, and dipole targets.

    DeepChem performs no split and no target transformation here.  Splitting and
    molecular standardization remain owned by ``generate_data.py``.
    """
    load = loader or dc.molnet.load_qm9
    tasks, datasets, transformers = load(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter=None,
        transformers=[],
    )

    if transformers:
        raise ValueError("QM9 loader unexpectedly returned target transformers")
    if len(datasets) != 1:
        raise ValueError(
            f"QM9 loader must return one unsplit dataset, got {len(datasets)}"
        )
    if len(tasks) != len(set(tasks)):
        raise ValueError("QM9 loader returned duplicate task names")
    missing = [target for target in QM9_TARGETS if target not in tasks]
    if missing:
        raise ValueError(f"QM9 loader is missing tasks: {', '.join(missing)}")

    dataset = datasets[0]
    features = np.asarray(dataset.X, dtype=object).reshape(-1)
    targets = np.asarray(dataset.y)
    weights = np.asarray(dataset.w) if getattr(dataset, "w", None) is not None else None
    if targets.ndim != 2 or targets.shape[0] != len(features):
        raise ValueError(
            "QM9 feature/target shape mismatch: "
            f"X={features.shape}, y={targets.shape}"
        )
    if targets.shape[1] != len(tasks):
        raise ValueError(
            f"QM9 task/target shape mismatch: tasks={len(tasks)}, y={targets.shape}"
        )
    if weights is not None and weights.shape[0] != len(features):
        raise ValueError(
            "QM9 feature/weight shape mismatch: "
            f"X={features.shape}, w={weights.shape}"
        )

    indices = [tasks.index(target) for target in QM9_TARGETS]
    frame = pd.DataFrame(
        {
            "source_index": np.arange(len(features), dtype=np.int64),
            "smiles": [_as_smiles(value) for value in features],
            **{
                target: targets[:, index]
                for target, index in zip(QM9_TARGETS, indices)
            },
        }
    )
    for target in QM9_TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")

    valid = frame["smiles"].notna()
    valid &= np.isfinite(frame[list(QM9_TARGETS)]).all(axis=1)
    if weights is not None:
        if weights.ndim == 1:
            target_weights = np.repeat(weights[:, None], len(indices), axis=1)
        elif weights.ndim == 2 and weights.shape[1] == len(tasks):
            target_weights = weights[:, indices]
        else:
            raise ValueError(
                f"QM9 task/weight shape mismatch: tasks={len(tasks)}, w={weights.shape}"
            )
        valid &= np.isfinite(target_weights).all(axis=1)
        valid &= (target_weights > 0).all(axis=1)

    result = frame.loc[valid].copy()
    if result.empty:
        raise ValueError("QM9 loader returned no rows with all required targets")
    return result
