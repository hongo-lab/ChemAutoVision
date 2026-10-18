from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.dataset_registry import (
    dataset_name_for_task,
    target_column_for_task,
    target_columns_for_generation,
)
from data.qm9 import load_qm9_dataframe
from generate_data import _exc_dup_mol, _preprocess_smiles


def test_qm9_loader_selects_named_targets_independent_of_task_order():
    tasks = ["mu", "gap", "lumo", "homo"]
    dataset = SimpleNamespace(
        X=np.asarray(["CCO", "CCN"], dtype=object),
        y=np.asarray(
            [
                [1.1, 9.9, 0.2, -0.3],
                [1.2, 8.8, 0.4, -0.5],
            ]
        ),
        w=np.ones((2, 4)),
    )

    def loader(**kwargs):
        assert kwargs["splitter"] is None
        assert kwargs["transformers"] == []
        return tasks, (dataset,), []

    frame = load_qm9_dataframe(loader)

    assert frame.columns.tolist() == ["source_index", "smiles", "homo", "lumo", "mu"]
    assert frame["source_index"].tolist() == [0, 1]
    assert frame["homo"].tolist() == [-0.3, -0.5]
    assert frame["lumo"].tolist() == [0.2, 0.4]
    assert frame["mu"].tolist() == [1.1, 1.2]


def test_qm9_loader_rejects_missing_required_target():
    dataset = SimpleNamespace(
        X=np.asarray(["CCO"], dtype=object),
        y=np.asarray([[1.0, 2.0]]),
        w=np.ones((1, 2)),
    )

    with pytest.raises(ValueError, match="missing tasks: mu"):
        load_qm9_dataframe(lambda **_: (["homo", "lumo"], (dataset,), []))


def test_qm9_deduplication_compares_all_targets():
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCO", "CCN", "CCN"],
            "homo": [-0.3, -0.3, -0.4, -0.4],
            "lumo": [0.2, 0.2, 0.1, 0.9],
            "mu": [1.0, 1.0, 2.0, 2.0],
        }
    )

    result = _exc_dup_mol(frame, ("homo", "lumo", "mu"))

    assert result["smiles"].tolist() == ["CCO"]


def test_qm9_task_name_resolves_shared_dataset_and_target_column():
    assert target_columns_for_generation("QM9_lumo") == ("homo", "lumo", "mu")
    assert dataset_name_for_task("QM9_lumo") == "QM9"
    assert target_column_for_task("QM9_lumo") == "lumo"
    with pytest.raises(ValueError, match="task-specific name"):
        target_column_for_task("QM9")


def test_smiles_preprocessing_canonicalizes_and_keeps_largest_fragment():
    assert _preprocess_smiles("[Na+].CCO") == "CCO"
