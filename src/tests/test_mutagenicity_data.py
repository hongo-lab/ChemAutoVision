from pathlib import Path

import pandas as pd
import pytest

from generate_data import _exc_dup_mol, _load_mutagenicity_data, _preprocess_smiles


def _write_hansen_csv(path: Path, activities=(0, 1)) -> None:
    pd.DataFrame(
        {
            "CAS_NO": ["1-00-0", "2-00-0"],
            "Source": ["VITIC", "CCRIS"],
            "Activity": list(activities),
            "Steroid": [-1, -1],
            "WDI": [None, None],
            "Canonical_Smiles": ["CCO.[Na+]", "c1ccccc1"],
            "REFERENCE": ["first", "second"],
        }
    ).to_csv(path, index=False)


def test_hansen_schema_is_adapted_for_shared_pipeline(tmp_path: Path):
    path = tmp_path / "Mutagenicity.csv"
    _write_hansen_csv(path)

    frame = _load_mutagenicity_data(path)

    assert frame["smiles"].tolist() == ["CCO.[Na+]", "c1ccccc1"]
    assert frame["Mutagenicity"].tolist() == [0, 1]
    assert frame["source_index"].tolist() == [0, 1]
    assert {"CAS_NO", "Source", "REFERENCE"}.issubset(frame.columns)


def test_mutagenicity_uses_existing_smiles_and_duplicate_processing(tmp_path: Path):
    path = tmp_path / "Mutagenicity.csv"
    pd.DataFrame(
        {
            "Activity": [0, 0, 1, 0],
            "Canonical_Smiles": ["CCO.[Na+]", "CCO", "CCN", "CCN"],
        }
    ).to_csv(path, index=False)

    frame = _load_mutagenicity_data(path)
    frame["smiles"] = frame["smiles"].apply(_preprocess_smiles)
    result = _exc_dup_mol(frame, "Mutagenicity")

    assert result[["smiles", "Mutagenicity"]].to_dict("records") == [
        {"smiles": "CCO", "Mutagenicity": 0}
    ]


def test_mutagenicity_rejects_non_binary_labels(tmp_path: Path):
    path = tmp_path / "Mutagenicity.csv"
    _write_hansen_csv(path, activities=(0, 2))

    with pytest.raises(ValueError, match="must be 0 or 1"):
        _load_mutagenicity_data(path)
