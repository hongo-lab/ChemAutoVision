import json
import pickle

import pandas as pd
import pytest

from utils.atom_descriptor_metadata import (
    atom_descriptor_metadata_path,
    validate_atom_descriptor_metadata,
    write_atom_descriptor_metadata,
)


def _write_csv(path, smiles):
    pd.DataFrame({"smiles": smiles, "target": range(len(smiles))}).to_csv(
        path, index=False
    )


def _write_descriptors(path, count):
    with path.open("wb") as stream:
        pickle.dump([[[float(index)]] for index in range(count)], stream)


def test_metadata_round_trip_proves_scaffold_split(tmp_path):
    csv_path = tmp_path / "train_balanced_scaffold_seed42_BBBP_img.csv"
    descriptor_path = tmp_path / "BBBP_atom_desc_train.pkl"
    _write_csv(csv_path, ["CCO", "c1ccccc1"])
    _write_descriptors(descriptor_path, 2)

    metadata_path = write_atom_descriptor_metadata(
        descriptor_path,
        csv_path,
        split_type="balanced_scaffold",
        split_seed=42,
        subset="train",
        extra={"n_cam_features": 1},
    )

    assert metadata_path == atom_descriptor_metadata_path(descriptor_path)
    metadata = validate_atom_descriptor_metadata(
        descriptor_path,
        csv_path,
        split_type="balanced_scaffold",
        split_seed=42,
        subset="train",
    )
    assert metadata["n_cam_features"] == 1
    assert metadata["num_molecules"] == 2


def test_wrong_split_seed_is_rejected(tmp_path):
    csv_path = tmp_path / "val_balanced_scaffold_seed1_BBBP_img.csv"
    descriptor_path = tmp_path / "BBBP_atom_desc_val.pkl"
    _write_csv(csv_path, ["CCO"])
    _write_descriptors(descriptor_path, 1)
    write_atom_descriptor_metadata(
        descriptor_path,
        csv_path,
        split_type="balanced_scaffold",
        split_seed=1,
        subset="val",
    )

    with pytest.raises(ValueError, match="split_seed"):
        validate_atom_descriptor_metadata(
            descriptor_path,
            csv_path,
            split_type="balanced_scaffold",
            split_seed=42,
            subset="val",
        )


def test_reordered_smiles_are_rejected(tmp_path):
    csv_path = tmp_path / "test_BBBP_img.csv"
    descriptor_path = tmp_path / "BBBP_atom_desc_test.pkl"
    _write_csv(csv_path, ["CCO", "CCN"])
    _write_descriptors(descriptor_path, 2)
    write_atom_descriptor_metadata(
        descriptor_path,
        csv_path,
        split_type="random",
        split_seed=None,
        subset="test",
    )
    _write_csv(csv_path, ["CCN", "CCO"])

    with pytest.raises(ValueError, match="ordered_smiles_sha256"):
        validate_atom_descriptor_metadata(
            descriptor_path,
            csv_path,
            split_type="random",
            split_seed=None,
            subset="test",
        )


def test_replaced_descriptor_pickle_is_rejected(tmp_path):
    csv_path = tmp_path / "train_BBBP_img.csv"
    descriptor_path = tmp_path / "BBBP_atom_desc_train.pkl"
    _write_csv(csv_path, ["CCO", "CCN"])
    _write_descriptors(descriptor_path, 2)
    write_atom_descriptor_metadata(
        descriptor_path,
        csv_path,
        split_type="random",
        split_seed=None,
        subset="train",
    )
    _write_descriptors(descriptor_path, 3)

    with pytest.raises(ValueError, match="descriptor_sha256"):
        validate_atom_descriptor_metadata(
            descriptor_path,
            csv_path,
            split_type="random",
            split_seed=None,
            subset="train",
        )


def test_missing_or_legacy_metadata_is_rejected(tmp_path):
    csv_path = tmp_path / "train_BBBP_img.csv"
    descriptor_path = tmp_path / "BBBP_atom_desc_train.pkl"
    _write_csv(csv_path, ["CCO"])
    _write_descriptors(descriptor_path, 1)

    with pytest.raises(ValueError, match="metadata not found"):
        validate_atom_descriptor_metadata(
            descriptor_path,
            csv_path,
            split_type="random",
            split_seed=None,
            subset="train",
        )

    atom_descriptor_metadata_path(descriptor_path).write_text(
        json.dumps({"n_cam_features": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing fields"):
        validate_atom_descriptor_metadata(
            descriptor_path,
            csv_path,
            split_type="random",
            split_seed=None,
            subset="train",
        )
