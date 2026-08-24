"""Provenance metadata for Chemprop per-atom descriptor pickle files.

The pickle format consumed by Chemprop contains arrays only and therefore cannot
identify the SMILES rows used to create it.  A sidecar JSON binds a descriptor
pickle to the ordered SMILES, split settings, and subset that produced it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from utils.split import BALANCED_SCAFFOLD_SPLIT_TYPE, split_method_name


ATOM_DESCRIPTOR_METADATA_SCHEMA_VERSION = 1
VALID_SUBSETS = {"train", "val", "test"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_smiles_sha256(smiles: Iterable[str]) -> str:
    """Hash SMILES without delimiter ambiguity while preserving row order."""
    digest = hashlib.sha256()
    digest.update(b"ChemAutoVision ordered SMILES v1\0")
    for value in smiles:
        if not isinstance(value, str):
            raise ValueError("all SMILES values must be strings")
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def atom_descriptor_metadata_path(descriptor_path: str | Path) -> Path:
    """Return ``*_train.meta.json`` for ``*_train.pkl``."""
    return Path(descriptor_path).with_suffix(".meta.json")


def _normalized_split_seed(split_type: str, split_seed: int | None) -> int | None:
    split_method_name(split_type)  # Validate the method.
    if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE:
        if split_seed is None:
            raise ValueError(
                "--split_seed is required when --split_type balanced_scaffold"
            )
        return split_seed
    return None


def write_atom_descriptor_metadata(
    descriptor_path: str | Path,
    source_csv_path: str | Path,
    *,
    split_type: str,
    split_seed: int | None,
    subset: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a provenance sidecar immediately after generating a descriptor pkl.

    Feature-generation code must call this only after it has written the final
    pickle.  The CSV's ordered SMILES and the pickle bytes are both fingerprinted.
    """
    descriptor = Path(descriptor_path)
    source_csv = Path(source_csv_path)
    if subset not in VALID_SUBSETS:
        raise ValueError(f"unsupported subset: {subset}")
    if not descriptor.is_file():
        raise FileNotFoundError(f"atom descriptor pkl not found: {descriptor}")
    if not source_csv.is_file():
        raise FileNotFoundError(f"source CSV not found: {source_csv}")

    frame = pd.read_csv(source_csv, usecols=["smiles"])
    smiles = frame["smiles"].tolist()
    metadata: dict[str, Any] = {
        "schema_version": ATOM_DESCRIPTOR_METADATA_SCHEMA_VERSION,
        "split_type": split_type,
        "split_seed": _normalized_split_seed(split_type, split_seed),
        "subset": subset,
        "source_csv": source_csv.name,
        "num_molecules": len(smiles),
        "ordered_smiles_sha256": ordered_smiles_sha256(smiles),
        "descriptor_sha256": _sha256_file(descriptor),
    }
    if extra:
        reserved = set(metadata).intersection(extra)
        if reserved:
            raise ValueError(
                "extra metadata cannot replace reserved fields: "
                + ", ".join(sorted(reserved))
            )
        metadata.update(extra)

    output_path = atom_descriptor_metadata_path(descriptor)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return output_path


def _descriptor_row_count(descriptor_path: Path) -> int:
    with descriptor_path.open("rb") as stream:
        descriptors = pickle.load(stream)
    try:
        return len(descriptors)
    except TypeError as exc:
        raise ValueError(
            f"atom descriptor pkl has no molecule dimension: {descriptor_path}"
        ) from exc


def validate_atom_descriptor_metadata(
    descriptor_path: str | Path,
    source_csv_path: str | Path,
    *,
    split_type: str,
    split_seed: int | None,
    subset: str,
) -> dict[str, Any]:
    """Fail if a descriptor pkl is not proven to match the selected CSV split."""
    descriptor = Path(descriptor_path)
    source_csv = Path(source_csv_path)
    metadata_path = atom_descriptor_metadata_path(descriptor)
    if subset not in VALID_SUBSETS:
        raise ValueError(f"unsupported subset: {subset}")
    if not descriptor.is_file():
        raise FileNotFoundError(f"atom descriptor pkl not found: {descriptor}")
    if not source_csv.is_file():
        raise FileNotFoundError(f"split CSV not found: {source_csv}")
    if not metadata_path.is_file():
        raise ValueError(
            f"atom descriptor metadata not found: {metadata_path}. "
            "The descriptor pkl cannot be safely matched to the selected split; "
            "regenerate it with provenance metadata."
        )

    try:
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid atom descriptor metadata: {metadata_path}") from exc

    required = {
        "schema_version",
        "split_type",
        "split_seed",
        "subset",
        "source_csv",
        "num_molecules",
        "ordered_smiles_sha256",
        "descriptor_sha256",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            f"atom descriptor metadata is missing fields {sorted(missing)}: "
            f"{metadata_path}"
        )

    frame = pd.read_csv(source_csv, usecols=["smiles"])
    smiles = frame["smiles"].tolist()
    expected = {
        "schema_version": ATOM_DESCRIPTOR_METADATA_SCHEMA_VERSION,
        "split_type": split_type,
        "split_seed": _normalized_split_seed(split_type, split_seed),
        "subset": subset,
        "source_csv": source_csv.name,
        "num_molecules": len(smiles),
        "ordered_smiles_sha256": ordered_smiles_sha256(smiles),
        "descriptor_sha256": _sha256_file(descriptor),
    }
    mismatches = {
        key: (metadata.get(key), expected_value)
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    descriptor_rows = _descriptor_row_count(descriptor)
    if descriptor_rows != len(smiles):
        mismatches["descriptor_row_count"] = (descriptor_rows, len(smiles))
    if mismatches:
        details = "; ".join(
            f"{key}: metadata/pkl={actual!r}, expected={wanted!r}"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(
            f"atom descriptor provenance mismatch for {subset}: {details}"
        )
    return metadata
