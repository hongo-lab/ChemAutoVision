"""Shared configuration helpers for dataset split artifacts."""

from __future__ import annotations

from pathlib import Path

BALANCED_SCAFFOLD_SPLIT_TYPE = "balanced_scaffold"
LABEL_SCHEME_ARTIFACT_PREFIXES = {
    "tertile_extremes": "cls_t_",
    "quartile_extremes": "cls_q_",
}


def label_scheme_artifact_prefix(label_scheme: str | None) -> str:
    """Return the filename prefix for an optional derived label scheme."""
    if label_scheme is None:
        return ""
    try:
        return LABEL_SCHEME_ARTIFACT_PREFIXES[label_scheme]
    except KeyError as exc:
        raise ValueError(f"unsupported label scheme: {label_scheme}") from exc


def split_method_name(split_type: str) -> str:
    """Return the unambiguous method name used in experiment metadata."""
    if split_type == "random":
        return "random"
    if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE:
        return "balanced_scaffold"
    raise ValueError(f"unsupported split type: {split_type}")


def make_split_prefix(split_type: str, split_seed: int | None) -> str:
    """Return the shared prefix for data files and image directories."""
    if split_type == "random":
        return ""
    if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE:
        if split_seed is None:
            raise ValueError(
                "--split_seed is required when --split_type balanced_scaffold"
            )
        return f"balanced_scaffold_seed{split_seed}_"
    raise ValueError(f"unsupported split type: {split_type}")


def make_split_csv_paths(
    data_dir: str | Path,
    task_name: str,
    split_type: str,
    split_seed: int | None,
    label_scheme: str | None = None,
) -> dict[str, Path]:
    """Return the train/val/test CSV paths for a generated split."""
    prefix = make_split_prefix(split_type, split_seed)
    label_prefix = label_scheme_artifact_prefix(label_scheme)
    directory = Path(data_dir)
    return {
        subset: directory / f"{subset}_{prefix}{label_prefix}{task_name}_img.csv"
        for subset in ("train", "val", "test")
    }


def make_dataset_pickle_path(
    data_dir: str | Path,
    task_name: str,
    split_type: str,
    split_seed: int | None,
    label_scheme: str | None = None,
) -> Path:
    """Return the pickle path for a generated dataset artifact."""
    prefix = make_split_prefix(split_type, split_seed)
    label_prefix = label_scheme_artifact_prefix(label_scheme)
    return Path(data_dir) / f"{prefix}{label_prefix}{task_name}_img.pkl"
