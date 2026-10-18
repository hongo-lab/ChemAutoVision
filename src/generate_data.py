import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from data.dataset_registry import (
    QM9_TARGETS,
    dataset_name_for_task,
    target_columns_for_generation,
)
from data.qm9 import load_qm9_dataframe
from settings import IMG_SIZE, IS_DOWNSCALE_BEFORE_DAUG
from utils.split import BALANCED_SCAFFOLD_SPLIT_TYPE, make_split_prefix

RAW_HERG_CSV_PATH = "../data/raw/hERG_inhibition/hERG.csv"
RAW_CYP3A4_CSV_PATH = "../data/raw/CYP3A4_inhibition/CYP3A4.csv"
RAW_PGP_CSV_PATH = "../data/raw/P-gp_substrate/P-gp.csv"
CLASSIFICATION_TASKS = {"BBBP", "hERG", "P-gp", "CYP3A4"}


def _generate_image(
    _df_row: pd.Series,
    task_name: str,
    split_prefix: str = "",
    overwrite: bool = False,
) -> pd.Series:
    from data.images import padding_image, smi_to_img

    if task_name in ["london", "lz", "dm"]:
        img_dir = "qm"
    else:
        img_dir = task_name
    # 構造ベースsplitではidx→分子の対応がrandom版と変わるため、
    # 画像も別ディレクトリに出して上書きを防ぐ。
    img_dir = f"{split_prefix}{img_dir}"
    downscale_ratio = 1 / 1.5 if IS_DOWNSCALE_BEFORE_DAUG else 1
    original_img_size = (
        math.floor(IMG_SIZE[0] * downscale_ratio),
        math.floor(IMG_SIZE[1] * downscale_ratio),
    )

    img_output_dir = f"../data/images/{img_dir}"
    os.makedirs(img_output_dir, exist_ok=True)
    if task_name == "QM9":
        image_key = hashlib.sha256(_df_row.smiles.encode("utf-8")).hexdigest()
    else:
        image_key = str(_df_row.idx)
    img_file_path = f"{img_output_dir}/{image_key}.png"

    if overwrite or not os.path.isfile(img_file_path):
        original_img = smi_to_img(
            _df_row.smiles,
            original_img_size,
        )
        padded_img = padding_image(original_img, (IMG_SIZE[0], IMG_SIZE[1]))
        temporary_path = f"{img_file_path}.tmp"
        padded_img.save(temporary_path, format="PNG")
        os.replace(temporary_path, img_file_path)

    _df_row["img_path"] = img_file_path

    return _df_row


def _exc_dup_mol(
    df: pd.DataFrame, target_columns: str | Sequence[str]
) -> pd.DataFrame:
    columns = (
        (target_columns,) if isinstance(target_columns, str) else tuple(target_columns)
    )
    if "smiles" not in df.columns or any(column not in df.columns for column in columns):
        raise ValueError("Invalid DataFrame columns")

    dup_smiles = set(df[df.duplicated(subset=["smiles"], keep=False)]["smiles"])
    if not dup_smiles:
        return df

    smiles_to_remove = set()
    for smiles in dup_smiles:
        subset = df[df["smiles"] == smiles]
        if len(subset.loc[:, columns].drop_duplicates()) > 1:
            smiles_to_remove.add(smiles)

    if smiles_to_remove:
        # Remove all rows with conflicting SMILES
        result_df = df[~df["smiles"].isin(smiles_to_remove)]
    else:
        result_df = df

    # 目的変数の値が同一なのであれば、1件だけ残す
    result_df = result_df.drop_duplicates(subset=["smiles"], keep="first")
    return result_df


def _find_largest_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        lfc = MolStandardize.fragment.LargestFragmentChooser()
        mol = lfc.choose(mol)
        smi = Chem.MolToSmiles(mol)
    else:
        print(f"invalid smiles: {smiles}")
        return None
    return smi


def _preprocess_smiles(smiles: str) -> str:
    smiles = _smi_to_canonical_smi(smiles)
    if smiles is None:
        return None
    smiles = _find_largest_smiles(smiles)
    if smiles is None:
        return None
    smiles = _smi_to_canonical_smi(smiles)
    return smiles


def _load_data(task_name: str) -> pd.DataFrame:
    if task_name == "QM9":
        datasets = load_qm9_dataframe()
    elif task_name == "FreeSolv":
        _, datasets, _ = dc.molnet.load_freesolv(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    elif task_name == "ESOL":
        _, datasets, _ = dc.molnet.load_delaney(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    elif task_name == "Lipo":
        _, datasets, _ = dc.molnet.load_lipo(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    elif task_name == "CYP3A4":
        datasets = pd.read_csv(RAW_CYP3A4_CSV_PATH)
    elif task_name == "P-gp":
        datasets = pd.read_csv(RAW_PGP_CSV_PATH)
    elif task_name == "hERG":
        datasets = pd.read_csv(RAW_HERG_CSV_PATH)
    elif task_name == "BBBP":
        _, datasets, _ = dc.molnet.load_bbbp(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    else:
        raise ValueError("Invalid task name is input")

    if not isinstance(datasets, pd.DataFrame):
        datasets = pd.DataFrame(
            {"smiles": datasets[0].ids, task_name: datasets[0].y.flatten()}
        )

    if "source_index" not in datasets.columns:
        datasets["index"] = datasets.index
    return datasets


def _split_data(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    _df_train, df_test = train_test_split(df, test_size=ratio, random_state=99)
    df_train, df_val = train_test_split(_df_train, test_size=ratio, random_state=99)

    df_train["group"] = "train"
    df_val["group"] = "val"
    df_test["group"] = "test"
    df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
    return df


def _split_data_balanced_scaffold(
    df: pd.DataFrame,
    seed: int,
    frac_train: float = 0.64,
    frac_valid: float = 0.16,
    frac_test: float = 0.2,
) -> pd.DataFrame:
    """Chemprop-style balanced Bemis-Murcko scaffold split.

    Scaffold groups are kept intact. Large and small groups are shuffled
    separately with a local RNG, then greedily assigned to the requested folds.
    reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py#L56
    """
    if not math.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("split fractions must sum to 1")
    if min(frac_train, frac_valid, frac_test) <= 0:
        raise ValueError("split fractions must all be positive")

    scaffold_to_indices: dict[str, set[int]] = {}
    for index, smiles in enumerate(df["smiles"]):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=False
        )
        scaffold_to_indices.setdefault(scaffold, set()).add(index)

    train_size = frac_train * len(df)
    valid_size = frac_valid * len(df)
    test_size = frac_test * len(df)
    big_sets: list[set[int]] = []
    small_sets: list[set[int]] = []
    for index_set in scaffold_to_indices.values():
        if len(index_set) > valid_size / 2 or len(index_set) > test_size / 2:
            big_sets.append(index_set)
        else:
            small_sets.append(index_set)

    rng = random.Random(seed)
    rng.shuffle(big_sets)
    rng.shuffle(small_sets)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for index_set in big_sets + small_sets:
        if len(train_idx) + len(index_set) <= train_size:
            train_idx.extend(index_set)
        elif len(val_idx) + len(index_set) <= valid_size:
            val_idx.extend(index_set)
        else:
            test_idx.extend(index_set)

    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if not indices:
            raise ValueError(
                f"balanced scaffold split produced an empty {name} fold (seed={seed})"
            )

    parts = []
    for idx, name in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
        part = df.iloc[idx].copy()
        part["group"] = name
        parts.append(part)
    return pd.concat(parts, axis=0, ignore_index=True)


def _dispatch_split(
    df: pd.DataFrame, split_type: str, split_seed: int | None = None
) -> pd.DataFrame:
    if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE:
        if split_seed is None:
            raise ValueError(
                "--split_seed is required when --split_type balanced_scaffold"
            )
        return _split_data_balanced_scaffold(df, seed=split_seed)
    return _split_data(df, 0.2)


def _validate_split(df: pd.DataFrame, task_name: str) -> None:
    """Fail early when a generated fold cannot support model evaluation."""
    counts = df["group"].value_counts()
    if set(counts.index) != {"train", "val", "test"}:
        raise ValueError(f"split produced missing folds: {counts.to_dict()}")

    print(f"Split sizes: {counts.to_dict()}")
    if task_name in CLASSIFICATION_TASKS:
        class_counts = pd.crosstab(df["group"], df[task_name])
        print(f"Class distribution:\n{class_counts}")
        for group in ("train", "val", "test"):
            labels = set(df.loc[df["group"] == group, task_name].dropna().unique())
            if not {0, 1}.issubset(labels):
                raise ValueError(
                    f"{task_name} {group} fold must contain classes 0 and 1; "
                    f"found {sorted(labels)}"
                )


def _smi_to_canonical_smi(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def _save_split_csvs(
    df: pd.DataFrame,
    task_name: str,
    target_columns: Sequence[str],
    prefix: str,
    split_prefix: str = "",
) -> None:
    identifier_columns = ["idx"]
    if "source_index" in df.columns:
        identifier_columns.append("source_index")
    cols = [*identifier_columns, "smiles", *target_columns, "group", "img_path"]
    for split in ("train", "val", "test"):
        name = (
            f"{split}_{split_prefix}{prefix}_{task_name}_img.csv"
            if prefix
            else f"{split}_{split_prefix}{task_name}_img.csv"
        )
        out = f"../data/{name}"
        df[df["group"] == split][cols].to_csv(out, index=False)
        print(f"Saved: {out}")


def _ordered_data_sha256(df: pd.DataFrame, target_columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in df.loc[:, ["smiles", *target_columns]].itertuples(
        index=False, name=None
    ):
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _write_qm9_metadata(
    df: pd.DataFrame,
    output_path: str,
    split_type: str,
    split_seed: int | None,
    loader_count: int,
) -> None:
    metadata = {
        "schema_version": 1,
        "dataset": "QM9",
        "loader": "deepchem.molnet.load_qm9",
        "deepchem_version": importlib.metadata.version("deepchem"),
        "rdkit_version": importlib.metadata.version("rdkit"),
        "pandas_version": pd.__version__,
        "target_units": {"homo": "Hartree", "lumo": "Hartree", "mu": "Debye"},
        "loader_row_count": loader_count,
        "processed_row_count": len(df),
        "fold_counts": {
            str(key): int(value) for key, value in df["group"].value_counts().items()
        },
        "split_type": split_type,
        "split_seed": split_seed if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE else None,
        "image_size": list(IMG_SIZE),
        "ordered_data_sha256": _ordered_data_sha256(df, QM9_TARGETS),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = Path(output_path).with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument("--task_name", type=str, required=True, help="the task name")
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=["random", BALANCED_SCAFFOLD_SPLIT_TYPE],
        help="data split strategy",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="seed for balanced scaffold assignment (required for balanced_scaffold)",
    )
    parser.add_argument(
        "--overwrite_images",
        action="store_true",
        help="regenerate molecular images that already exist",
    )
    args = parser.parse_args()
    task_name = args.task_name
    dataset_name = dataset_name_for_task(task_name)
    split_type = args.split_type
    split_seed = args.split_seed
    # balanced scaffold splitの生成物は方式とseedをファイル名に含める。
    # random（デフォルト）では空文字なので従来の挙動を完全維持する。
    split_prefix = make_split_prefix(split_type, split_seed)

    output_path = f"../data/{split_prefix}{dataset_name}_img.pkl"

    _df = _load_data(dataset_name)
    loader_count = len(_df)
    target_columns = target_columns_for_generation(task_name)

    # desalts
    _df["smiles"] = _df["smiles"].apply(_preprocess_smiles)
    _df = _df[_df["smiles"].notna()]
    if dataset_name == "QM9":
        _df = _df[np.isfinite(_df[list(target_columns)]).all(axis=1)]
    # remove duplicated mol
    _df = _exc_dup_mol(_df, target_columns)
    # data split
    _df = _dispatch_split(_df, split_type, split_seed)
    _validate_split(_df, dataset_name)
    _df.index.name = 'idx'
    _df = _df.reset_index()

    image_paths = []
    total_images = len(_df)
    for image_number, (_, row) in enumerate(_df.iterrows(), start=1):
        generated = _generate_image(
            row,
            dataset_name,
            split_prefix,
            overwrite=args.overwrite_images,
        )
        image_paths.append(generated["img_path"])
        if image_number % 1000 == 0 or image_number == total_images:
            print(f"Images: {image_number}/{total_images}")
    df = _df.copy()
    df["img_path"] = image_paths

    df.to_pickle(output_path)
    _save_split_csvs(df, dataset_name, target_columns, "", split_prefix)
    if dataset_name == "QM9":
        _write_qm9_metadata(
            df,
            output_path,
            split_type,
            split_seed,
            loader_count,
        )

    print("End Creating Data")


if __name__ == "__main__":
    main()
