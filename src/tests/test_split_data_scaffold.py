"""Chemprop-style balanced Bemis-Murcko scaffold split tests.

fixture には異なる Murcko 骨格を複数含む実 SMILES（ベンゼン/ピリジン/ナフタレン/
シクロヘキサン/キノリン/ビフェニル系など）を用意し、同一骨格の分子と非環分子
（scaffold が空）を意図的に混在させる。
"""

from pathlib import Path
from random import Random

import pandas as pd
import pytest
from rdkit.Chem.Scaffolds import MurckoScaffold

from generate_data import (
    _dispatch_split,
    _split_data,
    _split_data_balanced_scaffold,
    _validate_split,
)
from utils.split import make_split_csv_paths, make_split_prefix, split_method_name

# 同一骨格を複数持つグループと、非環分子（空 scaffold）を混在させた 24 件。
SMILES = [
    # benzene 骨格 (5)
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "OC(=O)c1ccccc1",
    # pyridine 骨格 (3)
    "c1ccncc1", "Cc1ccncc1", "Oc1ccncc1",
    # naphthalene 骨格 (3)
    "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "Oc1ccc2ccccc2c1",
    # cyclohexane 骨格 (2)
    "C1CCCCC1", "CC1CCCCC1",
    # quinoline 骨格 (2)
    "c1ccc2ncccc2c1", "Cc1ccc2ncccc2c1",
    # biphenyl 骨格 (2)
    "c1ccc(-c2ccccc2)cc1", "Cc1ccc(-c2ccccc2)cc1",
    # 単発の環骨格 (2)
    "c1ccoc1", "c1ccsc1",
    # 非環分子（scaffold が空）(5)
    "CCO", "CCCC", "CC(=O)O", "CCN", "CCCCO",
]
SPLITS = {"train", "val", "test"}
CHEMPROP_PARITY_DATASETS = (
    "BBBP",
    "hERG",
    "P-gp",
    "CYP3A4",
    "FreeSolv",
    "ESOL",
    "Lipo",
)


def _murcko(smiles: str) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=False)


def _chemprop_64335d5_reference_indices(
    smiles: list[str], seed: int, sizes: tuple[float, float, float] = (0.64, 0.16, 0.2)
) -> dict[str, set[int]]:
    """Independent transcription of Chemprop scaffold_split(..., balanced=True)."""
    scaffold_to_indices: dict[str, set[int]] = {}
    for index, value in enumerate(smiles):
        scaffold_to_indices.setdefault(_murcko(value), set()).add(index)

    train_size, val_size, test_size = (size * len(smiles) for size in sizes)
    big_index_sets, small_index_sets = [], []
    for index_set in scaffold_to_indices.values():
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    rng = Random(seed)
    rng.seed(seed)  # Present in the pinned Chemprop implementation.
    rng.shuffle(big_index_sets)
    rng.shuffle(small_index_sets)

    train, val, test = [], [], []
    for index_set in big_index_sets + small_index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
        elif len(val) + len(index_set) <= val_size:
            val += index_set
        else:
            test += index_set
    return {"train": set(train), "val": set(val), "test": set(test)}


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "idx": list(range(len(SMILES))),
        "smiles": SMILES,
        "FreeSolv": [float(i) for i in range(len(SMILES))],
        "img_path": [
            f"../data/images/balanced_scaffold_seed1_FreeSolv/{i}.png"
            for i in range(len(SMILES))
        ],
    })


@pytest.fixture
def split_df(df) -> pd.DataFrame:
    return _split_data_balanced_scaffold(df, seed=1)


class TestOutputShape:
    def test_group_column_values(self, split_df):
        """A1: group 列が存在し、値は train/val/test の 3 種のみ。"""
        assert "group" in split_df.columns
        assert set(split_df["group"].unique()).issubset(SPLITS)

    def test_no_data_loss(self, df, split_df):
        """A2: 行数保存かつ SMILES 集合が入力と完全一致（欠落・重複・改変なし）。"""
        assert len(split_df) == len(df)
        assert sorted(split_df["smiles"].tolist()) == sorted(df["smiles"].tolist())


class TestSplitExclusivity:
    def test_groups_are_disjoint(self, split_df):
        """A3: train/val/test 間で同一分子が重複しない。"""
        sets = {g: set(split_df[split_df["group"] == g]["smiles"]) for g in SPLITS}
        assert sets["train"].isdisjoint(sets["val"])
        assert sets["train"].isdisjoint(sets["test"])
        assert sets["val"].isdisjoint(sets["test"])


class TestScaffoldLeak:
    def test_train_test_scaffolds_disjoint(self, split_df):
        """A4: train と test の Murcko 骨格集合が交差ゼロ（scaffold リーク防止）。"""
        train_sc = {_murcko(s) for s in split_df[split_df["group"] == "train"]["smiles"]}
        test_sc = {_murcko(s) for s in split_df[split_df["group"] == "test"]["smiles"]}
        assert train_sc.isdisjoint(test_sc)

    def test_same_scaffold_same_group(self, split_df):
        """A5: 同一 scaffold を持つ分子はすべて同じ group に割り当てられる。"""
        by_scaffold = {}
        for _, row in split_df.iterrows():
            by_scaffold.setdefault(_murcko(row["smiles"]), set()).add(row["group"])
        for scaffold, groups in by_scaffold.items():
            assert len(groups) == 1, f"scaffold {scaffold!r} spans groups {groups}"


class TestDeterminism:
    def test_repeated_split_is_identical(self, df):
        """A6: 同一seedなら行順・group割当が完全一致する。"""
        first = _split_data_balanced_scaffold(df, seed=1)
        second = _split_data_balanced_scaffold(df, seed=1)
        pd.testing.assert_frame_equal(first, second)


class TestRatio:
    def test_train_is_largest_group(self, split_df):
        """A7: train が最多グループ（MoleculeNet 流）で、全群が非空。"""
        counts = split_df["group"].value_counts()
        assert set(counts.index) == SPLITS
        assert counts["train"] >= counts["val"]
        assert counts["train"] >= counts["test"]
        # 概ね 0.64/0.16/0.20（scaffold単位で境界が動くため緩めに検証）
        n = len(split_df)
        assert 0.4 <= counts["train"] / n <= 0.85


class TestAcyclicEdgeCase:
    def test_empty_fold_is_rejected(self):
        """A8: 全分子が同じ空scaffoldなら空foldを明示的に拒否する。"""
        acyclic = pd.DataFrame({
            "idx": [0, 1, 2, 3],
            "smiles": ["CCO", "CCCC", "CC(=O)O", "CCN"],
            "FreeSolv": [0.0, 1.0, 2.0, 3.0],
            "img_path": ["../data/images/balanced_scaffold_seed1_FreeSolv/0.png"]
            * 4,
        })
        with pytest.raises(ValueError, match="empty .* fold"):
            _split_data_balanced_scaffold(acyclic, seed=1)


class TestValidation:
    def test_invalid_fractions_are_rejected(self, df):
        with pytest.raises(ValueError, match="sum to 1"):
            _split_data_balanced_scaffold(
                df, frac_train=0.6, frac_valid=0.1, frac_test=0.2, seed=1
            )

    def test_single_class_classification_fold_is_rejected(self):
        split = pd.DataFrame(
            {
                "group": ["train", "train", "val", "val", "test", "test"],
                "BBBP": [0, 1, 1, 1, 0, 1],
            }
        )
        with pytest.raises(ValueError, match="val fold must contain classes 0 and 1"):
            _validate_split(split, "BBBP")

    def test_random_dispatch_preserves_legacy_split(self, df):
        expected = _split_data(df, 0.2)
        actual = _dispatch_split(df, "random", split_seed=999)
        pd.testing.assert_frame_equal(actual, expected)


class TestArtifactNaming:
    def test_random_name_is_backward_compatible(self):
        assert make_split_prefix("random", 1) == ""

    def test_scaffold_name_contains_method_and_seed(self):
        assert (
            make_split_prefix("balanced_scaffold", 3)
            == "balanced_scaffold_seed3_"
        )
        assert split_method_name("balanced_scaffold") == "balanced_scaffold"

    def test_scaffold_seed_is_required(self):
        with pytest.raises(ValueError, match="--split_seed is required"):
            make_split_prefix("balanced_scaffold", None)

    def test_graph_csv_paths_follow_split_naming(self):
        paths = make_split_csv_paths(
            "../data", "BBBP", "balanced_scaffold", 42
        )
        assert paths["train"] == Path(
            "../data/train_balanced_scaffold_seed42_BBBP_img.csv"
        )
        assert paths["val"] == Path(
            "../data/val_balanced_scaffold_seed42_BBBP_img.csv"
        )
        assert paths["test"] == Path(
            "../data/test_balanced_scaffold_seed42_BBBP_img.csv"
        )


class TestRepositoryDatasets:
    DATA_DIR = Path(__file__).resolve().parents[2] / "data"

    def test_bbbp_seed_1_has_both_classes_in_every_fold(self):
        path = self.DATA_DIR / "BBBP_img.pkl"
        if not path.exists():
            pytest.skip("repository BBBP dataset is unavailable")
        result = _split_data_balanced_scaffold(pd.read_pickle(path), seed=1)
        _validate_split(result, "BBBP")
        assert result["group"].value_counts().to_dict() == {
            "train": 1251,
            "test": 392,
            "val": 312,
        }
        assert result[result["group"] == "val"]["BBBP"].nunique() == 2

    def test_freesolv_seed_1_has_no_empty_fold(self):
        path = self.DATA_DIR / "FreeSolv_img.pkl"
        if not path.exists():
            pytest.skip("repository FreeSolv dataset is unavailable")
        source_order = pd.read_pickle(path).sort_values("index")
        result = _split_data_balanced_scaffold(source_order, seed=1)
        assert result["group"].value_counts().to_dict() == {
            "train": 410,
            "test": 152,
            "val": 80,
        }

    @pytest.mark.parametrize("dataset_name", CHEMPROP_PARITY_DATASETS)
    def test_seed_1_matches_pinned_chemprop(self, dataset_name):
        path = self.DATA_DIR / f"{dataset_name}_img.pkl"
        if not path.exists():
            pytest.skip(f"repository {dataset_name} dataset is unavailable")
        source = pd.read_pickle(path)
        if "index" in source.columns:
            source = source.sort_values("index")
        source = source.reset_index(drop=True).copy()
        source["parity_row_id"] = range(len(source))

        expected = _chemprop_64335d5_reference_indices(
            source["smiles"].tolist(), seed=1
        )
        actual = _split_data_balanced_scaffold(source, seed=1)
        actual_indices = {
            group: set(actual.loc[actual["group"] == group, "parity_row_id"])
            for group in SPLITS
        }
        assert actual_indices == expected
