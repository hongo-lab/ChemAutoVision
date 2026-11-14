import pandas as pd
import pytest

from data.label_utils import (
    QUARTILE_EXTREMES,
    TERTILE_EXTREMES,
    build_quantile_classification,
)
from utils.split import make_dataset_pickle_path, make_split_csv_paths


def _frame(task_name: str = "ESOL") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": [f"mol-{index}" for index in range(12)],
            task_name: [float(index) for index in range(12)],
            "group": ["train"] * 6 + ["val"] * 3 + ["test"] * 3,
            "img_path": [f"{index}.png" for index in range(12)],
        }
    )


def test_test_targets_do_not_change_fitted_thresholds_or_train_labels():
    original = _frame()
    labeled1, thresholds1 = build_quantile_classification(
        original, "ESOL", TERTILE_EXTREMES
    )

    modified = original.copy()
    modified.loc[modified["group"] == "test", "ESOL"] += 10_000
    labeled2, thresholds2 = build_quantile_classification(
        modified, "ESOL", TERTILE_EXTREMES
    )

    assert thresholds1 == thresholds2
    pd.testing.assert_series_equal(
        labeled1.loc[labeled1["group"] == "train", "ESOL"].reset_index(drop=True),
        labeled2.loc[labeled2["group"] == "train", "ESOL"].reset_index(drop=True),
    )


@pytest.mark.parametrize(
    ("task_name", "expected_low", "expected_high"),
    [
        ("ESOL", 0, 1),
        ("Lipo", 0, 1),
        ("FreeSolv", 1, 0),
    ],
)
def test_task_direction_is_applied(task_name, expected_low, expected_high):
    labeled, _ = build_quantile_classification(
        _frame(task_name), task_name, TERTILE_EXTREMES
    )
    train = labeled[labeled["group"] == "train"]
    assert train.iloc[0][task_name] == expected_low
    assert train.iloc[-1][task_name] == expected_high


def test_quartile_uses_train_quartiles_and_preserves_groups():
    frame = _frame()
    labeled, thresholds = build_quantile_classification(
        frame, "ESOL", QUARTILE_EXTREMES
    )

    assert thresholds.lower == pytest.approx(1.25)
    assert thresholds.upper == pytest.approx(3.75)
    assert labeled["group"].tolist() == frame["group"].tolist()
    assert labeled["ESOL_continuous"].tolist() == frame["ESOL"].tolist()


def test_derived_artifact_paths_include_scheme_and_split():
    pkl = make_dataset_pickle_path(
        "../data",
        "FreeSolv",
        "balanced_scaffold",
        42,
        TERTILE_EXTREMES,
    )
    csvs = make_split_csv_paths(
        "../data",
        "FreeSolv",
        "balanced_scaffold",
        42,
        QUARTILE_EXTREMES,
    )

    assert pkl.name == "balanced_scaffold_seed42_cls_t_FreeSolv_img.pkl"
    assert (
        csvs["train"].name
        == "train_balanced_scaffold_seed42_cls_q_FreeSolv_img.csv"
    )
