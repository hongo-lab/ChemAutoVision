import warnings

import numpy as np
import pandas as pd

from notebooks.seed_metric_plots import (
    select_dmpnn_seed_runs,
    select_hpsklearn_seed_runs,
    select_model_seed_runs,
    select_paper_performance_runs,
    summarize_seed_runs,
)


def _candidate(
    run_id,
    *,
    seed,
    batch_size,
    score,
    metric="roc_auc",
    split="random",
    split_seed=np.nan,
):
    row = {
        "Run ID": run_id,
        "model_name": "autokeras",
        "target": "BBBP",
        "seed": seed,
        "ak_seed": np.nan,
        "batch_size": batch_size,
        "max_trials": 100,
        "execute_data_aug": True,
        "data_split": split,
        "split_seed": split_seed,
        "roc_auc": np.nan,
        "rmse": np.nan,
    }
    row[metric] = score
    return row


def test_selects_best_run_for_each_seed_across_all_batch_sizes():
    runs = pd.DataFrame([
        _candidate("s10-b16", seed=10, batch_size=16, score=0.80),
        _candidate("s10-b32-best", seed=10, batch_size=32, score=0.95),
        _candidate("s42-b16-best", seed=42, batch_size=16, score=0.90),
        _candidate("s42-b32", seed=42, batch_size=32, score=0.85),
        _candidate("s99-b16", seed=99, batch_size=16, score=0.70),
        _candidate("s99-b64-best", seed=99, batch_size=64, score=0.96),
    ])

    selected = select_model_seed_runs(
        runs,
        model_name="autokeras",
        targets=["BBBP"],
        metric="roc_auc",
        direction="max",
    )

    assert selected["Run ID"].tolist() == [
        "s10-b32-best",
        "s42-b16-best",
        "s99-b64-best",
    ]
    assert selected["selected_batch_size"].tolist() == [32, 16, 64]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        summary = summarize_seed_runs(
            selected,
            targets=["BBBP"],
            require_complete=False,
        )
    random_summary = summary.loc[summary["data_split"].eq("random")].iloc[0]
    assert random_summary["batch_size"] == {10: 32, 42: 16, 99: 64}


def test_selects_minimum_metric_independently_for_each_seed():
    runs = pd.DataFrame([
        _candidate("s10-b8-best", seed=10, batch_size=8, score=0.30, metric="rmse"),
        _candidate("s10-b16", seed=10, batch_size=16, score=0.40, metric="rmse"),
        _candidate("s42-b16-best", seed=42, batch_size=16, score=0.20, metric="rmse"),
        _candidate("s42-b8", seed=42, batch_size=8, score=0.50, metric="rmse"),
        _candidate("s99-b32-best", seed=99, batch_size=32, score=0.25, metric="rmse"),
        _candidate("s99-b16", seed=99, batch_size=16, score=0.60, metric="rmse"),
    ])

    selected = select_model_seed_runs(
        runs,
        model_name="autokeras",
        targets=["BBBP"],
        metric="rmse",
        direction="min",
    )

    assert selected["Run ID"].tolist() == [
        "s10-b8-best",
        "s42-b16-best",
        "s99-b32-best",
    ]


def test_hpsklearn_variants_are_selected_by_descriptor_flag():
    rows = []
    for seed in (10, 42, 99):
        for excluded, explanatory, score in (
            (True, "MFPS", 0.70),
            (False, "MFPS/Discriptor", 0.80),
        ):
            rows.append({
                "Run ID": f"hps-{seed}-{excluded}",
                "model_name": "HyperoptEstimator",
                "target": "BBBP",
                "training_seed": seed,
                "max_evals": 100,
                "data_split": "label",
                "split_seed": np.nan,
                "roc_auc": score,
                "explanatory_val": explanatory,
                "exclude_descriptors": excluded,
            })
    runs = pd.DataFrame(rows)

    ecfp6 = select_hpsklearn_seed_runs(
        runs,
        targets=["BBBP"],
        metric="roc_auc",
        direction="max",
        explanatory_val="MFPS",
        exclude_descriptors=True,
    )
    descriptor = select_hpsklearn_seed_runs(
        runs,
        targets=["BBBP"],
        metric="roc_auc",
        direction="max",
        explanatory_val="MFPS/Discriptor",
        exclude_descriptors=False,
    )

    assert ecfp6["Run ID"].tolist() == [
        "hps-10-True", "hps-42-True", "hps-99-True",
    ]
    assert descriptor["Run ID"].tolist() == [
        "hps-10-False", "hps-42-False", "hps-99-False",
    ]


def test_dmpnn_uses_split_type_and_excludes_null_class_weight():
    rows = []
    for seed in (10, 42, 99):
        rows.extend([
            {
                "Run ID": f"weighted-{seed}-b8",
                "model_name": "chemprops",
                "target": "BBBP",
                "seed": seed,
                "batch_size": 8,
                "split_type": "balanced_scaffold",
                "split_seed": 42,
                "roc_auc": 0.80,
                "class_weight": "0:1.0, 1:1.0",
            },
            {
                "Run ID": f"weighted-{seed}-best",
                "model_name": "chemprops",
                "target": "BBBP",
                "seed": seed,
                "batch_size": 32,
                "split_type": "balanced_scaffold",
                "split_seed": 42,
                "roc_auc": 0.90,
                "class_weight": "0:1.0, 1:1.0",
            },
            {
                "Run ID": f"unweighted-{seed}",
                "model_name": "chemprop",
                "target": "BBBP",
                "seed": seed,
                "batch_size": 64,
                "split_type": "balanced_scaffold",
                "split_seed": 42,
                "roc_auc": 0.99,
                "class_weight": np.nan,
            },
        ])

    selected = select_dmpnn_seed_runs(
        pd.DataFrame(rows),
        targets=["BBBP"],
        metric="roc_auc",
        direction="max",
        require_class_weight=True,
    )

    assert selected["Run ID"].tolist() == [
        "weighted-10-best", "weighted-42-best", "weighted-99-best",
    ]
    assert selected["data_split_normalized"].eq("balanced_scaffold").all()


def _paper_row(
    run_id,
    *,
    model_name,
    target,
    seed=np.nan,
    training_seed=np.nan,
    batch_size=np.nan,
    roc_auc=np.nan,
    rmse=np.nan,
    class_weight=np.nan,
    weighted_balanced=False,
):
    return {
        "Run ID": run_id,
        "model_name": model_name,
        "target": target,
        "seed": seed,
        "ak_seed": np.nan,
        "training_seed": training_seed,
        "batch_size": batch_size,
        "max_trials": 100,
        "max_evals": 100,
        "execute_data_aug": True,
        "data_split": "random",
        "split_seed": np.nan,
        "roc_auc": roc_auc,
        "rmse": rmse,
        "class_weight": class_weight,
        "weighted_balanced": weighted_balanced,
        "explanatory_val": "MFPS/Discriptor",
        "exclude_descriptors": False,
    }


def test_paper_run_selection_recomputes_from_supplied_raw_frames():
    classification_rows = []
    for model_name in ("autokeras", "resnet18"):
        for seed, best_batch in ((10, 32), (42, 16), (99, 64)):
            classification_rows.extend([
                _paper_row(
                    f"{model_name}-s{seed}-b8",
                    model_name=model_name,
                    target="BBBP",
                    seed=seed,
                    batch_size=8,
                    roc_auc=0.70,
                    class_weight="balanced",
                ),
                _paper_row(
                    f"{model_name}-s{seed}-best",
                    model_name=model_name,
                    target="BBBP",
                    seed=seed,
                    batch_size=best_batch,
                    roc_auc=0.90 + seed / 1000,
                    class_weight="balanced",
                ),
            ])
    classification_rows.append(
        _paper_row(
            "autokeras-s10-b128-best",
            model_name="autokeras",
            target="BBBP",
            seed=10,
            batch_size=128,
            roc_auc=0.99,
            class_weight="balanced",
        )
    )
    classification_rows.extend(
        _paper_row(
            f"hps-s{seed}",
            model_name="HyperoptEstimator",
            target="BBBP",
            training_seed=seed,
            roc_auc=0.80,
        )
        for seed in (10, 42, 99)
    )
    classification_rows.extend([
        _paper_row(
            "imagemol-weighted",
            model_name="ImageMol",
            target="BBBP",
            roc_auc=0.80,
            weighted_balanced=True,
        ),
        _paper_row(
            "imagemol-unweighted-better",
            model_name="ImageMol",
            target="BBBP",
            roc_auc=0.99,
            weighted_balanced=False,
        ),
        _paper_row(
            "imagemol-batch128-better",
            model_name="ImageMol",
            target="BBBP",
            batch_size=128,
            roc_auc=0.98,
            weighted_balanced=True,
        ),
    ])

    regression_rows = []
    for model_name in ("autokeras", "resnet18"):
        for seed, best_batch in ((10, 32), (42, 16), (99, 64)):
            regression_rows.extend([
                _paper_row(
                    f"{model_name}-s{seed}-b8",
                    model_name=model_name,
                    target="FreeSolv",
                    seed=seed,
                    batch_size=8,
                    rmse=0.70,
                ),
                _paper_row(
                    f"{model_name}-s{seed}-best",
                    model_name=model_name,
                    target="FreeSolv",
                    seed=seed,
                    batch_size=best_batch,
                    rmse=0.30 + seed / 1000,
                ),
            ])
    regression_rows.extend(
        _paper_row(
            f"hps-s{seed}",
            model_name="HyperoptEstimator",
            target="FreeSolv",
            training_seed=seed,
            rmse=0.40,
        )
        for seed in (10, 42, 99)
    )
    regression_rows.append(
        _paper_row(
            "imagemol",
            model_name="ImageMol",
            target="FreeSolv",
            rmse=0.35,
        )
    )

    classification = pd.DataFrame(classification_rows)
    regression = pd.DataFrame(regression_rows)
    selected = select_paper_performance_runs(
        classification,
        regression,
        classification_targets=["BBBP"],
        regression_targets=["FreeSolv"],
    )

    chem_cls = selected["classification"]["repeated_runs"]["ChemAutoVision"]
    assert chem_cls["selected_batch_size"].tolist() == [128, 16, 64]
    imagemol = selected["classification"]["point_runs"]["ImageMol"]
    assert imagemol["Run ID"].tolist() == ["imagemol-weighted"]

    updated = classification.copy()
    updated.loc[updated["Run ID"].eq("autokeras-s10-b8"), "roc_auc"] = 1.0
    recomputed = select_paper_performance_runs(
        updated,
        regression,
        classification_targets=["BBBP"],
        regression_targets=["FreeSolv"],
    )
    recomputed_chem = recomputed["classification"]["repeated_runs"]["ChemAutoVision"]
    seed_10 = recomputed_chem.loc[recomputed_chem["compare_seed"].eq(10)].iloc[0]
    assert seed_10["Run ID"] == "autokeras-s10-b8"
    assert seed_10["selected_batch_size"] == 8
