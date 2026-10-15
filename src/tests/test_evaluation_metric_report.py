from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from notebooks.evaluation_metric_report import (
    build_metric_table,
    compare_architecture_records,
    compare_selected_hyperparameters,
    flatten_legacy_hyperparameters,
    compare_with_paper_baseline,
    load_hyperopt_test_runs,
    load_mlflow_test_runs,
    summarize_test_metrics,
    validate_imagemol_v2_candidates,
)


def _classification_row(run_id: str, start_time: str, roc_auc: float) -> dict:
    return {
        "Start Time": start_time,
        "Run ID": run_id,
        "Status": "FINISHED",
        "model_name": "HyperoptEstimator",
        "target": "CYP3A4",
        "training_seed": 10,
        "max_evals": 100,
        "explanatory_val": "MFPS/Discriptor",
        "exclude_descriptors": False,
        "data_split": "label",
        "split_seed": np.nan,
        "dataset_split_seed": np.nan,
        "roc_auc": roc_auc,
        "acc": 0.8,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "mcc": 0.6,
    }


def _regression_row() -> dict:
    return {
        "Start Time": "2026-09-01 00:00:00",
        "Run ID": "regression",
        "Status": "FINISHED",
        "model_name": "HyperoptEstimator",
        "target": "Dalaney",
        "training_seed": 10,
        "max_evals": 100,
        "explanatory_val": "MFPS",
        "exclude_descriptors": True,
        "data_split": "balanced_scaffold",
        "split_seed": 42,
        "dataset_split_seed": np.nan,
        "rmse": 0.5,
        "mse": 0.25,
        "mae": 0.4,
        "r2": 0.7,
    }


def test_hyperopt_loader_uses_latest_run_without_optimizing_test_metric(
    tmp_path: Path,
):
    classification_path = tmp_path / "classification.csv"
    regression_path = tmp_path / "regression.csv"
    pd.DataFrame([
        _classification_row("older-better", "2026-09-01 00:00:00", 0.99),
        _classification_row("latest-worse", "2026-09-02 00:00:00", 0.80),
    ]).to_csv(classification_path, index=False)
    pd.DataFrame([_regression_row()]).to_csv(regression_path, index=False)

    runs = load_hyperopt_test_runs(
        classification_path,
        regression_path,
        seeds=(10,),
    )

    classification = runs.loc[runs["task_type"].eq("classification")].iloc[0]
    assert classification["source_run_id"] == "latest-worse"
    assert classification["test_roc_auc"] == pytest.approx(0.80)
    assert classification["split_type"] == "random"
    regression = runs.loc[runs["task_type"].eq("regression")].iloc[0]
    assert regression["target"] == "ESOL"
    assert regression["test_rmse"] == pytest.approx(0.5)


class _FakeClient:
    def __init__(self, runs):
        self.runs = runs

    def search_runs(self, **kwargs):
        return self.runs


def test_mlflow_loader_reads_only_finished_matching_evaluation_runs():
    def run(
        run_id,
        *,
        status="FINISHED",
        version="1",
        model_type="ChemAutoVision",
        training_seed="10",
    ):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, status=status),
            data=SimpleNamespace(
                tags={
                    "selected_for_test": "true",
                    "evaluation_version": version,
                    "model_type": model_type,
                    "task_type": "classification",
                    "target": "BBBP",
                    "split_type": "random",
                    "training_seed": training_seed,
                    "data_augmentation": "enabled",
                },
                params={"batch_size": "32", "learning_rate": "0.005"},
                metrics={
                    "test_roc_auc": 0.9,
                    "test_accuracy": 0.8,
                    "test_precision": 0.8,
                    "test_recall": 0.8,
                    "test_f1": 0.8,
                    "test_mcc": 0.6,
                },
            ),
        )

    loaded = load_mlflow_test_runs(
        "unused",
        ("classification",),
        client=_FakeClient([
            run("included"),
            run(
                "imagemol",
                version="imagemol-validation-selection-v2",
                model_type="ImageMol",
                training_seed=None,
            ),
            run(
                "obsolete-imagemol",
                version="imagemol-validation-selection-v1",
                model_type="ImageMol",
                training_seed=None,
            ),
            run("failed", status="FAILED"),
            run("old-version", version="0"),
        ]),
    )

    assert loaded["source_run_id"].tolist() == ["included", "imagemol"]
    assert loaded.iloc[0]["test_roc_auc"] == pytest.approx(0.9)
    assert loaded.iloc[0]["batch_size"] == pytest.approx(32)
    assert loaded.iloc[1]["model_type"] == "ImageMol"
    assert pd.isna(loaded.iloc[1]["training_seed"])


def test_imagemol_v2_candidate_validation_requires_complete_lr_batch_grid():
    def run(task_type, target, split_type, batch_size, learning_rate):
        selected = batch_size == 8 and learning_rate == min(
            {"classification": (0.0005, 0.005, 0.05, 0.5),
             "regression": (0.0005, 0.005)}[task_type]
        )
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id=f"{task_type}-{target}-{split_type}-{batch_size}-{learning_rate}",
                status="FINISHED",
            ),
            data=SimpleNamespace(
                tags={
                    "model_type": "ImageMol",
                    "evaluation_version": "imagemol-validation-selection-v2",
                    "task_type": task_type,
                    "target": target,
                    "split_type": split_type,
                    "selected_for_test": str(selected).lower(),
                },
                params={
                    "batch_size": str(batch_size),
                    "learning_rate": str(learning_rate),
                },
                metrics={},
            ),
        )

    runs = []
    targets = {
        "classification": ("BBBP", "P-gp", "CYP3A4", "hERG"),
        "regression": ("FreeSolv", "ESOL", "Lipo"),
    }
    learning_rates = {
        "classification": (0.0005, 0.005, 0.05, 0.5),
        "regression": (0.0005, 0.005),
    }
    for task_type, task_targets in targets.items():
        for target in task_targets:
            for split_type in ("random", "balanced_scaffold"):
                for batch_size in (8, 16, 32, 64):
                    for learning_rate in learning_rates[task_type]:
                        runs.append(run(
                            task_type, target, split_type, batch_size, learning_rate
                        ))

    summary = validate_imagemol_v2_candidates(
        "unused", ("classification", "regression"), client=_FakeClient(runs)
    )

    assert len(runs) == 176
    assert len(summary) == 14
    assert summary["selected_runs"].eq(1).all()
    assert set(summary.loc[
        summary["task_type"].eq("classification"), "candidate_runs"
    ]) == {16}
    assert set(summary.loc[
        summary["task_type"].eq("regression"), "candidate_runs"
    ]) == {8}


def test_summary_uses_sample_sd_and_table_marks_best_mean():
    records = []
    for model, values in {
        "ResNet18": (0.7, 0.8, 0.9),
        "ChemAutoVision": (0.8, 0.9, 1.0),
    }.items():
        for seed, value in zip((10, 42, 99), values):
            records.append({
                "source": "test",
                "source_run_id": f"{model}-{seed}",
                "model_type": model,
                "task_type": "classification",
                "target": "BBBP",
                "split_type": "random",
                "split_seed": np.nan,
                "training_seed": seed,
                "test_roc_auc": value,
                "test_accuracy": value,
                "test_precision": value,
                "test_recall": value,
                "test_f1": value,
                "test_mcc": value,
            })
    summary = summarize_test_metrics(pd.DataFrame(records))
    chem_auc = summary.loc[
        summary["model_type"].eq("ChemAutoVision")
        & summary["metric"].eq("test_roc_auc")
    ].iloc[0]
    assert chem_auc["mean"] == pytest.approx(0.9)
    assert chem_auc["std"] == pytest.approx(0.1)

    display, latex = build_metric_table(
        summary,
        task_type="classification",
        split_type="random",
        targets=("BBBP",),
        model_order=("ResNet18", "ChemAutoVision"),
    )
    assert display.loc[1, "AUC"] == "0.900 ± 0.100"
    assert latex.loc[1, "AUC"].startswith("\\textbf{")


def test_paper_comparison_orients_lower_rmse_as_improvement():
    baseline = pd.DataFrame([{
        "task_type": "regression",
        "model_type": "ChemAutoVision",
        "target": "ESOL",
        "split_type": "random",
        "metric": "test_rmse",
        "old_mean": 0.50,
        "old_std": 0.02,
    }])
    summary = pd.DataFrame([{
        "task_type": "regression",
        "model_type": "ChemAutoVision",
        "target": "ESOL",
        "split_type": "random",
        "split_seed": np.nan,
        "metric": "test_rmse",
        "n_seeds": 3,
        "mean": 0.45,
        "std": 0.01,
        "training_seeds": (10, 42, 99),
        "source_run_ids": ("a", "b", "c"),
    }])

    comparison = compare_with_paper_baseline(baseline, summary).iloc[0]

    assert comparison["delta"] == pytest.approx(-0.05)
    assert comparison["improvement"] == pytest.approx(0.05)
    assert comparison["impact_level"] == "major"


def test_architecture_comparison_reports_changed_component_and_run():
    base = {
        "task_type": "classification",
        "data_split": "random",
        "target": "BBBP",
        "seed": 10,
        "batch_size": 32,
        "model_path": "old.keras",
        "model_exists": True,
        "Input": "InputLayer (256 × 256 × 3)",
        "CNN Architecture": "Xception",
    }
    old = pd.DataFrame([{**base, "Run ID": "old"}])
    new = pd.DataFrame([{
        **base,
        "Run ID": "new",
        "model_path": "new.keras",
        "CNN Architecture": "ResNet50",
    }])

    comparison = compare_architecture_records(
        old,
        new,
        components=("Input", "CNN Architecture"),
    )

    input_row = comparison.loc[comparison["component"].eq("Input")].iloc[0]
    cnn_row = comparison.loc[
        comparison["component"].eq("CNN Architecture")
    ].iloc[0]
    assert not input_row["changed"]
    assert cnn_row["changed"]
    assert cnn_row["run_changed"]
    assert cnn_row["old_value"] == "Xception"
    assert cnn_row["new_value"] == "ResNet50"


def test_selected_hyperparameter_comparison_reports_changes_and_missing_values():
    old_runs = {
        "classification": {
            "repeated_runs": {
                "ChemAutoVision": pd.DataFrame([{
                    "target_display": "BBBP",
                    "data_split_normalized": "random",
                    "compare_seed": 10,
                    "selected_batch_size": 16,
                    "learning_rate": 0.001,
                    "Run ID": "old",
                }]),
            },
            "point_runs": {
                "ImageMol": pd.DataFrame([{
                    "target_display": "BBBP",
                    "data_split_normalized": "random",
                    "batch_size": 64,
                    "learning_rate": np.nan,
                    "Run ID": "old-image",
                }]),
            },
        }
    }
    old = flatten_legacy_hyperparameters(old_runs)
    new = pd.DataFrame([
        {
            "task_type": "classification", "model_type": "ChemAutoVision",
            "target": "BBBP", "split_type": "random", "training_seed": 10,
            "source_run_id": "new", "batch_size": 32, "learning_rate": 0.001,
        },
        {
            "task_type": "classification", "model_type": "ImageMol",
            "target": "BBBP", "split_type": "random", "training_seed": np.nan,
            "source_run_id": "new-image", "batch_size": 64, "learning_rate": np.nan,
        },
    ])

    comparison = compare_selected_hyperparameters(old, new)
    chem = comparison.loc[comparison["model_type"].eq("ChemAutoVision")].iloc[0]
    image = comparison.loc[comparison["model_type"].eq("ImageMol")].iloc[0]
    assert chem["batch_size_status"] == "changed"
    assert chem["learning_rate_status"] == "unchanged"
    assert image["batch_size_status"] == "unchanged"
    assert image["learning_rate_status"] == "not_recorded"
