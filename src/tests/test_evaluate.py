import csv
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import evaluate
from evaluate import (
    Candidate,
    EvaluationFailuresError,
    RESULT_COLUMNS,
    SELECTED_COLUMNS,
    _candidate_from_mlflow_run,
    _get_finished_run,
    _load_subset_frame,
    calculate_metrics,
    data_augmentation_for,
    data_augmentation_source_for,
    main,
    run_selection,
    run_test,
    run_validation,
    normalize_split_type,
    resolve_candidate_model_path,
    select_best_rows,
    write_test_summary,
)


def candidate(family: str = "chem_autovision") -> Candidate:
    return Candidate(
        family=family,
        target="BBBP",
        task_type="classification",
        split_type="random",
        split_seed=None,
        training_seed=42,
        batch_size=8,
        run_id="source",
        model_path=Path("model.h5"),
    )


class FakeRun:
    def __init__(self, run_id: str, metrics=None, tags=None, params=None):
        self.info = SimpleNamespace(
            run_id=run_id, status="FINISHED", experiment_id="source-experiment"
        )
        self.data = SimpleNamespace(
            metrics=dict(metrics or {}),
            tags=dict(tags or {}),
            params=dict(params or {}),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeMlflow:
    def __init__(self, runs=None):
        self.logged_metric_dicts = []
        self.logged_single_metrics = []
        self.tags = []
        self.artifacts = []
        self.runs = {run.info.run_id: run for run in (runs or [FakeRun("source-run")])}
        self.active_run_id = None

    def get_run(self, run_id):
        if run_id not in self.runs:
            raise RuntimeError("run does not exist")
        return self.runs[run_id]

    def start_run(self, **kwargs):
        run_id = kwargs["run_id"]
        self.active_run_id = run_id
        return self.runs[run_id]

    def set_tags(self, tags):
        self.tags.append(tags)
        self.runs[self.active_run_id].data.tags.update(tags)

    def log_params(self, params):
        pass

    def log_metrics(self, metrics):
        self.logged_metric_dicts.append(metrics)
        self.runs[self.active_run_id].data.metrics.update(metrics)

    def log_metric(self, key, value):
        self.logged_single_metrics.append((key, value))

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))


def regression_candidate(model_path: Path) -> Candidate:
    return Candidate(
        family="dmpnn",
        target="FreeSolv",
        task_type="regression",
        split_type="random",
        split_seed=None,
        training_seed=42,
        batch_size=8,
        run_id="source-run",
        model_path=model_path,
    )


def evaluation_args(results_dir: Path):
    return SimpleNamespace(
        results_dir=results_dir,
        overwrite=False,
        tracking_uri="unused",
        source_classification_experiment_id="classification-experiment",
        source_regression_experiment_id="regression-experiment",
        allow_incomplete_groups=False,
        family=None,
        target=None,
        split_type=None,
        training_seed=None,
    )


def test_metric_names_are_prefixed_by_the_caller():
    classification = calculate_metrics(
        "classification", np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.8])
    )
    regression = calculate_metrics(
        "regression", np.array([1.0, 2.0]), np.array([1.5, 1.5])
    )

    assert set(classification) == {
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
    }
    assert set(regression) == {"rmse", "mse", "mae", "r2"}


def test_data_augmentation_is_explicit_for_each_model_type():
    assert data_augmentation_for(candidate(), {"execute_data_aug": "True"}) == "enabled"
    assert (
        data_augmentation_for(candidate(), {"execute_data_aug": "False"}) == "disabled"
    )
    assert data_augmentation_for(candidate()) == "unknown"
    assert data_augmentation_source_for(candidate()) == "missing"
    assert (
        data_augmentation_source_for(candidate(), {"execute_data_aug": "True"})
        == "source_metadata"
    )
    assert data_augmentation_for(candidate("resnet18_aug")) == "unknown"
    assert data_augmentation_for(candidate("dmpnn")) == "not_applicable"


def test_missing_mlflow_run_is_an_error():
    class MissingSourceMlflow:
        def get_run(self, run_id):
            raise RuntimeError("run does not exist")

    with pytest.raises(ValueError, match="MLflow run not found: missing-run"):
        _get_finished_run(MissingSourceMlflow(), "missing-run")


def test_finished_mlflow_run_is_returned():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run", status="FINISHED"),
        data=SimpleNamespace(params={}, tags={}, metrics={}),
    )
    mlflow = SimpleNamespace(get_run=lambda _: run)
    assert _get_finished_run(mlflow, "run") is run


def test_scaffold_label_is_normalized_to_balanced_scaffold():
    assert normalize_split_type("scaffold") == "balanced_scaffold"
    assert normalize_split_type("balanced_scaffold") == "balanced_scaffold"
    assert normalize_split_type("random") == "random"


def make_result_row(batch_size: int, score: float, task_type: str = "classification"):
    metric_name = "val_roc_auc" if task_type == "classification" else "val_rmse"
    return {
        "evaluation_key": f"key-{batch_size}",
        "run_id": f"run-{batch_size}",
        "status": "FINISHED",
        "family": "chem_autovision",
        "model_type": "ChemAutoVision",
        "task_type": task_type,
        "target": "BBBP" if task_type == "classification" else "FreeSolv",
        "split_type": "random",
        "split_seed": "",
        "training_seed": "42",
        "batch_size": str(batch_size),
        metric_name: str(score),
    }


def test_selection_maximizes_validation_roc_auc_and_uses_batch_size_as_tie_breaker():
    rows = [
        make_result_row(8, 0.7),
        make_result_row(16, 0.8),
        make_result_row(32, 0.8),
        make_result_row(64, 0.6),
    ]

    selected, ranks = select_best_rows(rows)

    assert selected[0]["batch_size"] == "16"
    assert selected[0]["selection_metric"] == "val_roc_auc"
    assert ranks["key-16"] == 1


def test_selection_minimizes_validation_rmse():
    rows = [
        make_result_row(8, 1.1, "regression"),
        make_result_row(16, 0.8, "regression"),
        make_result_row(32, 0.9, "regression"),
        make_result_row(64, 1.0, "regression"),
    ]

    selected, _ = select_best_rows(rows)

    assert selected[0]["batch_size"] == "16"
    assert selected[0]["selection_metric"] == "val_rmse"


def test_selection_rejects_incomplete_batch_size_group():
    rows = [make_result_row(8, 0.7), make_result_row(16, 0.8)]

    with pytest.raises(ValueError, match="all four batch sizes"):
        select_best_rows(rows)


def test_selection_rejects_group_when_every_validation_failed():
    rows = [
        {
            **make_result_row(batch_size, 0.0),
            "status": "FAILED",
        }
        for batch_size in (8, 16, 32, 64)
    ]

    with pytest.raises(ValueError, match=r"got \[\]"):
        select_best_rows(rows)


def test_mlflow_training_run_is_converted_to_candidate(tmp_path: Path):
    model = tmp_path / "model.h5"
    model.write_bytes(b"model")
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run", status="FINISHED"),
        data=SimpleNamespace(
            params={"seed": "42", "batch_size": "8", "class_weight": "0:1, 1:2"},
            tags={
                "model_name": "autokeras",
                "target": "BBBP",
                "data_split": "random",
                "model_path": str(model),
                "execute_data_aug": "True",
            },
            metrics={},
        ),
    )

    result = _candidate_from_mlflow_run(run, repo_root=tmp_path)

    assert result is not None
    assert result.run_id == "run"
    assert result.model_type == "ChemAutoVision"
    assert result.model_path == model.resolve()


def test_non_augmented_resnet_run_is_out_of_scope(tmp_path: Path):
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run", status="FINISHED"),
        data=SimpleNamespace(
            params={"seed": "42", "batch_size": "8", "class_weight": "0:1, 1:2"},
            tags={
                "model_name": "resnet18",
                "target": "BBBP",
                "data_split": "random",
                "model_path": str(tmp_path / "model.h5"),
                "execute_data_aug": "False",
            },
            metrics={},
        ),
    )

    assert _candidate_from_mlflow_run(run, repo_root=tmp_path) is None


def test_model_path_from_another_host_is_reanchored(tmp_path: Path):
    model = tmp_path / "models" / "model.h5"
    model.parent.mkdir()
    model.write_bytes(b"model")

    resolved = resolve_candidate_model_path(
        {"model_path_from_csv": "/server/work/ChemAutoVision/models/model.h5"},
        tmp_path,
    )

    assert resolved == model.resolve()


def test_test_summary_calculates_mean_and_sample_standard_deviation(tmp_path: Path):
    metrics_path = tmp_path / "test_metrics.csv"
    output_path = tmp_path / "summary.csv"
    fields = [
        "status",
        "model_type",
        "target",
        "task_type",
        "split_type",
        "data_augmentation",
        "test_rmse",
        "test_mse",
        "test_mae",
        "test_r2",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rmse in (1.0, 2.0, 3.0):
            writer.writerow(
                {
                    "status": "FINISHED",
                    "model_type": "D-MPNN",
                    "target": "FreeSolv",
                    "task_type": "regression",
                    "split_type": "random",
                    "data_augmentation": "not_applicable",
                    "test_rmse": rmse,
                    "test_mse": rmse**2,
                    "test_mae": rmse,
                    "test_r2": 0.5,
                }
            )

    write_test_summary(metrics_path, output_path)
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summary = {row["metric"]: row for row in csv.DictReader(handle)}

    assert float(summary["test_rmse"]["mean"]) == pytest.approx(2.0)
    assert float(summary["test_rmse"]["std"]) == pytest.approx(1.0)
    assert summary["test_rmse"]["n_seeds"] == "3"


def test_subset_loader_returns_validation_and_test_rows_separately(
    tmp_path: Path, monkeypatch
):
    dataset_path = tmp_path / "FreeSolv_img.pkl"
    pd.DataFrame(
        {
            "group": ["train", "val", "val", "test", "test"],
            "smiles": ["train", "val-1", "val-2", "test-1", "test-2"],
            "FreeSolv": [0.0, 11.0, 12.0, 21.0, 22.0],
        }
    ).to_pickle(dataset_path)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model = regression_candidate(model_path)
    monkeypatch.setattr(evaluate, "_dataset_pickle_path", lambda _: dataset_path)

    validation_frame, _ = _load_subset_frame(model, "val")
    test_frame, _ = _load_subset_frame(model, "test")

    assert validation_frame["smiles"].tolist() == ["val-1", "val-2"]
    assert validation_frame["FreeSolv"].tolist() == [11.0, 12.0]
    assert test_frame["smiles"].tolist() == ["test-1", "test-2"]
    assert test_frame["FreeSolv"].tolist() == [21.0, 22.0]
    assert set(validation_frame.index).isdisjoint(test_frame.index)


def test_validation_uses_validation_dataset_and_logs_only_val_metrics(tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model = regression_candidate(model_path)
    dataset_path = tmp_path / "validation.csv"
    dataset_path.write_text("smiles,FreeSolv\nval,1.0\n", encoding="utf-8")
    validation_frame = pd.DataFrame(
        {"smiles": ["val-1", "val-2"], "FreeSolv": [1.0, 3.0]},
        index=[10, 11],
    )
    fake_mlflow = FakeMlflow()
    subsets = []

    def fake_predict(candidate, subset, results_dir):
        subsets.append(subset)
        return (
            validation_frame,
            np.array([1.0, 3.0]),
            np.array([1.5, 2.5]),
            dataset_path,
            "validation-fingerprint",
            0.25,
        )

    with (
        patch.object(evaluate, "_set_tracking_uri", return_value=fake_mlflow),
        patch.object(evaluate, "predict_candidate", side_effect=fake_predict),
    ):
        rows = run_validation(evaluation_args(tmp_path / "results"), [model])

    assert subsets == ["val"]
    assert rows[0]["dataset_path"] == str(dataset_path)
    assert rows[0]["dataset_sha256"] == "validation-fingerprint"
    assert rows[0]["val_rmse"] == pytest.approx(0.5)
    assert not any(key.startswith("test_") for key in rows[0])
    logged_keys = set().union(*fake_mlflow.logged_metric_dicts)
    assert logged_keys == {
        "val_rmse",
        "val_mse",
        "val_mae",
        "val_r2",
        "val_inference_time_sec",
    }
    assert fake_mlflow.artifacts == []
    validation_tags = {
        key: value for tag_batch in fake_mlflow.tags for key, value in tag_batch.items()
    }
    assert validation_tags["val_prediction_path"] == rows[0]["prediction_path"]
    assert Path(validation_tags["val_prediction_path"]).is_file()


def test_test_evaluation_uses_test_dataset_and_logs_only_test_metrics(tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model = regression_candidate(model_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    selected_path = results_dir / "selected_models.csv"
    selected_row = {
        "selection_group": model.selection_group,
        "selection_metric": "val_rmse",
        "selection_metric_value": "0.4",
        "selection_rank": "1",
        "evaluation_key": model.evaluation_key,
        "run_id": model.run_id,
        "status": "FINISHED",
        "family": model.family,
        "model_type": model.model_type,
        "task_type": model.task_type,
        "target": model.target,
        "split_type": model.split_type,
        "split_seed": "",
        "training_seed": model.training_seed,
        "batch_size": model.batch_size,
        "model_path": str(model.model_path),
        "data_augmentation": "not_applicable",
        "val_rmse": "0.4",
    }
    with selected_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_COLUMNS)
        writer.writeheader()
        writer.writerow({name: selected_row.get(name, "") for name in SELECTED_COLUMNS})

    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("smiles,FreeSolv\ntest,10.0\n", encoding="utf-8")
    test_frame = pd.DataFrame(
        {"smiles": ["test-1", "test-2"], "FreeSolv": [10.0, 14.0]},
        index=[20, 21],
    )
    fake_mlflow = FakeMlflow()
    subsets = []

    def fake_predict(candidate, subset, current_results_dir):
        subsets.append(subset)
        return (
            test_frame,
            np.array([10.0, 14.0]),
            np.array([11.0, 13.0]),
            dataset_path,
            "test-fingerprint",
            0.5,
        )

    with (
        patch.object(evaluate, "_set_tracking_uri", return_value=fake_mlflow),
        patch.object(evaluate, "predict_candidate", side_effect=fake_predict),
    ):
        rows = run_test(evaluation_args(results_dir))

    assert subsets == ["test"]
    assert rows[0]["dataset_path"] == str(dataset_path)
    assert rows[0]["dataset_sha256"] == "test-fingerprint"
    assert rows[0]["test_rmse"] == pytest.approx(1.0)
    assert not any(key.startswith("val_") for key in rows[0])
    logged_keys = set().union(*fake_mlflow.logged_metric_dicts)
    assert logged_keys == {
        "test_rmse",
        "test_mse",
        "test_mae",
        "test_r2",
        "test_inference_time_sec",
    }
    assert fake_mlflow.artifacts == []
    test_tags = {
        key: value for tag_batch in fake_mlflow.tags for key, value in tag_batch.items()
    }
    assert test_tags["test_prediction_path"] == rows[0]["prediction_path"]
    assert Path(test_tags["test_prediction_path"]).is_file()


def test_validate_command_routes_only_to_validation_dataset_pipeline():
    candidates = [candidate()]
    with (
        patch.object(evaluate, "discover_candidates", return_value=candidates),
        patch.object(evaluate, "audit_candidates", return_value=0),
        patch.object(evaluate, "run_validation") as run_validation_mock,
        patch.object(evaluate, "run_selection") as run_selection_mock,
        patch.object(evaluate, "run_test") as run_test_mock,
    ):
        exit_code = main(["validate"])

    assert exit_code == 0
    run_validation_mock.assert_called_once()
    assert run_validation_mock.call_args.args[1] == candidates
    run_selection_mock.assert_not_called()
    run_test_mock.assert_not_called()


def test_test_command_routes_only_to_test_dataset_pipeline():
    candidates = [candidate()]
    with (
        patch.object(evaluate, "discover_candidates", return_value=candidates),
        patch.object(evaluate, "audit_candidates", return_value=0),
        patch.object(evaluate, "run_validation") as run_validation_mock,
        patch.object(evaluate, "run_selection") as run_selection_mock,
        patch.object(evaluate, "run_test") as run_test_mock,
    ):
        exit_code = main(["test"])

    assert exit_code == 0
    run_validation_mock.assert_not_called()
    run_selection_mock.assert_not_called()
    run_test_mock.assert_called_once()


def test_dry_run_returns_nonzero_when_a_model_file_is_missing(tmp_path: Path):
    missing = regression_candidate(tmp_path / "missing-model.pt")
    with (patch.object(evaluate, "discover_candidates", return_value=[missing]),):
        exit_code = main(["dry-run"])

    assert exit_code == 1


def test_validation_records_missing_model_and_raises_error(tmp_path: Path):
    missing = regression_candidate(tmp_path / "missing-model.pt")
    results_dir = tmp_path / "results"

    with patch.object(evaluate, "_set_tracking_uri", return_value=FakeMlflow()):
        with pytest.raises(
            EvaluationFailuresError, match="validation failed for 1 model"
        ):
            run_validation(evaluation_args(results_dir), [missing])

    with (results_dir / "validation_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "FAILED"
    assert "FileNotFoundError" in rows[0]["error"]
    assert "missing-model.pt" in rows[0]["error"]


def test_test_evaluation_records_missing_selected_model_and_raises_error(
    tmp_path: Path,
):
    missing = regression_candidate(tmp_path / "missing-selected-model.pt")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    selected_path = results_dir / "selected_models.csv"
    selected_row = {
        "selection_group": missing.selection_group,
        "selection_metric": "val_rmse",
        "selection_metric_value": "0.4",
        "selection_rank": "1",
        "evaluation_key": missing.evaluation_key,
        "run_id": missing.run_id,
        "status": "FINISHED",
        "family": missing.family,
        "model_type": missing.model_type,
        "task_type": missing.task_type,
        "target": missing.target,
        "split_type": missing.split_type,
        "split_seed": "",
        "training_seed": missing.training_seed,
        "batch_size": missing.batch_size,
        "model_path": str(missing.model_path),
        "data_augmentation": "not_applicable",
        "val_rmse": "0.4",
    }
    with selected_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_COLUMNS)
        writer.writeheader()
        writer.writerow({name: selected_row.get(name, "") for name in SELECTED_COLUMNS})

    with (patch.object(evaluate, "_set_tracking_uri", return_value=FakeMlflow()),):
        with pytest.raises(
            EvaluationFailuresError, match="test evaluation failed for 1 model"
        ):
            run_test(evaluation_args(results_dir))

    with (results_dir / "test_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "FAILED"
    assert "FileNotFoundError" in rows[0]["error"]
    assert "missing-selected-model.pt" in rows[0]["error"]


def test_validation_uses_mlflow_metrics_instead_of_csv_checkpoint_status(
    tmp_path: Path,
):
    first_model_path = tmp_path / "finished.pt"
    retry_model_path = tmp_path / "retry.pt"
    first_model_path.write_bytes(b"model")
    retry_model_path.write_bytes(b"model")
    finished_candidate = replace(
        regression_candidate(first_model_path),
        run_id="finished-source-run",
        batch_size=8,
    )
    retry_candidate = replace(
        regression_candidate(retry_model_path),
        run_id="retry-source-run",
        batch_size=16,
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    validation_path = results_dir / "validation_metrics.csv"
    existing_rows = [
        {
            "evaluation_key": finished_candidate.evaluation_key,
            "status": "FINISHED",
            "family": finished_candidate.family,
            "task_type": finished_candidate.task_type,
            "target": finished_candidate.target,
            "split_type": finished_candidate.split_type,
            "training_seed": finished_candidate.training_seed,
            "batch_size": finished_candidate.batch_size,
            "run_id": finished_candidate.run_id,
            "model_path": str(finished_candidate.model_path),
            "val_rmse": "0.1",
        },
        {
            "evaluation_key": retry_candidate.evaluation_key,
            "status": "FAILED",
            "error": "previous metadata failure",
            "family": retry_candidate.family,
            "task_type": retry_candidate.task_type,
            "target": retry_candidate.target,
            "split_type": retry_candidate.split_type,
            "training_seed": retry_candidate.training_seed,
            "batch_size": retry_candidate.batch_size,
            "run_id": retry_candidate.run_id,
            "model_path": str(retry_candidate.model_path),
        },
    ]
    with validation_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({name: row.get(name, "") for name in RESULT_COLUMNS})

    dataset_path = tmp_path / "validation.csv"
    dataset_path.write_text(
        "smiles,FreeSolv\nretry-1,2.0\nretry-2,4.0\n", encoding="utf-8"
    )
    frame = pd.DataFrame(
        {"smiles": ["retry-1", "retry-2"], "FreeSolv": [2.0, 4.0]},
        index=[5, 6],
    )
    predicted_source_run_ids = []
    fake_mlflow = FakeMlflow(
        [
            FakeRun(
                "finished-source-run",
                metrics={
                    "val_rmse": 0.1,
                    "val_mse": 0.01,
                    "val_mae": 0.1,
                    "val_r2": 0.9,
                },
            ),
            FakeRun("retry-source-run"),
        ]
    )

    def fake_predict(current_candidate, subset, current_results_dir):
        predicted_source_run_ids.append(current_candidate.run_id)
        assert subset == "val"
        return (
            frame,
            np.array([2.0, 4.0]),
            np.array([2.5, 3.5]),
            dataset_path,
            "retry-fingerprint",
            0.1,
        )

    with (
        patch.object(evaluate, "_set_tracking_uri", return_value=fake_mlflow),
        patch.object(evaluate, "predict_candidate", side_effect=fake_predict),
    ):
        processed = run_validation(
            evaluation_args(results_dir), [finished_candidate, retry_candidate]
        )

    assert predicted_source_run_ids == ["retry-source-run"]
    assert len(processed) == 2
    assert all(row["status"] == "FINISHED" for row in processed)
    with validation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows_by_key = {row["evaluation_key"]: row for row in csv.DictReader(handle)}
    assert rows_by_key[finished_candidate.evaluation_key]["val_rmse"] == "0.1"
    assert rows_by_key[retry_candidate.evaluation_key]["status"] == "FINISHED"
    assert rows_by_key[retry_candidate.evaluation_key]["val_rmse"] == "0.5"


def test_validation_reuses_training_metrics_without_prediction(tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model = regression_candidate(model_path)
    run = FakeRun(
        model.run_id,
        metrics={
            "val_rmse": 0.4,
            "val_mse": 0.16,
            "val_mae": 0.3,
            "val_r2": 0.8,
            "val_inference_time_sec": 0.2,
        },
    )
    fake_mlflow = FakeMlflow([run])

    with (
        patch.object(evaluate, "_set_tracking_uri", return_value=fake_mlflow),
        patch.object(evaluate, "predict_candidate") as predict_mock,
    ):
        rows = run_validation(evaluation_args(tmp_path / "results"), [model])

    predict_mock.assert_not_called()
    assert rows[0]["val_rmse"] == 0.4
    assert rows[0]["validation_metric_source"] == "training_run"
    assert fake_mlflow.logged_metric_dicts == []


def test_validation_preserves_existing_metric_and_logs_only_missing(tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model = regression_candidate(model_path)
    fake_mlflow = FakeMlflow([FakeRun(model.run_id, metrics={"val_rmse": 9.0})])
    dataset_path = tmp_path / "validation.csv"
    dataset_path.write_text("smiles,FreeSolv\na,1\nb,3\n", encoding="utf-8")
    frame = pd.DataFrame({"smiles": ["a", "b"], "FreeSolv": [1.0, 3.0]}, index=[1, 2])
    prediction = (
        frame,
        np.array([1.0, 3.0]),
        np.array([1.5, 2.5]),
        dataset_path,
        "fingerprint",
        0.25,
    )

    with (
        patch.object(evaluate, "_set_tracking_uri", return_value=fake_mlflow),
        patch.object(evaluate, "predict_candidate", return_value=prediction),
    ):
        rows = run_validation(evaluation_args(tmp_path / "results"), [model])

    assert rows[0]["val_rmse"] == 9.0
    assert rows[0]["validation_metric_source"] == "mixed"
    assert "val_rmse" not in fake_mlflow.logged_metric_dicts[0]
    assert set(fake_mlflow.logged_metric_dicts[0]) == {
        "val_mse",
        "val_mae",
        "val_r2",
        "val_inference_time_sec",
    }


def test_selection_stops_when_training_run_is_missing(tmp_path: Path):
    rows = [make_result_row(size, 0.5) for size in (8, 16, 32, 64)]
    validation_path = tmp_path / "validation_metrics.csv"
    with validation_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in RESULT_COLUMNS})

    with (
        patch.object(evaluate, "_set_tracking_uri", return_value=FakeMlflow()),
        pytest.raises(ValueError, match="MLflow run validation failed"),
    ):
        run_selection(evaluation_args(tmp_path))

    assert not (tmp_path / "selected_models.csv").exists()
