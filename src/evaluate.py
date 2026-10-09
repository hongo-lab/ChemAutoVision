"""Select saved models by validation metrics and test only the winners.

Training runs in MLflow are the source of truth.  Existing ``val_*`` metrics
are preserved; when they are missing, this program calculates only the missing
validation metrics and adds them to the same training run.  It then selects the
best batch size per model/target/split/training-seed group and adds ``test_*``
metrics only to the selected training runs.  No evaluation-only MLflow runs are
created.

Examples (run from ``src`` or the repository root)::

    python evaluate.py dry-run
    python evaluate.py validate --gpu 0
    python evaluate.py select
    python evaluate.py test --gpu 0
    python evaluate.py all --gpu 0

The CSV files under ``results/evaluation`` are reports.  MLflow remains the
source of truth when deciding whether a metric already exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from settings import IMG_SIZE
from utils.split import BALANCED_SCAFFOLD_SPLIT_TYPE, make_split_prefix


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_FAMILIES = ("chem_autovision", "resnet18_aug", "dmpnn")
CLASSIFICATION_TARGETS = frozenset(("BBBP", "P-gp", "CYP3A4", "hERG"))
REGRESSION_TARGETS = frozenset(("FreeSolv", "ESOL", "Lipo"))
EXPECTED_BATCH_SIZES = frozenset((8, 16, 32, 64))
EXPECTED_TRAINING_SEEDS = frozenset((10, 42, 99))
EVALUATION_VERSION = "1"
SOURCE_CLASSIFICATION_EXP_ID = "645489900222138469"
SOURCE_REGRESSION_EXP_ID = "346038581243336323"

MODEL_TYPES = {
    "chem_autovision": "ChemAutoVision",
    "resnet18_aug": "ResNet18",
    "dmpnn": "D-MPNN",
}


class EvaluationFailuresError(RuntimeError):
    """Raised after a stage records one or more candidate failures."""


CLASSIFICATION_METRIC_NAMES = (
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
)
REGRESSION_METRIC_NAMES = ("rmse", "mse", "mae", "r2")

RESULT_COLUMNS = (
    "evaluation_key",
    "run_id",
    "status",
    "error",
    "family",
    "model_type",
    "task_type",
    "target",
    "split_type",
    "split_seed",
    "training_seed",
    "batch_size",
    "model_path",
    "dataset_path",
    "dataset_sha256",
    "sample_count",
    "data_augmentation",
    "prediction_path",
    "val_roc_auc",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_mcc",
    "val_rmse",
    "val_mse",
    "val_mae",
    "val_r2",
    "val_inference_time_sec",
    "validation_metric_source",
)

TEST_RESULT_COLUMNS = (
    "evaluation_key",
    "run_id",
    "status",
    "error",
    "family",
    "model_type",
    "task_type",
    "target",
    "split_type",
    "split_seed",
    "training_seed",
    "batch_size",
    "model_path",
    "dataset_path",
    "dataset_sha256",
    "sample_count",
    "data_augmentation",
    "prediction_path",
    "test_roc_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "test_rmse",
    "test_mse",
    "test_mae",
    "test_r2",
    "test_inference_time_sec",
)

SELECTED_COLUMNS = (
    "selection_group",
    "selection_metric",
    "selection_metric_value",
    "selection_rank",
    *RESULT_COLUMNS,
)


@dataclass(frozen=True)
class Candidate:
    family: str
    target: str
    task_type: str
    split_type: str
    split_seed: int | None
    training_seed: int
    batch_size: int
    run_id: str
    model_path: Path

    @property
    def model_type(self) -> str:
        return MODEL_TYPES[self.family]

    @property
    def evaluation_key(self) -> str:
        parts = (
            self.family,
            self.target,
            self.split_type,
            str(self.split_seed or "none"),
            str(self.training_seed),
            str(self.batch_size),
            self.run_id,
        )
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]

    @property
    def selection_group(self) -> str:
        return "|".join(
            (
                self.family,
                self.target,
                self.split_type,
                str(self.split_seed or "none"),
                str(self.training_seed),
            )
        )


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "none", "null", "nan"} else text


def _optional_int(value: object) -> int | None:
    text = _clean(value)
    return None if not text else int(float(text))


def task_type_for_target(target: str) -> str:
    if target in CLASSIFICATION_TARGETS:
        return "classification"
    if target in REGRESSION_TARGETS:
        return "regression"
    raise ValueError(f"unsupported target: {target}")


def normalize_split_type(value: object) -> str:
    """Normalize the legacy ``scaffold`` label for public output."""

    text = _clean(value)
    if text == "random":
        return text
    if text in {"scaffold", BALANCED_SCAFFOLD_SPLIT_TYPE}:
        return BALANCED_SCAFFOLD_SPLIT_TYPE
    raise ValueError(f"unsupported split type: {text or '<empty>'}")


def resolve_candidate_model_path(row: Mapping[str, str], repo_root: Path) -> Path:
    """Resolve model paths recorded on either the current or another host."""

    raw_values = (
        row.get("resolved_model_path"),
        row.get("artifact_path_from_workspace"),
        row.get("model_path_from_csv"),
    )
    attempts: list[Path] = []
    for raw_value in raw_values:
        raw = _clean(raw_value)
        if not raw:
            continue
        path = Path(raw)
        if path.is_absolute():
            attempts.append(path)
        else:
            attempts.extend(
                (repo_root / raw, repo_root.parent / raw, repo_root / "src" / raw)
            )

        # MLflow stores paths exactly as seen by the training process.  A run
        # copied from a server may therefore contain /models/... or a different
        # absolute ChemAutoVision path.  Re-anchor only the known artifact
        # roots, preserving the suffix beneath them.
        parts = tuple(part for part in raw.replace("\\", "/").split("/") if part)
        lowered = tuple(part.casefold() for part in parts)
        for anchor, local_root in (
            ("models", repo_root / "models"),
            ("graph_hyperopt", repo_root / "src" / "graph_hyperopt"),
        ):
            if anchor in lowered:
                index = len(lowered) - 1 - lowered[::-1].index(anchor)
                suffix = parts[index + 1 :]
                if suffix:
                    attempts.append(local_root.joinpath(*suffix))

    for path in attempts:
        if path.is_file() and path.stat().st_size > 0:
            return path.resolve()
    return attempts[0].resolve() if attempts else Path("")


def _as_bool(value: object) -> bool | None:
    text = _clean(value).casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _candidate_from_mlflow_run(
    run: Any, repo_root: Path = REPO_ROOT
) -> Candidate | None:
    """Convert an in-scope FINISHED training run into an evaluation candidate."""

    if _clean(getattr(run.info, "status", "")).upper() != "FINISHED":
        return None
    params = dict(run.data.params)
    tags = dict(run.data.tags)
    model_name = _clean(tags.get("model_name"))
    if model_name == "autokeras":
        family = "chem_autovision"
    elif model_name == "resnet18" and _as_bool(tags.get("execute_data_aug")) is True:
        family = "resnet18_aug"
    elif model_name == "chemprops":
        family = "dmpnn"
    else:
        return None

    target = _clean(tags.get("target"))
    if target not in CLASSIFICATION_TARGETS | REGRESSION_TARGETS:
        return None
    task_type = task_type_for_target(target)
    if task_type == "classification" and not _clean(params.get("class_weight")):
        return None

    raw_split = (
        params.get("split_type") if family == "dmpnn" else tags.get("data_split")
    )
    if not _clean(raw_split) and family != "dmpnn":
        raw_split = "random"
    try:
        split_type = normalize_split_type(raw_split)
    except ValueError:
        return None
    split_seed = _optional_int(
        params.get("split_seed") if family == "dmpnn" else tags.get("split_seed")
    )
    if split_type == BALANCED_SCAFFOLD_SPLIT_TYPE and split_seed != 42:
        return None

    training_seed = _optional_int(params.get("seed"))
    batch_size = _optional_int(params.get("batch_size"))
    if (
        training_seed not in EXPECTED_TRAINING_SEEDS
        or batch_size not in EXPECTED_BATCH_SIZES
    ):
        return None

    model_path = resolve_candidate_model_path(
        {"model_path_from_csv": _clean(tags.get("model_path"))}, repo_root
    )
    return Candidate(
        family=family,
        target=target,
        task_type=task_type,
        split_type=split_type,
        split_seed=split_seed,
        training_seed=training_seed,
        batch_size=batch_size,
        run_id=str(run.info.run_id),
        model_path=model_path,
    )


def discover_candidates(args: argparse.Namespace) -> list[Candidate]:
    """Discover and de-duplicate candidate models directly from MLflow."""

    mlflow = _set_tracking_uri(args.tracking_uri)
    experiment_ids = (
        args.source_classification_experiment_id,
        args.source_regression_experiment_id,
    )
    missing_experiments = [
        experiment_id
        for experiment_id in experiment_ids
        if mlflow.get_experiment(experiment_id) is None
    ]
    if missing_experiments:
        raise ValueError(
            "source MLflow experiment(s) not found: " + ", ".join(missing_experiments)
        )
    runs = mlflow.search_runs(
        experiment_ids=list(experiment_ids),
        output_format="list",
        order_by=["attributes.start_time DESC"],
    )
    grouped: dict[tuple[object, ...], list[Candidate]] = {}
    expected_task_by_experiment = {
        str(args.source_classification_experiment_id): "classification",
        str(args.source_regression_experiment_id): "regression",
    }
    for run in runs:
        candidate = _candidate_from_mlflow_run(run)
        if candidate is None:
            continue
        expected_task = expected_task_by_experiment.get(str(run.info.experiment_id))
        if expected_task is None or candidate.task_type != expected_task:
            raise ValueError(
                f"MLflow run {run.info.run_id} has task type {candidate.task_type} "
                f"but belongs to experiment {run.info.experiment_id}"
            )
        if args.family and candidate.family not in args.family:
            continue
        if args.target and candidate.target not in args.target:
            continue
        if args.split_type and candidate.split_type not in args.split_type:
            continue
        if args.training_seed and candidate.training_seed not in args.training_seed:
            continue
        key = (
            candidate.family,
            candidate.target,
            candidate.split_type,
            candidate.split_seed,
            candidate.training_seed,
            candidate.batch_size,
        )
        grouped.setdefault(key, []).append(candidate)

    candidates = [
        next((item for item in items if item.model_path.is_file()), items[0])
        for items in grouped.values()
    ]
    family_order = {"dmpnn": 0, "chem_autovision": 1, "resnet18_aug": 2}
    return sorted(
        candidates,
        key=lambda item: (
            family_order[item.family],
            item.target,
            item.split_type,
            item.split_seed or -1,
            item.training_seed,
            item.batch_size,
        ),
    )


def data_augmentation_for(
    candidate: Candidate, source_tags: Mapping[str, str] | None = None
) -> str:
    if candidate.family == "dmpnn":
        return "not_applicable"
    raw = _clean((source_tags or {}).get("execute_data_aug")).casefold()
    if raw in {"true", "1", "yes"}:
        return "enabled"
    if raw in {"false", "0", "no"}:
        return "disabled"
    return "unknown"


def data_augmentation_source_for(
    candidate: Candidate, source_tags: Mapping[str, str] | None = None
) -> str:
    raw = _clean((source_tags or {}).get("execute_data_aug")).casefold()
    if raw in {"true", "1", "yes", "false", "0", "no"}:
        return "source_metadata"
    if candidate.family == "dmpnn":
        return "not_applicable"
    return "missing"


def calculate_metrics(
    task_type: str, y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(y_true) != len(y_score):
        raise ValueError(
            f"prediction count mismatch: y_true={len(y_true)}, y_score={len(y_score)}"
        )
    if not np.isfinite(y_score).all():
        raise ValueError("predictions contain NaN or infinity")

    if task_type == "classification":
        if len(np.unique(y_true)) < 2:
            raise ValueError("ROC-AUC is undefined because the subset has one class")
        y_pred = (y_score > 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
    if task_type == "regression":
        mse = float(mean_squared_error(y_true, y_score))
        return {
            "rmse": float(np.sqrt(mse)),
            "mse": mse,
            "mae": float(mean_absolute_error(y_true, y_score)),
            "r2": float(r2_score(y_true, y_score)),
        }
    raise ValueError(f"unsupported task type: {task_type}")


def select_best_rows(
    rows: Iterable[Mapping[str, str]], allow_incomplete_groups: bool = False
) -> tuple[list[dict[str, str]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for source_row in rows:
        row = dict(source_row)
        group = "|".join(
            str(row.get(name, ""))
            for name in (
                "family",
                "target",
                "split_type",
                "split_seed",
                "training_seed",
            )
        )
        grouped.setdefault(group, [])
        if row.get("status") == "FINISHED":
            grouped[group].append(row)

    selected: list[dict[str, str]] = []
    ranks: dict[str, int] = {}
    incomplete: list[str] = []
    for group, group_rows in sorted(grouped.items()):
        batch_sizes = {int(row["batch_size"]) for row in group_rows}
        if batch_sizes != EXPECTED_BATCH_SIZES:
            incomplete.append(
                f"{group}: expected {sorted(EXPECTED_BATCH_SIZES)}, got {sorted(batch_sizes)}"
            )
            if not allow_incomplete_groups:
                continue
        if not group_rows:
            # ``allow_incomplete_groups`` is useful for development subsets,
            # but cannot choose a winner when every candidate failed.
            continue
        task_type = group_rows[0]["task_type"]
        metric = "val_roc_auc" if task_type == "classification" else "val_rmse"
        reverse = task_type == "classification"
        ranked = sorted(
            group_rows,
            key=lambda row: (
                -float(row[metric]) if reverse else float(row[metric]),
                int(row["batch_size"]),
                row["evaluation_key"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            ranks[row["evaluation_key"]] = rank
        winner = dict(ranked[0])
        winner.update(
            {
                "selection_group": group,
                "selection_metric": metric,
                "selection_metric_value": winner[metric],
                "selection_rank": "1",
            }
        )
        selected.append(winner)

    if incomplete and not allow_incomplete_groups:
        details = "\n".join(incomplete[:20])
        suffix = (
            "" if len(incomplete) <= 20 else f"\n... and {len(incomplete) - 20} more"
        )
        raise ValueError(
            "selection aborted because some groups do not have all four batch sizes:\n"
            + details
            + suffix
        )
    return selected, ranks


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subset_sha256(frame: pd.DataFrame, target: str) -> str:
    digest = hashlib.sha256()
    for index, row in frame.iterrows():
        payload = json.dumps(
            [str(index), str(row.get("smiles", "")), str(row[target])],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dataset_pickle_path(candidate: Candidate) -> Path:
    prefix = make_split_prefix(candidate.split_type, candidate.split_seed)
    return REPO_ROOT / "data" / f"{prefix}{candidate.target}_img.pkl"


def _load_subset_frame(candidate: Candidate, subset: str) -> tuple[pd.DataFrame, Path]:
    path = _dataset_pickle_path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    frame = pd.read_pickle(path)
    required = {"group", "smiles", candidate.target}
    if candidate.family != "dmpnn":
        required.add("img_path")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    frame = frame.loc[frame["group"] == subset].copy()
    if candidate.task_type == "classification":
        frame = frame.loc[frame[candidate.target] != -1].copy()
    if frame.empty:
        raise ValueError(f"{subset} subset is empty: {path}")
    return frame, path


def _configure_tensorflow_gpu_memory_growth(tf: Any) -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _predict_image_model(
    candidate: Candidate, subset: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Path, str, float]:
    import tensorflow as tf

    _configure_tensorflow_gpu_memory_growth(tf)
    frame, dataset_path = _load_subset_frame(candidate, subset)

    def resolve_image_path(raw_value: object) -> str:
        raw_path = Path(str(raw_value))
        attempts = (
            (raw_path,)
            if raw_path.is_absolute()
            else (
                REPO_ROOT / "src" / raw_path,
                REPO_ROOT / raw_path,
                dataset_path.parent / raw_path,
            )
        )
        for attempt in attempts:
            if attempt.is_file():
                return str(attempt.resolve())
        raise FileNotFoundError(f"image not found: {raw_value}")

    paths = frame["img_path"].map(resolve_image_path).to_numpy()
    labels = frame[candidate.target].to_numpy()

    def load_image(path: Any, label: Any) -> tuple[Any, Any]:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
        return tf.cast(image, tf.float32) / 255.0, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(candidate.batch_size).prefetch(tf.data.AUTOTUNE)

    if candidate.family == "chem_autovision":
        import autokeras as ak

        model = tf.keras.models.load_model(
            candidate.model_path,
            custom_objects=ak.CUSTOM_OBJECTS,
            compile=False,
        )
    else:
        model = tf.keras.models.load_model(candidate.model_path, compile=False)

    started = time.perf_counter()
    predictions = np.asarray(model.predict(dataset, verbose=0)).reshape(-1)
    elapsed = time.perf_counter() - started
    tf.keras.backend.clear_session()
    return (
        frame,
        np.asarray(labels).reshape(-1),
        predictions,
        dataset_path,
        _subset_sha256(frame, candidate.target),
        elapsed,
    )


def _prepared_dmpnn_csv(
    candidate: Candidate, subset: str, results_dir: Path
) -> tuple[Path, pd.DataFrame]:
    frame, source_path = _load_subset_frame(candidate, subset)
    prepared_dir = results_dir / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output_path = prepared_dir / f"{candidate.evaluation_key}_{subset}.csv"
    frame.loc[:, ["smiles", candidate.target]].to_csv(output_path, index=False)
    return output_path, frame


def _predict_dmpnn(
    candidate: Candidate, subset: str, results_dir: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Path, str, float]:
    from chemprop.args import PredictArgs
    from chemprop.train import make_predictions
    from training.chemprop_checkpoint_compat import trusted_chemprop_checkpoint_loading

    data_path, frame = _prepared_dmpnn_csv(candidate, subset, results_dir)
    raw_prediction_dir = results_dir / "raw_dmpnn_predictions"
    raw_prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = raw_prediction_dir / f"{candidate.evaluation_key}_{subset}.csv"
    args = PredictArgs().parse_args(
        [
            "--test_path",
            str(data_path),
            "--checkpoint_path",
            str(candidate.model_path),
            "--preds_path",
            str(prediction_path),
            "--smiles_column",
            "smiles",
        ]
    )
    started = time.perf_counter()
    with trusted_chemprop_checkpoint_loading():
        make_predictions(args)
    elapsed = time.perf_counter() - started
    prediction_frame = pd.read_csv(prediction_path)
    if candidate.target not in prediction_frame.columns:
        raise ValueError(
            f"D-MPNN prediction has no '{candidate.target}' column: {prediction_path}"
        )
    return (
        frame,
        frame[candidate.target].to_numpy(),
        prediction_frame[candidate.target].to_numpy(),
        data_path,
        _subset_sha256(frame, candidate.target),
        elapsed,
    )


def predict_candidate(
    candidate: Candidate, subset: str, results_dir: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Path, str, float]:
    if candidate.family == "dmpnn":
        return _predict_dmpnn(candidate, subset, results_dir)
    return _predict_image_model(candidate, subset)


def _write_prediction_csv(
    path: Path,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            "row_id": frame.index.astype(str),
            "smiles": frame["smiles"].astype(str).to_numpy(),
            "y_true": np.asarray(y_true).reshape(-1),
            "y_score": np.asarray(y_score).reshape(-1),
        }
    )
    if task_type == "classification":
        output["y_pred"] = (output["y_score"] > 0.5).astype(int)
    output.to_csv(path, index=False)


def _write_roc_curve(path: Path, y_true: np.ndarray, y_score: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], "k--")
    axis.plot(false_positive_rate, true_positive_rate)
    axis.set(
        xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve"
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_csv_row(
    path: Path, columns: Sequence[str], row: Mapping[str, object]
) -> None:
    """Upsert a checkpoint row, keeping one latest row per evaluation key."""

    existing = _read_csv_rows(path)
    key = str(row.get("evaluation_key", ""))
    retained = [item for item in existing if item.get("evaluation_key") != key]
    retained.append(dict(row))
    _write_csv_rows(path, columns, retained)


def _write_csv_rows(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    temporary.replace(path)


def _gpu_names() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
        )
        return ", ".join(
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _set_tracking_uri(tracking_uri: str) -> Any:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def _metric_names(task_type: str, subset: str) -> tuple[str, ...]:
    base = (
        CLASSIFICATION_METRIC_NAMES
        if task_type == "classification"
        else REGRESSION_METRIC_NAMES
    )
    return tuple(f"{subset}_{name}" for name in base)


def _get_finished_run(mlflow: Any, run_id: str) -> Any:
    if not run_id:
        raise ValueError("MLflow run_id is empty")
    try:
        run = mlflow.get_run(run_id)
    except Exception as exc:
        raise ValueError(f"MLflow run not found: {run_id}") from exc
    if _clean(getattr(run.info, "status", "")).upper() != "FINISHED":
        raise ValueError(
            f"MLflow run is not FINISHED: {run_id} "
            f"({getattr(run.info, 'status', '<unknown>')})"
        )
    return run


def _require_finished_runs(mlflow: Any, run_ids: Iterable[str]) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    errors: list[str] = []
    for run_id in sorted(set(run_ids)):
        try:
            runs[run_id] = _get_finished_run(mlflow, run_id)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("MLflow run validation failed:\n- " + "\n- ".join(errors))
    return runs


def _nonempty(values: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in values.items() if _clean(value)}


def run_validation(
    args: argparse.Namespace, candidates: Sequence[Candidate]
) -> list[dict[str, object]]:
    mlflow = _set_tracking_uri(args.tracking_uri)
    source_runs = _require_finished_runs(mlflow, (item.run_id for item in candidates))
    output_path = args.results_dir / "validation_metrics.csv"
    all_rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[{index}/{len(candidates)}] validation {candidate.model_type} "
            f"{candidate.target} {candidate.split_type} seed={candidate.training_seed} "
            f"batch={candidate.batch_size}"
        )
        base: dict[str, object] = {
            "evaluation_key": candidate.evaluation_key,
            "status": "FAILED",
            "error": "",
            "family": candidate.family,
            "model_type": candidate.model_type,
            "task_type": candidate.task_type,
            "target": candidate.target,
            "split_type": candidate.split_type,
            "split_seed": candidate.split_seed or "",
            "training_seed": candidate.training_seed,
            "batch_size": candidate.batch_size,
            "run_id": candidate.run_id,
            "model_path": str(candidate.model_path),
        }
        try:
            if not candidate.model_path.is_file():
                raise FileNotFoundError(f"model not found: {candidate.model_path}")
            source_run = source_runs[candidate.run_id]
            source_tags = dict(source_run.data.tags)
            source_metrics = dict(source_run.data.metrics)
            augmentation = data_augmentation_for(candidate, source_tags)
            if augmentation == "unknown":
                raise ValueError(
                    "data augmentation setting is unknown; source run has no execute_data_aug tag"
                )
            required_names = _metric_names(candidate.task_type, "val")
            existing_metrics = {
                name: float(source_metrics[name])
                for name in required_names
                if name in source_metrics and np.isfinite(source_metrics[name])
            }
            missing_names = [
                name for name in required_names if name not in existing_metrics
            ]
            calculated: dict[str, float] = {}
            dataset_path: Path | str = _clean(source_tags.get("csv_path"))
            fingerprint = _clean(source_tags.get("val_smiles_hash"))
            prediction_path: Path | str = _clean(
                source_tags.get("validation_result_csv_path")
            )
            sample_count: int | str = ""
            elapsed: float | str = source_metrics.get("val_inference_time_sec", "")
            metric_source = "training_run"
            tags: dict[str, object] = {
                "evaluation_version": EVALUATION_VERSION,
                "selection_group": candidate.selection_group,
            }
            if missing_names:
                (
                    frame,
                    y_true,
                    y_score,
                    dataset_path,
                    fingerprint,
                    elapsed,
                ) = predict_candidate(candidate, "val", args.results_dir)
                calculated = {
                    f"val_{name}": value
                    for name, value in calculate_metrics(
                        candidate.task_type, y_true, y_score
                    ).items()
                }
                prediction_path = (
                    args.results_dir
                    / "predictions"
                    / "validation"
                    / f"{candidate.evaluation_key}.csv"
                )
                _write_prediction_csv(
                    prediction_path, frame, y_true, y_score, candidate.task_type
                )
                if candidate.task_type == "classification":
                    roc_path = (
                        args.results_dir
                        / "plots"
                        / "validation"
                        / f"{candidate.evaluation_key}_roc.png"
                    )
                    _write_roc_curve(roc_path, y_true, y_score)
                    tags["validation_roc_path"] = str(roc_path)
                sample_count = len(y_true)
                to_log = {name: calculated[name] for name in missing_names}
                if "val_inference_time_sec" not in source_metrics:
                    to_log["val_inference_time_sec"] = float(elapsed)
                tags.update(
                    {
                        "validation_metric_source": (
                            "mixed" if existing_metrics else "evaluate.py"
                        ),
                        "validation_evaluated_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "val_dataset_path": str(dataset_path),
                        "val_dataset_sha256": fingerprint,
                        "val_prediction_path": str(prediction_path),
                        "val_sample_count": sample_count,
                        "evaluation_gpu_names": _gpu_names(),
                    }
                )
                with mlflow.start_run(run_id=candidate.run_id):
                    mlflow.log_metrics(to_log)
                    mlflow.set_tags(_nonempty(tags))
                metric_source = "mixed" if existing_metrics else "evaluate.py"
            else:
                tags["validation_metric_source"] = "training_run"
                with mlflow.start_run(run_id=candidate.run_id):
                    mlflow.set_tags(_nonempty(tags))
            final_metrics = {
                name: (
                    existing_metrics[name]
                    if name in existing_metrics
                    else calculated[name]
                )
                for name in required_names
            }
            base.update(
                {
                    "status": "FINISHED",
                    "dataset_path": str(dataset_path),
                    "dataset_sha256": fingerprint,
                    "sample_count": sample_count,
                    "data_augmentation": augmentation,
                    "prediction_path": str(prediction_path),
                    "val_inference_time_sec": elapsed,
                    "validation_metric_source": metric_source,
                    **final_metrics,
                }
            )
        except Exception as exc:
            base["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        _append_csv_row(output_path, RESULT_COLUMNS, base)
        all_rows.append(base)
    failed = [row for row in all_rows if row.get("status") == "FAILED"]
    if failed:
        raise EvaluationFailuresError(
            f"validation failed for {len(failed)} model(s); details: {output_path}"
        )
    return all_rows


def run_selection(args: argparse.Namespace) -> list[dict[str, str]]:
    validation_path = args.results_dir / "validation_metrics.csv"
    rows = [
        row
        for row in _read_csv_rows(validation_path)
        if _result_row_matches_filters(row, args)
    ]
    if not rows:
        raise FileNotFoundError(f"validation results not found: {validation_path}")
    mlflow = _set_tracking_uri(args.tracking_uri)
    _require_finished_runs(mlflow, (row.get("run_id", "") for row in rows))
    selected, ranks = select_best_rows(rows)
    selected_path = args.results_dir / "selected_models.csv"
    if any((args.family, args.target, args.split_type, args.training_seed)):
        selected_groups = {row["selection_group"] for row in selected}
        retained = [
            row
            for row in _read_csv_rows(selected_path)
            if row.get("selection_group") not in selected_groups
        ]
        output_rows = retained + selected
    else:
        output_rows = selected
    _write_csv_rows(selected_path, SELECTED_COLUMNS, output_rows)

    for row in rows:
        run_id = row.get("run_id", "")
        evaluation_key = row.get("evaluation_key", "")
        if not run_id or evaluation_key not in ranks:
            continue
        rank = ranks[evaluation_key]
        with mlflow.start_run(run_id=run_id):
            metric_name = (
                "val_roc_auc" if row["task_type"] == "classification" else "val_rmse"
            )
            mlflow.set_tags(
                {
                    "selection_rank": rank,
                    "selected_for_test": str(rank == 1).lower(),
                    "selection_metric": metric_name,
                    "selection_metric_value": row[metric_name],
                    "selection_evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    print(
        f"selected {len(selected)} models in this invocation "
        f"({len(output_rows)} total) -> {selected_path}"
    )
    return selected


def _candidate_from_result_row(row: Mapping[str, str]) -> Candidate:
    return Candidate(
        family=row["family"],
        target=row["target"],
        task_type=row["task_type"],
        split_type=row["split_type"],
        split_seed=_optional_int(row.get("split_seed")),
        training_seed=int(row["training_seed"]),
        batch_size=int(row["batch_size"]),
        run_id=row.get("run_id", ""),
        model_path=Path(row["model_path"]),
    )


def _result_row_matches_filters(
    row: Mapping[str, str], args: argparse.Namespace
) -> bool:
    return not (
        (args.family and row.get("family") not in args.family)
        or (args.target and row.get("target") not in args.target)
        or (args.split_type and row.get("split_type") not in args.split_type)
        or (
            args.training_seed
            and _optional_int(row.get("training_seed")) not in args.training_seed
        )
    )


def run_test(args: argparse.Namespace) -> list[dict[str, object]]:
    selected_path = args.results_dir / "selected_models.csv"
    selected_rows = [
        row
        for row in _read_csv_rows(selected_path)
        if _result_row_matches_filters(row, args)
    ]
    if not selected_rows:
        raise FileNotFoundError(f"selected models not found: {selected_path}")
    output_path = args.results_dir / "test_metrics.csv"
    result_rows: list[dict[str, object]] = []
    mlflow = _set_tracking_uri(args.tracking_uri)
    source_runs = _require_finished_runs(
        mlflow, (row.get("run_id", "") for row in selected_rows)
    )
    for index, selected in enumerate(selected_rows, start=1):
        candidate = _candidate_from_result_row(selected)
        print(
            f"[{index}/{len(selected_rows)}] test {candidate.model_type} "
            f"{candidate.target} {candidate.split_type} seed={candidate.training_seed} "
            f"batch={candidate.batch_size}"
        )
        base: dict[str, object] = {
            name: selected.get(name, "")
            for name in TEST_RESULT_COLUMNS
            if name in selected
        }
        base.update({"status": "FAILED", "error": ""})
        try:
            if not candidate.model_path.is_file():
                raise FileNotFoundError(f"model not found: {candidate.model_path}")
            source_run = source_runs[candidate.run_id]
            source_metrics = dict(source_run.data.metrics)
            source_tags = dict(source_run.data.tags)
            required_names = _metric_names(candidate.task_type, "test")
            existing_metrics = {
                name: float(source_metrics[name])
                for name in required_names
                if name in source_metrics and np.isfinite(source_metrics[name])
            }
            if len(existing_metrics) == len(required_names) and not args.overwrite:
                base.update(
                    {
                        "status": "FINISHED",
                        "dataset_path": _clean(source_tags.get("test_dataset_path")),
                        "dataset_sha256": _clean(
                            source_tags.get("test_dataset_sha256")
                        ),
                        "sample_count": _clean(source_tags.get("test_sample_count")),
                        "prediction_path": _clean(
                            source_tags.get("test_prediction_path")
                        ),
                        "test_inference_time_sec": source_metrics.get(
                            "test_inference_time_sec", ""
                        ),
                        **existing_metrics,
                    }
                )
                _append_csv_row(output_path, TEST_RESULT_COLUMNS, base)
                result_rows.append(base)
                print(f"[{index}/{len(selected_rows)}] SKIP existing test metrics")
                continue
            (
                frame,
                y_true,
                y_score,
                dataset_path,
                fingerprint,
                elapsed,
            ) = predict_candidate(candidate, "test", args.results_dir)
            calculated = {
                f"test_{key}": value
                for key, value in calculate_metrics(
                    candidate.task_type, y_true, y_score
                ).items()
            }
            prediction_path = (
                args.results_dir
                / "predictions"
                / "test"
                / f"{candidate.evaluation_key}.csv"
            )
            _write_prediction_csv(
                prediction_path, frame, y_true, y_score, candidate.task_type
            )
            roc_path: Path | None = None
            if candidate.task_type == "classification":
                roc_path = (
                    args.results_dir
                    / "plots"
                    / "test"
                    / f"{candidate.evaluation_key}_roc.png"
                )
                _write_roc_curve(roc_path, y_true, y_score)
            to_log = (
                calculated
                if args.overwrite
                else {
                    name: calculated[name]
                    for name in required_names
                    if name not in existing_metrics
                }
            )
            if args.overwrite or "test_inference_time_sec" not in source_metrics:
                to_log["test_inference_time_sec"] = elapsed
            with mlflow.start_run(run_id=candidate.run_id):
                mlflow.log_metrics(to_log)
                mlflow.set_tags(
                    {
                        "selected_for_test": "true",
                        "test_metric_source": "evaluate.py",
                        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
                        "test_dataset_path": str(dataset_path),
                        "test_dataset_sha256": fingerprint,
                        "test_sample_count": len(y_true),
                        "test_prediction_path": str(prediction_path),
                    }
                )
            base.update(
                {
                    "status": "FINISHED",
                    "dataset_path": str(dataset_path),
                    "dataset_sha256": fingerprint,
                    "sample_count": len(y_true),
                    "prediction_path": str(prediction_path),
                    "test_inference_time_sec": elapsed,
                    **{
                        name: (
                            calculated[name]
                            if args.overwrite or name not in existing_metrics
                            else existing_metrics[name]
                        )
                        for name in required_names
                    },
                }
            )
        except Exception as exc:
            base["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        _append_csv_row(output_path, TEST_RESULT_COLUMNS, base)
        result_rows.append(base)
    write_test_summary(output_path, args.results_dir / "test_metrics_summary.csv")
    failed = [row for row in result_rows if row.get("status") == "FAILED"]
    if failed:
        raise EvaluationFailuresError(
            f"test evaluation failed for {len(failed)} model(s); details: {output_path}"
        )
    return result_rows


def write_test_summary(test_metrics_path: Path, output_path: Path) -> None:
    """Write manuscript-ready mean and sample SD across training seeds."""

    rows = [
        row
        for row in _read_csv_rows(test_metrics_path)
        if row.get("status") == "FINISHED"
    ]
    summary_columns = (
        "model_type",
        "target",
        "task_type",
        "split_type",
        "data_augmentation",
        "metric",
        "n_seeds",
        "mean",
        "std",
    )
    summaries: list[dict[str, object]] = []
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(
            row.get(name, "")
            for name in (
                "model_type",
                "target",
                "task_type",
                "split_type",
                "data_augmentation",
            )
        )
        grouped.setdefault(key, []).append(row)
    for key, group_rows in sorted(grouped.items()):
        task_type = key[2]
        metric_names = (
            CLASSIFICATION_METRIC_NAMES
            if task_type == "classification"
            else REGRESSION_METRIC_NAMES
        )
        for metric_name in metric_names:
            values = np.asarray(
                [float(row[f"test_{metric_name}"]) for row in group_rows],
                dtype=float,
            )
            summaries.append(
                {
                    "model_type": key[0],
                    "target": key[1],
                    "task_type": task_type,
                    "split_type": key[3],
                    "data_augmentation": key[4],
                    "metric": f"test_{metric_name}",
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else "",
                }
            )
    _write_csv_rows(output_path, summary_columns, summaries)


def audit_candidates(
    candidates: Sequence[Candidate], args: argparse.Namespace | None = None
) -> int:
    missing = [
        candidate for candidate in candidates if not candidate.model_path.is_file()
    ]
    grouped: dict[str, set[int]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.selection_group, set()).add(candidate.batch_size)
    incomplete = {
        group: batches
        for group, batches in grouped.items()
        if batches != EXPECTED_BATCH_SIZES
    }
    missing_conditions: list[tuple[object, ...]] = []
    if args is not None:
        families = tuple(args.family or SUPPORTED_FAMILIES)
        targets = tuple(
            args.target or sorted(CLASSIFICATION_TARGETS | REGRESSION_TARGETS)
        )
        split_types = tuple(args.split_type or ("random", BALANCED_SCAFFOLD_SPLIT_TYPE))
        training_seeds = tuple(args.training_seed or sorted(EXPECTED_TRAINING_SEEDS))
        expected = {
            (family, target, split_type, seed, batch_size)
            for family in families
            for target in targets
            for split_type in split_types
            for seed in training_seeds
            for batch_size in EXPECTED_BATCH_SIZES
        }
        actual = {
            (
                item.family,
                item.target,
                item.split_type,
                item.training_seed,
                item.batch_size,
            )
            for item in candidates
        }
        missing_conditions = sorted(expected - actual)
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.family] = counts.get(candidate.family, 0) + 1
    print(f"candidates: {len(candidates)} {counts}")
    print(f"selection groups: {len(grouped)}")
    print(f"missing model files: {len(missing)}")
    for candidate in missing[:30]:
        print(
            f"  MISSING {candidate.selection_group} batch={candidate.batch_size}: {candidate.model_path}"
        )
    if len(missing) > 30:
        print(f"  ... and {len(missing) - 30} more")
    print(f"incomplete selection groups: {len(incomplete)}")
    for group, batches in list(incomplete.items())[:30]:
        print(f"  INCOMPLETE {group}: {sorted(batches)}")
    print(f"missing MLflow conditions: {len(missing_conditions)}")
    for condition in missing_conditions[:30]:
        print("  MISSING_RUN " + " / ".join(map(str, condition)))
    if len(missing_conditions) > 30:
        print(f"  ... and {len(missing_conditions) - 30} more")
    return 1 if missing or incomplete or missing_conditions else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("dry-run", "validate", "select", "test", "all"),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "evaluation"
    )
    parser.add_argument("--tracking-uri", default=str(REPO_ROOT / "mlruns"))
    parser.add_argument(
        "--source-classification-experiment-id",
        default=os.environ.get(
            "CHEMAUTOVISION_SOURCE_CLASSIFICATION_EXPERIMENT_ID",
            SOURCE_CLASSIFICATION_EXP_ID,
        ),
    )
    parser.add_argument(
        "--source-regression-experiment-id",
        default=os.environ.get(
            "CHEMAUTOVISION_SOURCE_REGRESSION_EXPERIMENT_ID",
            SOURCE_REGRESSION_EXP_ID,
        ),
    )
    parser.add_argument("--family", action="append", choices=SUPPORTED_FAMILIES)
    parser.add_argument("--target", action="append")
    parser.add_argument(
        "--split-type",
        action="append",
        choices=("random", BALANCED_SCAFFOLD_SPLIT_TYPE),
    )
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value, for example 0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    candidates = discover_candidates(args)
    if not candidates:
        raise ValueError("no candidates matched the requested filters")
    audit_status = audit_candidates(candidates, args)
    if args.command == "dry-run":
        return audit_status
    if audit_status:
        raise ValueError(
            "The MLflow candidate set is incomplete; run dry-run for details"
        )
    if args.command in {"validate", "all"}:
        run_validation(args, candidates)
    if args.command in {"select", "all"}:
        run_selection(args)
    if args.command in {"test", "all"}:
        run_test(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
