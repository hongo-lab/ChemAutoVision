"""Build paper performance figures and tables from final test metrics.

ChemAutoVision, ResNet18, D-MPNN, and ImageMol metrics are read from the
evaluation experiments written by ``evaluate.py test``. HyperoptEstimator metrics are
read from the historical flattened MLflow CSV exports; those columns have no
``test_`` prefix but already contain the final test-set measurements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (10, 42, 99)
DEFAULT_SPLITS = ("random", "balanced_scaffold")
IMAGEMOL_EVALUATION_VERSION = "imagemol-validation-selection-v2"
SUPPORTED_EVALUATION_VERSIONS = {
    "ChemAutoVision": frozenset({"1"}),
    "ResNet18": frozenset({"1"}),
    "D-MPNN": frozenset({"1"}),
    "ImageMol": frozenset({IMAGEMOL_EVALUATION_VERSION}),
}
IMAGEMOL_BATCH_SIZES = frozenset({8, 16, 32, 64})
IMAGEMOL_LEARNING_RATES = {
    "classification": frozenset({0.0005, 0.005, 0.05, 0.5}),
    "regression": frozenset({0.0005, 0.005}),
}
CLASSIFICATION_TARGETS = ("BBBP", "P-gp", "CYP3A4", "hERG")
REGRESSION_TARGETS = ("FreeSolv", "ESOL", "Lipo")
MODEL_ORDER = (
    "ResNet18",
    "ImageMol",
    "Fingerprint-based (ECFP6)",
    "Fingerprint-based (ECFP6 + descriptors)",
    "D-MPNN",
    "ChemAutoVision",
)
METRICS_BY_TASK = {
    "classification": (
        "test_roc_auc", "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_mcc",
    ),
    "regression": ("test_rmse", "test_mse", "test_mae", "test_r2"),
}
METRIC_LABELS = {
    "test_roc_auc": "ROC-AUC",
    "test_accuracy": "Acc",
    "test_precision": "Prec",
    "test_recall": "Rec",
    "test_f1": "F1",
    "test_mcc": "MCC",
    "test_rmse": "RMSE",
    "test_mse": "MSE",
    "test_mae": "MAE",
    "test_r2": r"$R^2$",
}
MINIMIZED_METRICS = {"test_rmse", "test_mse", "test_mae"}
RUN_COLUMNS = (
    "source", "source_run_id", "model_type", "task_type", "target",
    "split_type", "split_seed", "training_seed", "batch_size", "learning_rate",
    "data_augmentation", "model_path", "selection_metric",
    "selection_metric_value", *METRICS_BY_TASK["classification"],
    *METRICS_BY_TASK["regression"], "test_inference_time_sec",
)


def _empty_run_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RUN_COLUMNS)


def _numeric(value: object) -> float:
    return float(value) if value not in (None, "") else np.nan


def load_mlflow_test_runs(
    tracking_uri: str,
    experiment_ids: Sequence[str],
    *,
    supported_versions: Mapping[str, Sequence[str]] = SUPPORTED_EVALUATION_VERSIONS,
    client: Any | None = None,
) -> pd.DataFrame:
    """Load final test metrics from the supported evaluate.py run schemas."""

    if client is None:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=list(experiment_ids),
        filter_string="tags.selected_for_test = 'true'",
        max_results=1000,
    )
    records: list[dict[str, object]] = []
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        tags = run.data.tags
        model_type = tags.get("model_type", "")
        accepted_versions = {
            str(version) for version in supported_versions.get(model_type, ())
        }
        if tags.get("evaluation_version") not in accepted_versions:
            continue
        task_type = tags.get("task_type", "")
        if task_type not in METRICS_BY_TASK:
            continue
        record = {column: np.nan for column in RUN_COLUMNS}
        record.update({
            "source": "mlflow_evaluation",
            "source_run_id": run.info.run_id,
            "model_type": model_type,
            "task_type": task_type,
            "target": tags.get("target", ""),
            "split_type": tags.get("split_type", ""),
            "split_seed": _numeric(tags.get("split_seed")),
            "training_seed": _numeric(tags.get("training_seed")),
            "batch_size": _numeric(run.data.params.get("batch_size")),
            "learning_rate": _numeric(run.data.params.get("learning_rate")),
            "data_augmentation": tags.get("data_augmentation", ""),
            "model_path": tags.get("model_path", ""),
            "selection_metric": tags.get("selection_metric", ""),
            "selection_metric_value": _numeric(tags.get("selection_metric_value")),
        })
        for metric in (*METRICS_BY_TASK[task_type], "test_inference_time_sec"):
            record[metric] = _numeric(run.data.metrics.get(metric))
        if model_type == "ImageMol" and (
            np.isnan(record["batch_size"]) or np.isnan(record["learning_rate"])
        ):
            raise ValueError(
                f"ImageMol v2 run {run.info.run_id} is missing batch_size or learning_rate"
            )
        records.append(record)
    return pd.DataFrame(records, columns=RUN_COLUMNS) if records else _empty_run_frame()


def validate_imagemol_v2_candidates(
    tracking_uri: str,
    experiment_ids: Sequence[str],
    *,
    client: Any | None = None,
) -> pd.DataFrame:
    """Validate the complete ImageMol v2 learning-rate/batch-size grid."""

    if client is None:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=list(experiment_ids),
        filter_string=(
            "tags.model_type = 'ImageMol' and "
            f"tags.evaluation_version = '{IMAGEMOL_EVALUATION_VERSION}'"
        ),
        max_results=1000,
    )
    unfinished = [run.info.run_id for run in runs if run.info.status != "FINISHED"]
    if unfinished:
        raise ValueError(f"ImageMol v2 contains unfinished runs: {unfinished}")

    records = []
    for run in runs:
        tags = run.data.tags
        task_type = tags.get("task_type", "")
        if task_type not in IMAGEMOL_LEARNING_RATES:
            raise ValueError(
                f"ImageMol v2 run {run.info.run_id} has unknown task_type={task_type!r}"
            )
        batch_size = _numeric(run.data.params.get("batch_size"))
        learning_rate = _numeric(run.data.params.get("learning_rate"))
        if np.isnan(batch_size) or np.isnan(learning_rate):
            raise ValueError(
                f"ImageMol v2 run {run.info.run_id} is missing batch_size or learning_rate"
            )
        records.append({
            "run_id": run.info.run_id,
            "task_type": task_type,
            "target": tags.get("target", ""),
            "split_type": tags.get("split_type", ""),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "selected_for_test": tags.get("selected_for_test") == "true",
        })

    candidates = pd.DataFrame(records)
    if len(candidates) != 176:
        raise ValueError(
            f"Expected 176 ImageMol v2 candidate runs, found {len(candidates)}"
        )

    expected_targets = {
        "classification": set(CLASSIFICATION_TARGETS),
        "regression": set(REGRESSION_TARGETS),
    }
    expected_groups = {
        (task_type, target, split_type)
        for task_type, targets in expected_targets.items()
        for target in targets
        for split_type in DEFAULT_SPLITS
    }
    actual_groups = set(
        candidates[["task_type", "target", "split_type"]]
        .itertuples(index=False, name=None)
    )
    if actual_groups != expected_groups:
        missing = sorted(expected_groups - actual_groups)
        extra = sorted(actual_groups - expected_groups)
        raise ValueError(f"Invalid ImageMol v2 groups; missing={missing}, extra={extra}")

    summary_records = []
    group_columns = ["task_type", "target", "split_type"]
    for group_values, group in candidates.groupby(group_columns, sort=True):
        task_type = group_values[0]
        expected_pairs = {
            (batch_size, learning_rate)
            for batch_size in IMAGEMOL_BATCH_SIZES
            for learning_rate in IMAGEMOL_LEARNING_RATES[task_type]
        }
        actual_pairs = set(
            group[["batch_size", "learning_rate"]]
            .itertuples(index=False, name=None)
        )
        selected_count = int(group["selected_for_test"].sum())
        if actual_pairs != expected_pairs or len(group) != len(expected_pairs):
            raise ValueError(
                f"Incomplete ImageMol v2 candidate grid for {group_values}: "
                f"expected {len(expected_pairs)}, found {len(group)}"
            )
        if selected_count != 1:
            raise ValueError(
                f"Expected one selected ImageMol v2 run for {group_values}, "
                f"found {selected_count}"
            )
        summary_records.append({
            **dict(zip(group_columns, group_values)),
            "candidate_runs": len(group),
            "selected_runs": selected_count,
        })
    return pd.DataFrame(summary_records)


def _coalesce_numeric(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name in frame.columns:
            result = result.fillna(pd.to_numeric(frame[name], errors="coerce"))
    return result


def _load_hyperopt_file(
    path: Path,
    *,
    task_type: str,
    seeds: Sequence[int],
    max_evals: int,
    scaffold_split_seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "Start Time", "Run ID", "Status", "model_name", "target",
        "training_seed", "max_evals", "explanatory_val", "exclude_descriptors",
        "data_split", *[metric.removeprefix("test_") for metric in METRICS_BY_TASK[task_type]],
    }
    if task_type == "classification":
        required.remove("accuracy")
        required.add("acc")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing Hyperopt columns: {missing}")

    runs = frame.loc[
        frame["model_name"].eq("HyperoptEstimator")
        & frame["Status"].eq("FINISHED")
    ].copy()
    runs["training_seed"] = pd.to_numeric(runs["training_seed"], errors="coerce")
    runs["max_evals_num"] = pd.to_numeric(runs["max_evals"], errors="coerce")
    runs["split_seed_num"] = _coalesce_numeric(
        runs, ("split_seed", "split_seed.1", "dataset_split_seed")
    )
    runs["split_type"] = (
        runs["data_split"].fillna("random").astype(str).str.strip().str.lower()
        .replace({"label": "random", "label_split": "random"})
    )
    runs["target_normalized"] = runs["target"].replace(
        {"Dalaney": "ESOL", "freesolv": "FreeSolv"}
    )
    descriptor_flag = runs["exclude_descriptors"].astype(str).str.strip().str.lower()
    runs["model_type_normalized"] = np.select(
        [
            runs["explanatory_val"].eq("MFPS") & descriptor_flag.eq("true"),
            runs["explanatory_val"].eq("MFPS/Discriptor") & descriptor_flag.eq("false"),
        ],
        [
            "Fingerprint-based (ECFP6)",
            "Fingerprint-based (ECFP6 + descriptors)",
        ],
        default="",
    )
    runs = runs.loc[
        runs["training_seed"].isin(seeds)
        & runs["max_evals_num"].eq(max_evals)
        & runs["split_type"].isin(DEFAULT_SPLITS)
        & runs["model_type_normalized"].ne("")
    ].copy()
    scaffold = runs["split_type"].eq("balanced_scaffold")
    runs = runs.loc[
        ~scaffold | runs["split_seed_num"].eq(scaffold_split_seed)
    ].copy()
    runs.loc[~scaffold.loc[runs.index], "split_seed_num"] = np.nan
    runs["start_time_parsed"] = pd.to_datetime(runs["Start Time"], errors="coerce")
    if runs["start_time_parsed"].isna().any():
        bad_ids = runs.loc[runs["start_time_parsed"].isna(), "Run ID"].tolist()
        raise ValueError(f"Invalid Hyperopt Start Time for runs: {bad_ids}")

    group_columns = [
        "model_type_normalized", "target_normalized", "split_type",
        "split_seed_num", "training_seed",
    ]
    tied = runs.duplicated(group_columns + ["start_time_parsed"], keep=False)
    if tied.any():
        raise ValueError(
            "Hyperopt runs have identical latest Start Time: "
            + ", ".join(runs.loc[tied, "Run ID"].astype(str))
        )
    # A rerun supersedes an older run with the same experiment conditions.
    runs = (
        runs.sort_values("start_time_parsed", ascending=False)
        .drop_duplicates(group_columns, keep="first")
        .copy()
    )

    source_names = {
        "test_accuracy": "acc",
        **{
            metric: metric.removeprefix("test_")
            for metric in (*METRICS_BY_TASK["classification"], *METRICS_BY_TASK["regression"])
            if metric != "test_accuracy"
        },
    }
    records: list[dict[str, object]] = []
    for _, row in runs.iterrows():
        record = {column: np.nan for column in RUN_COLUMNS}
        record.update({
            "source": "hyperopt_csv",
            "source_run_id": row["Run ID"],
            "model_type": row["model_type_normalized"],
            "task_type": task_type,
            "target": row["target_normalized"],
            "split_type": row["split_type"],
            "split_seed": row["split_seed_num"],
            "training_seed": row["training_seed"],
            "learning_rate": _numeric(row.get("learning_rate")),
            "data_augmentation": "not_applicable",
        })
        for metric in METRICS_BY_TASK[task_type]:
            record[metric] = _numeric(row[source_names[metric]])
        records.append(record)
    return pd.DataFrame(records, columns=RUN_COLUMNS)


def load_hyperopt_test_runs(
    classification_csv: Path,
    regression_csv: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_evals: int = 100,
    scaffold_split_seed: int = 42,
) -> pd.DataFrame:
    """Load final Hyperopt CSV metrics, retaining the latest duplicate run."""

    frames = [
        _load_hyperopt_file(
            classification_csv, task_type="classification", seeds=seeds,
            max_evals=max_evals, scaffold_split_seed=scaffold_split_seed,
        ),
        _load_hyperopt_file(
            regression_csv, task_type="regression", seeds=seeds,
            max_evals=max_evals, scaffold_split_seed=scaffold_split_seed,
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def combine_test_runs(*frames: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True).reindex(columns=RUN_COLUMNS)
    key = ["model_type", "target", "split_type", "split_seed", "training_seed"]
    duplicates = combined.duplicated(key, keep=False)
    if duplicates.any():
        details = combined.loc[duplicates, ["source_run_id", *key]].to_dict("records")
        raise ValueError(f"Duplicate final test runs: {details}")
    return combined.sort_values(
        ["task_type", "target", "split_type", "model_type", "training_seed"]
    ).reset_index(drop=True)


def validate_test_runs(
    runs: pd.DataFrame,
    *,
    expected_models: Sequence[str] = MODEL_ORDER,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    point_estimate_models: Sequence[str] = ("ImageMol",),
    strict: bool = True,
) -> pd.DataFrame:
    """Return missing expected combinations and reject invalid metric values."""

    expected_records = []
    targets_by_task = {
        "classification": CLASSIFICATION_TARGETS,
        "regression": REGRESSION_TARGETS,
    }
    for task_type, targets in targets_by_task.items():
        for model in expected_models:
            for target in targets:
                for split in DEFAULT_SPLITS:
                    model_seeds = (
                        (np.nan,) if model in point_estimate_models else expected_seeds
                    )
                    for seed in model_seeds:
                        expected_records.append((task_type, model, target, split, seed))
    key = ["task_type", "model_type", "target", "split_type", "training_seed"]
    expected = pd.DataFrame(expected_records, columns=key)
    actual = runs[key].copy()
    actual["training_seed"] = pd.to_numeric(actual["training_seed"], errors="coerce")
    sentinel = -1
    expected["training_seed"] = expected["training_seed"].fillna(sentinel)
    actual["training_seed"] = actual["training_seed"].fillna(sentinel)
    missing = expected.merge(actual.drop_duplicates(), on=key, how="left", indicator=True)
    missing = missing.loc[missing["_merge"].eq("left_only"), key].reset_index(drop=True)
    missing["training_seed"] = missing["training_seed"].replace(sentinel, np.nan)

    for task_type, metrics in METRICS_BY_TASK.items():
        subset = runs.loc[runs["task_type"].eq(task_type)]
        values = subset[list(metrics)].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{task_type} test metrics contain missing/non-finite values")
    auc = pd.to_numeric(
        runs.loc[runs["task_type"].eq("classification"), "test_roc_auc"],
        errors="coerce",
    )
    if not auc.between(0.0, 1.0).all():
        raise ValueError("test_roc_auc must be between 0 and 1")
    if strict and not missing.empty:
        raise ValueError(f"Missing {len(missing)} expected final test runs:\n{missing}")
    return missing


def summarize_test_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    groups = ["task_type", "model_type", "target", "split_type", "split_seed"]
    for group_values, group in runs.groupby(groups, dropna=False, sort=True):
        task_type = group_values[0]
        for metric in METRICS_BY_TASK[task_type]:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            training_seeds = tuple(
                sorted(
                    pd.to_numeric(group["training_seed"], errors="coerce")
                    .dropna().astype(int).tolist()
                )
            )
            records.append({
                **dict(zip(groups, group_values)),
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "training_seeds": training_seeds,
                "source_run_ids": tuple(group.sort_values("training_seed")["source_run_id"]),
            })
    return pd.DataFrame(records)


def plot_test_metric_by_split(
    summary: pd.DataFrame,
    *,
    metric: str,
    targets: Sequence[str],
    model_order: Sequence[str] = MODEL_ORDER,
    y_limits: tuple[float, float] | None = None,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    colors = {
        "ResNet18": "#4D4D4D",
        "ImageMol": "#4C72B0",
        "Fingerprint-based (ECFP6)": "#79B98A",
        "Fingerprint-based (ECFP6 + descriptors)": "#3B7F5A",
        "D-MPNN": "#CCB974",
        "ChemAutoVision": "#E24A33",
    }
    metric_rows = summary.loc[summary["metric"].eq(metric)]
    fig, raw_axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    axes = dict(zip(DEFAULT_SPLITS, np.atleast_1d(raw_axes)))
    x = np.arange(len(targets), dtype=float)
    width = min(0.72 / len(model_order), 0.18)
    for split_index, split in enumerate(DEFAULT_SPLITS):
        ax = axes[split]
        for model_index, model in enumerate(model_order):
            rows = (
                metric_rows.loc[
                    metric_rows["split_type"].eq(split)
                    & metric_rows["model_type"].eq(model)
                ]
                .set_index("target").reindex(targets)
            )
            values = rows["mean"].to_numpy(dtype=float)
            errors = np.nan_to_num(rows["std"].to_numpy(dtype=float), nan=0.0)
            positions = x + (model_index - (len(model_order) - 1) / 2) * width
            bars = ax.bar(
                positions, values, width, yerr=errors, capsize=4, label=model,
                color=colors[model], edgecolor="black", linewidth=0.5, alpha=0.85,
            )
            for bar, value, error in zip(bars, values, errors):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.3f}",
                        (bar.get_x() + bar.get_width() / 2, value + error),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=18, rotation=90,
                    )
        title = "Random split" if split == "random" else "Balanced Scaffold split"
        ax.set_title(f"{METRIC_LABELS[metric]} — {title}")
        ax.set_ylabel(METRIC_LABELS[metric] if split_index == 0 else "")
        ax.set_xticks(x, targets)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    handles, labels = axes["random"].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075),
        ncol=3, frameon=True,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    return fig, axes


def build_metric_table(
    summary: pd.DataFrame,
    *,
    task_type: str,
    split_type: str,
    targets: Sequence[str],
    model_order: Sequence[str] = MODEL_ORDER,
    include_mse: bool = False,
    decimals: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = list(METRICS_BY_TASK[task_type])
    if task_type == "regression" and not include_mse:
        metrics.remove("test_mse")
    rows = summary.loc[
        summary["task_type"].eq(task_type)
        & summary["split_type"].eq(split_type)
        & summary["metric"].isin(metrics)
    ].copy()
    expected_counts = rows["model_type"].map(
        lambda model: 1 if model == "ImageMol" else len(DEFAULT_SEEDS)
    )
    if not rows["n_seeds"].eq(expected_counts).all():
        raise ValueError(f"{task_type}/{split_type} table has incomplete run groups")
    best = {}
    for (target, metric), group in rows.groupby(["target", "metric"]):
        best[(target, metric)] = (
            group["mean"].min() if metric in MINIMIZED_METRICS else group["mean"].max()
        )
    display_records = []
    latex_records = []
    for target in targets:
        for model in model_order:
            group = rows.loc[rows["target"].eq(target) & rows["model_type"].eq(model)]
            if group.empty:
                continue
            display_row = {"Dataset": target, "Model": model}
            latex_row = {"Dataset": target, "Model": model}
            for metric in metrics:
                result = group.loc[group["metric"].eq(metric)].iloc[0]
                if result["n_seeds"] == 1:
                    text = f"{result['mean']:.{decimals}f}"
                else:
                    text = f"{result['mean']:.{decimals}f} ± {result['std']:.{decimals}f}"
                label = METRIC_LABELS[metric]
                display_row[label] = text
                if " ± " in text:
                    mean_text, std_text = text.split(" ± ")
                    latex_text = rf"\makecell[tc]{{{mean_text}\\$\pm$ {std_text}}}"
                else:
                    latex_text = rf"\makecell[tc]{{{text}}}"
                if np.isclose(result["mean"], best[(target, metric)], rtol=0, atol=1e-12):
                    latex_text = f"\\textbf{{{latex_text}}}"
                latex_row[label] = latex_text
            display_records.append(display_row)
            latex_records.append(latex_row)
    return pd.DataFrame(display_records), pd.DataFrame(latex_records)


def build_combined_metric_table(
    summary: pd.DataFrame,
    *,
    task_type: str,
    targets: Sequence[str],
    model_order: Sequence[str] = MODEL_ORDER,
    include_mse: bool = False,
    decimals: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine Random and Balanced Scaffold results in the established row order."""

    split_names = {
        "random": "Random",
        "balanced_scaffold": "Balanced Scaffold",
    }
    split_tables = {}
    for split_type in DEFAULT_SPLITS:
        split_tables[split_type] = build_metric_table(
            summary,
            task_type=task_type,
            split_type=split_type,
            targets=targets,
            model_order=model_order,
            include_mse=include_mse,
            decimals=decimals,
        )

    combined = []
    for table_index in (0, 1):
        parts = []
        for target in targets:
            for split_type in DEFAULT_SPLITS:
                part = split_tables[split_type][table_index]
                part = part.loc[part["Dataset"].eq(target)].copy()
                part.insert(1, "Data split", split_names[split_type])
                parts.append(part)
        table = pd.concat(parts, ignore_index=True)
        table["Model"] = table["Model"].replace({"ResNet18": "ResNet18(w/ Aug)"})
        if task_type == "classification":
            table = table.rename(columns={"ROC-AUC": "AUC"})
            table = table[[
                "Dataset", "Data split", "Model",
                "Acc", "F1", "MCC", "Prec", "Rec", "AUC",
            ]]
        else:
            table = table[[
                "Dataset", "Data split", "Model",
                "MAE", "MSE", "$R^2$", "RMSE",
            ]]
        combined.append(table)
    return combined[0], combined[1]


def write_performance_latex_table(
    table: pd.DataFrame,
    *,
    task_type: str,
    output_path: Path,
) -> str:
    """Write current metrics using the established classification/regression format."""

    captions = {
        "classification": (
            "Performance metrics of classification models under Random and Balanced "
            "Scaffold data splits. Except for ImageMol, values are mean $\\pm$ sample "
            "SD over seeds 10, 42, and 99. ImageMol values are single-run point "
            "estimates because ImageMol does not perform AutoML hyperparameter "
            "optimization. The best mean value or single-run value for each dataset, "
            "data split, and metric is shown in bold."
        ),
        "regression": (
            "Performance metrics of regression models under Random and Balanced "
            "Scaffold data splits. Except for ImageMol, values are mean $\\pm$ sample "
            "SD over seeds 10, 42, and 99. ImageMol values are single-run point "
            "estimates because ImageMol does not perform AutoML hyperparameter "
            "optimization. The best mean value or single-run value for each dataset, "
            "data split, and metric is shown in bold."
        ),
    }
    if task_type not in captions:
        raise ValueError(f"Unknown task type: {task_type}")
    metric_columns = list(table.columns[3:])
    expected_metrics = {
        "classification": ["Acc", "F1", "MCC", "Prec", "Rec", "AUC"],
        "regression": ["MAE", "MSE", "$R^2$", "RMSE"],
    }[task_type]
    if metric_columns != expected_metrics:
        raise ValueError(
            f"Unexpected {task_type} metric columns: {metric_columns}"
        )

    column_count = len(table.columns)
    tabcolsep_count = 2 * column_count
    metric_width = (
        rf"\dimexpr(\textwidth-7.0cm-{tabcolsep_count}\tabcolsep)"
        rf"/{len(metric_columns)}\relax"
    )
    metric_column = rf">{{\centering\arraybackslash}}p{{{metric_width}}}"
    column_format = "p{1.1cm}p{2.1cm}p{3.8cm}" + metric_column * len(metric_columns)
    header = (
        "Dataset & Data split & Model & "
        + " & ".join(f"{{{column}}}" for column in metric_columns)
        + r" \\"
    )
    label = f"tab:{task_type}_performance"
    lines = [
        r"\begingroup",
        r"\small",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        rf"\begin{{longtable}}{{{column_format}}}",
        rf"\caption{{{captions[task_type]}}}\label{{{label}}}  \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{column_count}}}{{c}}{{\tablename\ \thetable{{}} -- continued from previous page}}  \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        rf"\midrule \multicolumn{{{column_count}}}{{r}}{{Continued on next page}}  \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    previous_dataset = None
    previous_split = None
    for _, row in table.iterrows():
        dataset = str(row["Dataset"])
        split = str(row["Data split"])
        if previous_dataset is not None and dataset != previous_dataset:
            lines.extend([r"\addlinespace[1pt]", r"\midrule"])
        elif previous_split is not None and split != previous_split:
            lines.append(r"\addlinespace[1pt]")
        dataset_cell = dataset if dataset != previous_dataset else ""
        split_cell = (
            split if dataset != previous_dataset or split != previous_split else ""
        )
        if split_cell == "Balanced Scaffold":
            split_cell = r"\makecell[tl]{Balanced\\Scaffold}"
        values = [dataset_cell, split_cell, str(row["Model"])]
        values.extend(str(row[column]) for column in metric_columns)
        lines.append(" & ".join(values) + r" \\")
        previous_dataset = dataset
        previous_split = split
    lines.extend([r"\end{longtable}", r"\endgroup"])
    latex = "\n".join(lines) + "\n"
    output_path.write_text(latex, encoding="utf-8")
    return latex


def load_paper_table_baseline(
    classification_csv: Path, regression_csv: Path
) -> pd.DataFrame:
    """Read the rounded mean/SD values currently published in paper_tables."""

    specs = {
        "classification": {
            "path": classification_csv,
            "columns": {
                "Acc": "test_accuracy", "F1": "test_f1", "MCC": "test_mcc",
                "Prec": "test_precision", "Rec": "test_recall", "AUC": "test_roc_auc",
            },
        },
        "regression": {
            "path": regression_csv,
            "columns": {
                "MAE": "test_mae", "MSE": "test_mse", "$R^2$": "test_r2",
                "RMSE": "test_rmse",
            },
        },
    }
    records = []
    for task_type, spec in specs.items():
        frame = pd.read_csv(spec["path"])
        frame["Model"] = frame["Model"].replace({"ResNet18(w/ Aug)": "ResNet18"})
        split = frame["Data split"].str.strip().str.lower().str.replace(" ", "_", regex=False)
        for index, row in frame.iterrows():
            for display_name, metric in spec["columns"].items():
                text = str(row[display_name]).strip()
                parts = [part.strip() for part in text.split("±", maxsplit=1)]
                records.append({
                    "task_type": task_type,
                    "model_type": row["Model"],
                    "target": row["Dataset"],
                    "split_type": split.loc[index],
                    "metric": metric,
                    "old_mean": float(parts[0]),
                    "old_std": float(parts[1]) if len(parts) == 2 else np.nan,
                })
    return pd.DataFrame(records)


def compare_with_paper_baseline(
    baseline: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    """Compare new summaries with the rounded values in the current paper tables."""

    key = ["task_type", "model_type", "target", "split_type", "metric"]
    new = summary.rename(columns={"mean": "new_mean", "std": "new_std"})
    columns = [*key, "new_mean", "new_std", "n_seeds", "source_run_ids"]
    comparison = baseline.merge(new[columns], on=key, how="outer", indicator=True)
    comparison["delta"] = comparison["new_mean"] - comparison["old_mean"]
    comparison["std_delta"] = comparison["new_std"] - comparison["old_std"]
    comparison["relative_delta_percent"] = np.where(
        comparison["old_mean"].ne(0),
        comparison["delta"] / comparison["old_mean"].abs() * 100,
        np.nan,
    )
    comparison["improvement"] = np.where(
        comparison["metric"].isin(MINIMIZED_METRICS),
        -comparison["delta"], comparison["delta"],
    )

    def impact(row: pd.Series) -> str:
        if row["_merge"] != "both":
            return "not_comparable"
        if row["task_type"] == "classification":
            change = abs(row["delta"])
            return "minor" if change < 0.005 else "review" if change < 0.02 else "major"
        change = abs(row["relative_delta_percent"])
        return "minor" if change < 1.0 else "review" if change < 5.0 else "major"

    comparison["impact_level"] = comparison.apply(impact, axis=1)
    for prefix, value_column in (("old", "old_mean"), ("new", "new_mean")):
        ascending = comparison["metric"].isin(MINIMIZED_METRICS)
        comparison[f"{prefix}_rank"] = np.nan
        for _, group in comparison.groupby(
            ["task_type", "target", "split_type", "metric"], dropna=False
        ):
            valid = group[value_column].notna()
            if not valid.any():
                continue
            metric_ascending = bool(ascending.loc[group.index[0]])
            ranks = group.loc[valid, value_column].rank(
                method="min", ascending=metric_ascending
            )
            comparison.loc[ranks.index, f"{prefix}_rank"] = ranks
    comparison["rank_delta"] = comparison["new_rank"] - comparison["old_rank"]
    comparison["best_model_changed"] = False
    group_columns = ["task_type", "target", "split_type", "metric"]
    for _, group in comparison.groupby(group_columns, dropna=False):
        old_best = set(group.loc[group["old_rank"].eq(1), "model_type"])
        new_best = set(group.loc[group["new_rank"].eq(1), "model_type"])
        comparison.loc[group.index, "best_model_changed"] = old_best != new_best
    return comparison.sort_values(key).reset_index(drop=True)


def plot_metric_delta_heatmap(
    comparison: pd.DataFrame,
    *,
    metric: str,
    model_order: Sequence[str] = MODEL_ORDER,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot improvement-oriented changes against the current paper table."""

    rows = comparison.loc[
        comparison["metric"].eq(metric) & comparison["_merge"].eq("both")
    ].copy()
    rows["condition"] = rows["target"] + "\n" + rows["split_type"].replace({
        "random": "Random", "balanced_scaffold": "Scaffold",
    })
    target_order = CLASSIFICATION_TARGETS if metric == "test_roc_auc" else REGRESSION_TARGETS
    condition_order = [
        f"{target}\n{split_name}"
        for target in target_order for split_name in ("Random", "Scaffold")
    ]
    matrix = (
        rows.pivot(index="model_type", columns="condition", values="improvement")
        .reindex(index=model_order, columns=condition_order)
    )
    values = matrix.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(finite.max()) if finite.size else 1.0
    limit = limit or 1.0
    fig, ax = plt.subplots(figsize=(14, 5.5))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                ax.text(column_index, row_index, f"{value:+.3f}", ha="center", va="center")
    ax.set_title(
        f"Change from current paper table — {METRIC_LABELS[metric]} "
        "(positive means improvement)"
    )
    fig.colorbar(image, ax=ax, label="Improvement-oriented difference")
    fig.tight_layout()
    return fig, ax


def _architecture_makecell(values: Sequence[str]) -> str:
    """Format one or more architecture labels for the legacy table layout."""

    labels = [str(value).strip() for value in values if str(value).strip()]
    if not labels:
        return "-"
    if len(labels) == 1:
        return labels[0]
    return r"\makecell[tl]{" + r"\\".join(labels) + "}"


def _format_augmentation_latex(value: Any) -> str:
    mapping = {
        "RandomTranslation": "Trans.",
        "RandomRotation": "Rot.",
    }
    parts = [part.strip() for part in str(value).split(",")]
    return _architecture_makecell([mapping.get(part, part) for part in parts])


def _format_cnn_latex(value: Any) -> str:
    text = str(value).strip()
    separable_pair = (
        "SeparableConv2D, MaxPooling2D, "
        "SeparableConv2D, MaxPooling2D"
    )
    if text == f"Vanilla CNN ({separable_pair})":
        return (
            r"\makecell[tl]{Vanilla CNN\\"
            r"$(\mathrm{SepConv2D},$\\"
            r"$\mathrm{MaxPooling2D})\times2$}"
        )
    match = re.fullmatch(
        r"Vanilla CNN \(Conv2D × (\d+), MaxPooling2D"
        r"(?:, Conv2D × \1, MaxPooling2D)+\)",
        text,
    )
    if match:
        pair_count = text.count("MaxPooling2D")
        conv_count = match.group(1)
        return (
            r"\makecell[tl]{Vanilla CNN\\$(\mathrm{Conv2D}\times"
            + conv_count
            + r",$\\$\mathrm{MaxPooling2D})\times"
            + str(pair_count)
            + "$}"
        )
    return text


def _format_post_cnn_latex(value: Any) -> str:
    mapping = {
        "GlobalAveragePooling2D": "GAP",
        "GlobalMaxPooling2D": "GMP",
    }
    labels = []
    for part in (item.strip() for item in str(value).split(",")):
        dropout = re.fullmatch(r"Dropout \(rate=([0-9.]+)\)", part)
        if dropout:
            labels.append(rf"Dropout ($p={float(dropout.group(1)):.2f}$)")
        else:
            labels.append(mapping.get(part, part))
    return _architecture_makecell(labels)


def write_architecture_latex_table(
    records: pd.DataFrame,
    output_path: Path,
) -> str:
    """Write current architecture rows using the established paper table format."""

    required = {
        "target", "task_type", "seed", "data_split",
        "Data Augmentation", "CNN Architecture", "Post-CNN layers",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Architecture table records are missing: {missing}")

    target_order = [*CLASSIFICATION_TARGETS, *REGRESSION_TARGETS]
    target_rank = {target: index for index, target in enumerate(target_order)}
    split_rank = {"random": 0, "balanced_scaffold": 1}
    rows = records.copy()
    rows["_target_rank"] = rows["target"].map(target_rank)
    rows["_split_rank"] = rows["data_split"].map(split_rank)
    rows["_seed_rank"] = pd.to_numeric(rows["seed"], errors="coerce")
    rows = rows.sort_values(["_target_rank", "_split_rank", "_seed_rank"])

    lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        "",
        r"\begin{longtable}{",
        r"    >{\raggedright\arraybackslash}p{1.45cm}",
        r"    >{\centering\arraybackslash}p{0.75cm}",
        r"    >{\centering\arraybackslash}p{0.65cm}",
        r"    >{\raggedright\arraybackslash}p{2.45cm}",
        r"    >{\raggedright\arraybackslash}p{2.85cm}",
        r"    >{\raggedright\arraybackslash}p{2.50cm}",
        r"    >{\raggedright\arraybackslash}p{3.75cm}",
        r"}",
        "",
        r"\caption{AutoKeras-optimized ChemAutoVision architectures across tasks, datasets, random seeds, and data splits. All selected models use an input shape of $256\times256\times3$, CastToFloat32 preprocessing,",
        r"and a final Dense layer with one unit. Classification models use a sigmoid output activation,",
        r"whereas regression models use a linear output activation. Balanced scaffold splitting uses",
        r"\texttt{split\_seed}=42. Post-CNN layers comprise feature aggregation or reshaping layers",
        r"(Flatten, GAP, or GMP) and, where selected, Dropout regularization. Cls.: classification;",
        r"Reg.: regression; Trans.: RandomTranslation; Rot.: RandomRotation;",
        r"GAP: GlobalAveragePooling2D; GMP: GlobalMaxPooling2D; $p$: dropout rate.}",
        r"\label{tab:autokeras_architectures} \\",
        "",
        r"\toprule",
        r"\textbf{Dataset} &",
        r"\textbf{Task} &",
        r"\textbf{Seed} &",
        r"\textbf{Split} &",
        r"\makecell{\textbf{Data}\\\textbf{augmentation}} &",
        r"\makecell{\textbf{CNN}\\\textbf{architecture}} &",
        r"\makecell{\textbf{Post-CNN}\\\textbf{layers}} \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\multicolumn{7}{c}{\tablename\ \thetable{} -- continued from previous page} \\",
        r"\toprule",
        r"\textbf{Dataset} &",
        r"\textbf{Task} &",
        r"\textbf{Seed} &",
        r"\textbf{Split} &",
        r"\makecell{\textbf{Data}\\\textbf{augmentation}} &",
        r"\makecell{\textbf{CNN}\\\textbf{architecture}} &",
        r"\makecell{\textbf{Post-CNN}\\\textbf{layers}} \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{7}{r}{Continued on next page} \\",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
        "",
    ]
    previous_target = None
    for _, row in rows.iterrows():
        target = str(row["target"])
        if previous_target is not None and target != previous_target:
            lines.extend(["", r"\midrule", ""])
        task = "Cls." if row["task_type"] == "classification" else "Reg."
        split = "Random" if row["data_split"] == "random" else "Balanced scaffold"
        augmentation = _format_augmentation_latex(row["Data Augmentation"])
        cnn = _format_cnn_latex(row["CNN Architecture"])
        post_cnn = _format_post_cnn_latex(row["Post-CNN layers"])
        lines.append(
            f"{target} & {task} & {int(row['seed'])} & {split} & "
            f"{augmentation} & {cnn} & {post_cnn} \\\\"
        )
        previous_target = target
    lines.extend(["", r"\end{longtable}", "", r"\endgroup"])
    latex = "\n".join(lines) + "\n"
    output_path.write_text(latex, encoding="utf-8")
    return latex


def compare_architecture_records(
    old_records: pd.DataFrame,
    new_records: pd.DataFrame,
    *,
    components: Sequence[str],
) -> pd.DataFrame:
    """Return a long-form component diff for old and newly selected models."""

    key = ["task_type", "data_split", "target", "seed"]
    metadata = ["Run ID", "batch_size", "model_path", "model_exists"]
    required = {*key, *metadata, *components}
    for label, frame in (("old", old_records), ("new", new_records)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} architecture records are missing: {missing}")
        if frame.duplicated(key).any():
            raise ValueError(f"{label} architecture records contain duplicate conditions")
    old = old_records[[*key, *metadata, *components]].rename(
        columns={column: f"old_{column}" for column in [*metadata, *components]}
    )
    new = new_records[[*key, *metadata, *components]].rename(
        columns={column: f"new_{column}" for column in [*metadata, *components]}
    )
    merged = old.merge(new, on=key, how="outer", indicator=True)
    records = []
    for _, row in merged.iterrows():
        run_changed = row.get("old_Run ID") != row.get("new_Run ID")
        models_available = (
            row["_merge"] == "both"
            and bool(row.get("old_model_exists"))
            and bool(row.get("new_model_exists"))
        )
        for component in components:
            old_value = row.get(f"old_{component}")
            new_value = row.get(f"new_{component}")
            equal = (
                pd.isna(old_value) and pd.isna(new_value)
            ) or str(old_value) == str(new_value)
            records.append({
                **{column: row[column] for column in key},
                "component": component,
                "old_value": old_value,
                "new_value": new_value,
                "changed": bool(not equal) if models_available else False,
                "run_changed": run_changed,
                "old_run_id": row.get("old_Run ID"),
                "new_run_id": row.get("new_Run ID"),
                "old_batch_size": row.get("old_batch_size"),
                "new_batch_size": row.get("new_batch_size"),
                "old_model_path": row.get("old_model_path"),
                "new_model_path": row.get("new_model_path"),
                "comparison_status": (
                    "comparable" if models_available
                    else "model_missing" if row["_merge"] == "both"
                    else row["_merge"]
                ),
            })
    return pd.DataFrame(records).sort_values([*key, "component"]).reset_index(drop=True)


def plot_architecture_delta_heatmap(
    comparison: pd.DataFrame,
    *,
    components: Sequence[str],
) -> tuple[plt.Figure, plt.Axes]:
    """Show old and new values for changed ChemAutoVision components."""

    rows = comparison.copy()
    rows["condition"] = (
        rows["target"].astype(str) + " / "
        + rows["data_split"].replace({
            "random": "Random", "balanced_scaffold": "Scaffold",
        }).astype(str)
        + " / seed=" + rows["seed"].astype(int).astype(str)
    )
    condition_order = rows[
        ["task_type", "target", "data_split", "seed", "condition"]
    ].drop_duplicates().sort_values(
        ["task_type", "target", "data_split", "seed"]
    )["condition"]
    matrix = (
        rows.pivot(index="condition", columns="component", values="changed")
        .reindex(index=condition_order, columns=components)
        .astype(float)
    )
    status_matrix = (
        rows.pivot(index="condition", columns="component", values="comparison_status")
        .reindex(index=condition_order, columns=components)
    )
    matrix = matrix.where(status_matrix.eq("comparable"))
    height = max(8.0, 0.38 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(22, height))
    color_map = plt.get_cmap("Reds").copy()
    color_map.set_bad("#D9D9D9")
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = ax.imshow(values, cmap=color_map, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            value = matrix.iloc[row_index, column_index]
            comparable = pd.notna(value)
            changed = comparable and bool(value)
            if changed:
                condition = matrix.index[row_index]
                component = matrix.columns[column_index]
                delta_row = rows.loc[
                    rows["condition"].eq(condition)
                    & rows["component"].eq(component)
                ].iloc[0]
                annotation = f"{delta_row['old_value']}\n→\n{delta_row['new_value']}"
            else:
                annotation = "—" if comparable else "N/A"
            ax.text(
                column_index, row_index,
                annotation,
                ha="center", va="center", fontsize=6 if changed else 8,
                color="white" if changed else "#444444",
            )
    ax.set_title("ChemAutoVision architecture changes from the current paper selection")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1], label="Architecture difference")
    colorbar.ax.set_yticklabels(["No difference", "Old → New shown"])
    fig.tight_layout()
    return fig, ax


def plot_hyperparameter_change_summary(
    comparison: pd.DataFrame,
    *,
    model_order: Sequence[str] = MODEL_ORDER,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot selected-run batch-size and learning-rate comparison statuses."""

    statuses = (
        "changed", "unchanged", "not_recorded",
        "recording_changed", "run_not_comparable",
    )
    colors = {
        "changed": "#D55E00",
        "unchanged": "#4C9F70",
        "not_recorded": "#BDBDBD",
        "recording_changed": "#E6AB02",
        "run_not_comparable": "#7F7F7F",
    }
    labels = {
        "changed": "Changed",
        "unchanged": "Unchanged",
        "not_recorded": "Not recorded",
        "recording_changed": "Recording availability changed",
        "run_not_comparable": "Run not comparable",
    }
    available_models = [
        model for model in model_order
        if comparison["model_type"].eq(model).any()
    ]
    fig, raw_axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    axes = np.atleast_1d(raw_axes)
    for axis, parameter, title in zip(
        axes,
        ("batch_size", "learning_rate"),
        ("Batch size", "Learning rate"),
    ):
        counts = (
            comparison.groupby(["model_type", f"{parameter}_status"])
            .size().unstack(fill_value=0)
            .reindex(index=available_models, columns=statuses, fill_value=0)
        )
        left = np.zeros(len(counts), dtype=float)
        y = np.arange(len(counts))
        for status in statuses:
            values = counts[status].to_numpy(dtype=float)
            bars = axis.barh(
                y, values, left=left, color=colors[status],
                label=labels[status], edgecolor="white", linewidth=0.5,
            )
            for bar, value in zip(bars, values):
                if value > 0:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(value)), ha="center", va="center", fontsize=9,
                        color="white" if status in {
                            "changed", "unchanged", "run_not_comparable"
                        } else "#333333",
                    )
            left += values
        axis.set_title(title)
        axis.set_xlabel("Selected run conditions")
        axis.set_yticks(y, counts.index)
        axis.grid(True, axis="x", alpha=0.25)
        axis.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle("Selected-model hyperparameter changes from the current paper")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    return fig, axes


def flatten_legacy_hyperparameters(
    selected_run_sets: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]]
) -> pd.DataFrame:
    """Normalize the runs used by the current paper into a hyperparameter table."""

    records = []
    for task_type, task_sets in selected_run_sets.items():
        for result_kind in ("repeated_runs", "point_runs"):
            for raw_model, frame in task_sets[result_kind].items():
                model = {"ResNet18(w/ Aug)": "ResNet18"}.get(raw_model, raw_model)
                for _, row in frame.iterrows():
                    seed = row.get("compare_seed", np.nan)
                    selected_batch = row.get("selected_batch_size", np.nan)
                    if pd.isna(selected_batch):
                        selected_batch = pd.to_numeric(row.get("batch_size"), errors="coerce")
                    records.append({
                        "task_type": task_type,
                        "model_type": model,
                        "target": row["target_display"],
                        "split_type": row["data_split_normalized"],
                        "training_seed": seed,
                        "old_run_id": row["Run ID"],
                        "old_batch_size": selected_batch,
                        "old_learning_rate": pd.to_numeric(
                            row.get("learning_rate"), errors="coerce"
                        ),
                    })
    result = pd.DataFrame(records)
    key = ["task_type", "model_type", "target", "split_type", "training_seed"]
    if result.duplicated(key).any():
        raise ValueError("Legacy selected runs contain duplicate hyperparameter keys")
    return result


def compare_selected_hyperparameters(
    old_hyperparameters: pd.DataFrame, new_runs: pd.DataFrame
) -> pd.DataFrame:
    """Compare batch size and learning rate for old and new selected runs."""

    key = ["task_type", "model_type", "target", "split_type", "training_seed"]
    new = new_runs[[*key, "source_run_id", "batch_size", "learning_rate"]].rename(
        columns={
            "source_run_id": "new_run_id",
            "batch_size": "new_batch_size",
            "learning_rate": "new_learning_rate",
        }
    )
    sentinel = -1
    old = old_hyperparameters.copy()
    old["training_seed"] = pd.to_numeric(old["training_seed"], errors="coerce").fillna(sentinel)
    new["training_seed"] = pd.to_numeric(new["training_seed"], errors="coerce").fillna(sentinel)
    comparison = old.merge(new, on=key, how="outer", indicator=True)
    comparison["training_seed"] = comparison["training_seed"].replace(sentinel, np.nan)

    def status(row: pd.Series, name: str) -> str:
        old_value = row[f"old_{name}"]
        new_value = row[f"new_{name}"]
        if row["_merge"] != "both":
            return "run_not_comparable"
        if pd.isna(old_value) and pd.isna(new_value):
            return "not_recorded"
        if pd.isna(old_value) or pd.isna(new_value):
            return "recording_changed"
        return "unchanged" if np.isclose(old_value, new_value) else "changed"

    for name in ("batch_size", "learning_rate"):
        comparison[f"{name}_status"] = comparison.apply(
            lambda row: status(row, name), axis=1
        )
        comparison[f"{name}_changed"] = comparison[f"{name}_status"].eq("changed")
    comparison["run_changed"] = comparison["old_run_id"] != comparison["new_run_id"]
    return comparison.sort_values(key).reset_index(drop=True)
