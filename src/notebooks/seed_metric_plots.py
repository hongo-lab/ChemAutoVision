"""Seed-aware performance aggregation and plotting for visualize_for_paper.ipynb."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (10, 42, 99)
DEFAULT_SPLITS = ("random", "balanced_scaffold")
TARGET_ALIASES = {"Dalaney": "ESOL", "freesolv": "FreeSolv"}


def _blank_to_nan(series: pd.Series) -> pd.Series:
    return series.replace(r"^\s*(?:None|null|nan|NaN)?\s*$", np.nan, regex=True)


def _boolean_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def _coalesce_numeric_columns(
    df: pd.DataFrame, columns: Sequence[str]
) -> pd.Series:
    values = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        if column in df.columns:
            candidate = pd.to_numeric(_blank_to_nan(df[column]), errors="coerce")
            values = values.fillna(candidate)
    return values


def _normalize_split(df: pd.DataFrame, *, use_imagemol_strategy: bool = False) -> pd.Series:
    if "data_split" in df.columns:
        split = _blank_to_nan(df["data_split"])
    else:
        split = pd.Series(np.nan, index=df.index, dtype=object)
    if "split_type" in df.columns:
        split = split.fillna(_blank_to_nan(df["split_type"]))
    split = split.fillna("random")
    split = split.astype(str).str.strip().str.lower().replace({
        "label": "random",
        "label_split": "random",
    })
    if use_imagemol_strategy and "dataset_split_strategy" in df.columns:
        raw_strategy = _blank_to_nan(df["dataset_split_strategy"])
        has_strategy = raw_strategy.notna()
        strategy = raw_strategy.astype(str).str.strip().str.lower()
        split = strategy.where(has_strategy, split)
    return split


def select_model_seed_runs(
    df: pd.DataFrame,
    *,
    model_name: str,
    targets: Sequence[str],
    metric: str,
    direction: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_trials: int = 100,
    scaffold_split_seed: int = 42,
    require_data_aug: bool = True,
) -> pd.DataFrame:
    """Select the best eligible run per target, split, and search seed.

    Every seed is optimized independently across all eligible batch sizes. This
    matches the paper aggregation protocol: select one best run for each seed,
    then calculate the mean and sample SD across the selected runs.
    """
    if direction not in {"max", "min"}:
        raise ValueError("direction must be 'max' or 'min'.")

    required = {
        "Run ID", "model_name", "target", "seed", "ak_seed", "batch_size",
        "max_trials", "execute_data_aug", "data_split", "split_seed", metric,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing result columns: {missing}")

    runs = df.loc[df["model_name"].eq(model_name)].copy()
    seed = pd.to_numeric(_blank_to_nan(runs["seed"]), errors="coerce")
    legacy_seed = pd.to_numeric(_blank_to_nan(runs["ak_seed"]), errors="coerce")
    runs["compare_seed"] = seed.fillna(legacy_seed)
    runs["target_display"] = runs["target"].replace(TARGET_ALIASES)
    runs["metric_value"] = pd.to_numeric(
        runs[metric].astype(str).str.strip("'"), errors="coerce"
    )
    runs["batch_size_num"] = pd.to_numeric(runs["batch_size"], errors="coerce")
    runs["max_trials_num"] = pd.to_numeric(runs["max_trials"], errors="coerce")
    runs["split_seed_num"] = _coalesce_numeric_columns(
        runs, ("split_seed", "split_seed.1", "dataset_split_seed")
    )
    runs["data_split_normalized"] = _normalize_split(runs)

    runs = runs[
        runs["target_display"].isin(targets)
        & runs["compare_seed"].isin(seeds)
        & runs["max_trials_num"].eq(max_trials)
        & runs["metric_value"].notna()
        & runs["data_split_normalized"].isin(DEFAULT_SPLITS)
    ].copy()
    if require_data_aug:
        runs = runs[_boolean_mask(runs["execute_data_aug"])]

    scaffold = runs["data_split_normalized"].eq("balanced_scaffold")
    runs = runs[~scaffold | runs["split_seed_num"].eq(scaffold_split_seed)].copy()

    if runs.empty:
        return pd.DataFrame(
            columns=[
                "data_split_normalized", "target_display", "compare_seed",
                "selected_batch_size", "metric_value", "Run ID",
            ]
        )

    group_columns = ["data_split_normalized", "target_display", "compare_seed"]
    chooser = runs.groupby(group_columns)["metric_value"]
    best_index = chooser.idxmax() if direction == "max" else chooser.idxmin()
    selected = runs.loc[best_index].copy()
    selected["selected_batch_size"] = selected["batch_size_num"]
    return selected.sort_values(group_columns).reset_index(drop=True)


def select_autokeras_seed_runs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Backward-compatible wrapper for AutoKeras seed aggregation."""
    return select_model_seed_runs(df, model_name="autokeras", **kwargs)


def select_hpsklearn_seed_runs(
    df: pd.DataFrame,
    *,
    targets: Sequence[str],
    metric: str,
    direction: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_evals: int = 100,
    scaffold_split_seed: int = 42,
    explanatory_val: str = "MFPS/Discriptor",
    exclude_descriptors: bool | None = None,
) -> pd.DataFrame:
    """Select the best hpsklearn run within each target, split, and seed."""
    if direction not in {"max", "min"}:
        raise ValueError("direction must be 'max' or 'min'.")

    required = {
        "Run ID", "model_name", "target", "training_seed", "max_evals",
        "data_split", metric,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing hpsklearn result columns: {missing}")

    runs = df.loc[df["model_name"].eq("HyperoptEstimator")].copy()
    runs["compare_seed"] = pd.to_numeric(
        _blank_to_nan(runs["training_seed"]), errors="coerce"
    )
    runs["target_display"] = runs["target"].replace(TARGET_ALIASES)
    runs["metric_value"] = pd.to_numeric(
        runs[metric].astype(str).str.strip("'"), errors="coerce"
    )
    runs["max_evals_num"] = pd.to_numeric(runs["max_evals"], errors="coerce")
    runs["split_seed_num"] = _coalesce_numeric_columns(
        runs, ("split_seed", "split_seed.1", "dataset_split_seed")
    )
    runs["data_split_normalized"] = _normalize_split(runs)

    runs = runs[
        runs["target_display"].isin(targets)
        & runs["compare_seed"].isin(seeds)
        & runs["max_evals_num"].eq(max_evals)
        & runs["metric_value"].notna()
        & runs["data_split_normalized"].isin(DEFAULT_SPLITS)
    ].copy()
    if "explanatory_val" in runs.columns:
        runs = runs[_blank_to_nan(runs["explanatory_val"]).eq(explanatory_val)].copy()
    if exclude_descriptors is not None:
        if "exclude_descriptors" not in runs.columns:
            raise ValueError("Missing hpsklearn result column: exclude_descriptors")
        descriptor_flag = _blank_to_nan(runs["exclude_descriptors"])
        descriptor_flag = descriptor_flag.astype(str).str.strip().str.lower()
        expected_flag = "true" if exclude_descriptors else "false"
        runs = runs[descriptor_flag.eq(expected_flag)].copy()
    scaffold = runs["data_split_normalized"].eq("balanced_scaffold")
    runs = runs[~scaffold | runs["split_seed_num"].eq(scaffold_split_seed)].copy()

    group_columns = ["data_split_normalized", "target_display", "compare_seed"]
    chooser = runs.groupby(group_columns)["metric_value"]
    best_index = chooser.idxmax() if direction == "max" else chooser.idxmin()
    selected = runs.loc[best_index].copy()
    selected["selected_batch_size"] = np.nan
    return selected.sort_values(group_columns).reset_index(drop=True)


def select_dmpnn_seed_runs(
    df: pd.DataFrame,
    *,
    targets: Sequence[str],
    metric: str,
    direction: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scaffold_split_seed: int = 42,
    require_class_weight: bool = False,
    model_names: Sequence[str] = ("chemprop", "chemprops"),
) -> pd.DataFrame:
    """Select the best D-MPNN run per target, split, and seed.

    Chemprop results use ``split_type`` rather than ``data_split`` in the
    current result CSVs. Both ``chemprop`` and the historical ``chemprops``
    model names are accepted.
    """
    if direction not in {"max", "min"}:
        raise ValueError("direction must be 'max' or 'min'.")

    required = {"Run ID", "model_name", "target", "seed", "batch_size", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing D-MPNN result columns: {missing}")

    runs = df.loc[df["model_name"].isin(model_names)].copy()
    runs["compare_seed"] = pd.to_numeric(
        _blank_to_nan(runs["seed"]), errors="coerce"
    )
    runs["target_display"] = runs["target"].replace(TARGET_ALIASES)
    runs["metric_value"] = pd.to_numeric(
        runs[metric].astype(str).str.strip("'"), errors="coerce"
    )
    runs["batch_size_num"] = pd.to_numeric(runs["batch_size"], errors="coerce")
    runs["split_seed_num"] = _coalesce_numeric_columns(
        runs, ("split_seed", "split_seed.1", "dataset_split_seed")
    )
    runs["data_split_normalized"] = _normalize_split(runs)

    runs = runs[
        runs["target_display"].isin(targets)
        & runs["compare_seed"].isin(seeds)
        & runs["metric_value"].notna()
        & runs["data_split_normalized"].isin(DEFAULT_SPLITS)
    ].copy()
    if require_class_weight:
        if "class_weight" not in runs.columns:
            raise ValueError("Missing D-MPNN result column: class_weight")
        runs = runs[_blank_to_nan(runs["class_weight"]).notna()].copy()

    scaffold = runs["data_split_normalized"].eq("balanced_scaffold")
    runs = runs[~scaffold | runs["split_seed_num"].eq(scaffold_split_seed)].copy()
    if runs.empty:
        return pd.DataFrame(
            columns=[
                "data_split_normalized", "target_display", "compare_seed",
                "selected_batch_size", "metric_value", "Run ID",
            ]
        )

    group_columns = ["data_split_normalized", "target_display", "compare_seed"]
    chooser = runs.groupby(group_columns)["metric_value"]
    best_index = chooser.idxmax() if direction == "max" else chooser.idxmin()
    selected = runs.loc[best_index].copy()
    selected["selected_batch_size"] = selected["batch_size_num"]
    return selected.sort_values(group_columns).reset_index(drop=True)


def select_imagemol_point_runs(
    df: pd.DataFrame,
    *,
    targets: Sequence[str],
    metric: str,
    direction: str,
) -> pd.DataFrame:
    """Select one ImageMol point estimate per target and split (no seed aggregation)."""
    if direction not in {"max", "min"}:
        raise ValueError("direction must be 'max' or 'min'.")
    required = {"Run ID", "model_name", "target", "data_split", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing ImageMol result columns: {missing}")

    runs = df.loc[df["model_name"].eq("ImageMol")].copy()
    runs["target_display"] = runs["target"].replace(TARGET_ALIASES)
    runs["metric_value"] = pd.to_numeric(
        runs[metric].astype(str).str.strip("'"), errors="coerce"
    )
    runs["data_split_normalized"] = _normalize_split(
        runs, use_imagemol_strategy=True
    )
    runs = runs[
        runs["target_display"].isin(targets)
        & runs["metric_value"].notna()
        & runs["data_split_normalized"].isin(DEFAULT_SPLITS)
    ].copy()

    group_columns = ["data_split_normalized", "target_display"]
    chooser = runs.groupby(group_columns)["metric_value"]
    best_index = chooser.idxmax() if direction == "max" else chooser.idxmin()
    return runs.loc[best_index].sort_values(group_columns).reset_index(drop=True)


def select_paper_performance_runs(
    classification_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    *,
    classification_targets: Sequence[str],
    regression_targets: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_trials: int = 100,
    max_evals: int = 100,
    scaffold_split_seed: int = 42,
    maximum_imagemol_batch_size: int | None = 64,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Recompute every run set used by the paper performance tables.

    This function intentionally starts from the raw result frames and reapplies
    the paper's candidate-run eligibility rules. Calling it immediately before
    table export prevents a live notebook kernel from reusing run selections
    created by an older version of the selection code.
    """
    classification_required = {
        "model_name", "class_weight", "weighted_balanced", "batch_size"
    }
    missing = sorted(classification_required - set(classification_df.columns))
    if missing:
        raise ValueError(f"Missing classification eligibility columns: {missing}")

    if "batch_size" not in regression_df.columns:
        raise ValueError("Missing regression eligibility column: batch_size")

    def imagemol_candidates(frame: pd.DataFrame) -> pd.DataFrame:
        if maximum_imagemol_batch_size is None:
            return frame.copy()
        batch_size = pd.to_numeric(frame["batch_size"], errors="coerce")
        return frame.loc[
            batch_size.isna() | batch_size.le(maximum_imagemol_batch_size)
        ].copy()

    classification_candidates = classification_df.copy()
    regression_candidates = regression_df.copy()
    class_weight = _blank_to_nan(classification_candidates["class_weight"])
    weighted_balanced = _boolean_mask(
        classification_candidates["weighted_balanced"]
    )
    is_hpsklearn = classification_candidates["model_name"].eq("HyperoptEstimator")
    is_imagemol = classification_candidates["model_name"].eq("ImageMol")
    eligible = (
        is_hpsklearn
        | (is_imagemol & weighted_balanced)
        | (~is_hpsklearn & ~is_imagemol & class_weight.notna())
    )
    classification_candidates = classification_candidates.loc[eligible].copy()

    selection_options = {
        "seeds": seeds,
        "max_trials": max_trials,
        "scaffold_split_seed": scaffold_split_seed,
    }
    hpsklearn_options = {
        "seeds": seeds,
        "max_evals": max_evals,
        "scaffold_split_seed": scaffold_split_seed,
    }
    dmpnn_options = {
        "seeds": seeds,
        "scaffold_split_seed": scaffold_split_seed,
    }

    classification_repeated = {
        "ResNet18(w/ Aug)": select_model_seed_runs(
            classification_candidates,
            model_name="resnet18",
            targets=classification_targets,
            metric="roc_auc",
            direction="max",
            **selection_options,
        ),
        "Fingerprint-based (ECFP6)": select_hpsklearn_seed_runs(
            classification_candidates,
            targets=classification_targets,
            metric="roc_auc",
            direction="max",
            explanatory_val="MFPS",
            exclude_descriptors=True,
            **hpsklearn_options,
        ),
        "Fingerprint-based (ECFP6 + descriptors)": select_hpsklearn_seed_runs(
            classification_candidates,
            targets=classification_targets,
            metric="roc_auc",
            direction="max",
            explanatory_val="MFPS/Discriptor",
            exclude_descriptors=False,
            **hpsklearn_options,
        ),
        "D-MPNN": select_dmpnn_seed_runs(
            classification_candidates,
            targets=classification_targets,
            metric="roc_auc",
            direction="max",
            require_class_weight=True,
            **dmpnn_options,
        ),
        "ChemAutoVision": select_autokeras_seed_runs(
            classification_candidates,
            targets=classification_targets,
            metric="roc_auc",
            direction="max",
            **selection_options,
        ),
    }
    regression_repeated = {
        "ResNet18(w/ Aug)": select_model_seed_runs(
            regression_candidates,
            model_name="resnet18",
            targets=regression_targets,
            metric="rmse",
            direction="min",
            **selection_options,
        ),
        "Fingerprint-based (ECFP6)": select_hpsklearn_seed_runs(
            regression_candidates,
            targets=regression_targets,
            metric="rmse",
            direction="min",
            explanatory_val="MFPS",
            exclude_descriptors=True,
            **hpsklearn_options,
        ),
        "Fingerprint-based (ECFP6 + descriptors)": select_hpsklearn_seed_runs(
            regression_candidates,
            targets=regression_targets,
            metric="rmse",
            direction="min",
            explanatory_val="MFPS/Discriptor",
            exclude_descriptors=False,
            **hpsklearn_options,
        ),
        "D-MPNN": select_dmpnn_seed_runs(
            regression_candidates,
            targets=regression_targets,
            metric="rmse",
            direction="min",
            **dmpnn_options,
        ),
        "ChemAutoVision": select_autokeras_seed_runs(
            regression_candidates,
            targets=regression_targets,
            metric="rmse",
            direction="min",
            **selection_options,
        ),
    }

    return {
        "classification": {
            "repeated_runs": classification_repeated,
            "point_runs": {
                "ImageMol": select_imagemol_point_runs(
                    imagemol_candidates(classification_candidates),
                    targets=classification_targets,
                    metric="roc_auc",
                    direction="max",
                )
            },
        },
        "regression": {
            "repeated_runs": regression_repeated,
            "point_runs": {
                "ImageMol": select_imagemol_point_runs(
                    imagemol_candidates(regression_candidates),
                    targets=regression_targets,
                    metric="rmse",
                    direction="min",
                )
            },
        },
    }


def summarize_seed_runs(
    selected_runs: pd.DataFrame,
    *,
    targets: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Return mean, sample SD, n, and missing seeds for every split and target."""
    records = []
    expected = {int(seed) for seed in seeds}
    for split_name in DEFAULT_SPLITS:
        for target in targets:
            rows = selected_runs[
                selected_runs["data_split_normalized"].eq(split_name)
                & selected_runs["target_display"].eq(target)
            ].sort_values("compare_seed")
            available = tuple(rows["compare_seed"].astype(int).tolist())
            missing = tuple(sorted(expected - set(available)))
            values = tuple(rows["metric_value"].astype(float).tolist())
            selected_batches = {
                int(row["compare_seed"]): int(row["selected_batch_size"])
                for _, row in rows.iterrows()
                if pd.notna(row["selected_batch_size"])
            }
            unique_batches = set(selected_batches.values())
            if not selected_batches:
                batch_size: int | dict[int, int] | float = np.nan
            elif len(unique_batches) == 1:
                batch_size = next(iter(unique_batches))
            else:
                batch_size = selected_batches
            records.append({
                "data_split": split_name,
                "target": target,
                "mean": float(np.mean(values)) if values else np.nan,
                "sd": float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan,
                "n": len(values),
                "available_seeds": available,
                "missing_seeds": missing,
                "seed_values": values,
                "batch_size": batch_size,
            })
            if missing:
                message = f"Missing seeds for {target} ({split_name}): {missing}"
                if require_complete:
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
    return pd.DataFrame(records)


def build_performance_metric_table(
    repeated_runs: Mapping[str, pd.DataFrame],
    *,
    point_runs: Mapping[str, pd.DataFrame] | None,
    split_name: str,
    targets: Sequence[str],
    metrics: Mapping[str, str],
    directions: Mapping[str, str],
    model_order: Sequence[str],
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    decimals: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build display, LaTeX-ready, and numeric audit tables for one split."""
    if split_name not in DEFAULT_SPLITS:
        raise ValueError(f"Unknown split: {split_name}")
    if set(metrics) != set(directions):
        raise ValueError("metrics and directions must contain the same metric keys.")
    invalid_directions = set(directions.values()) - {"max", "min"}
    if invalid_directions:
        raise ValueError(f"Invalid metric directions: {sorted(invalid_directions)}")

    point_runs = point_runs or {}
    expected = {int(seed) for seed in expected_seeds}
    records: list[dict] = []
    for target in targets:
        for model_name in model_order:
            if model_name in repeated_runs:
                rows = repeated_runs[model_name]
                rows = rows[
                    rows["data_split_normalized"].eq(split_name)
                    & rows["target_display"].eq(target)
                ].sort_values("compare_seed")
                available = set(rows["compare_seed"].astype(int).tolist())
                if available != expected:
                    raise ValueError(
                        f"{model_name}, {target}, {split_name}: expected seeds "
                        f"{sorted(expected)}, found {sorted(available)}."
                    )
                source_type = "mean_sd"
            elif model_name in point_runs:
                rows = point_runs[model_name]
                rows = rows[
                    rows["data_split_normalized"].eq(split_name)
                    & rows["target_display"].eq(target)
                ]
                if len(rows) != 1:
                    raise ValueError(
                        f"{model_name}, {target}, {split_name}: expected one point "
                        f"estimate, found {len(rows)}."
                    )
                source_type = "single_run"
            else:
                continue

            for metric_name, metric_label in metrics.items():
                values = pd.to_numeric(
                    rows[metric_name].astype(str).str.strip("'"), errors="coerce"
                ).dropna()
                if len(values) != len(rows):
                    raise ValueError(
                        f"Missing {metric_name} for {model_name}, {target}, {split_name}."
                    )
                records.append({
                    "Dataset": target,
                    "Model": model_name,
                    "Split": split_name,
                    "Metric": metric_label,
                    "metric_key": metric_name,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "n": int(len(values)),
                    "result_type": source_type,
                })

    audit = pd.DataFrame(records)
    if audit.empty:
        raise ValueError(f"No performance results found for {split_name}.")

    best_lookup: dict[tuple[str, str], float] = {}
    for (target, metric_key), group in audit.groupby(["Dataset", "metric_key"]):
        values = group["mean"]
        best_lookup[(target, metric_key)] = (
            float(values.max()) if directions[metric_key] == "max" else float(values.min())
        )

    display_rows: list[dict] = []
    latex_rows: list[dict] = []
    for target in targets:
        for model_name in model_order:
            model_rows = audit[
                audit["Dataset"].eq(target) & audit["Model"].eq(model_name)
            ]
            if model_rows.empty:
                continue
            display_row = {"Dataset": target, "Model": model_name}
            latex_row = {"Dataset": target, "Model": model_name}
            for metric_key, metric_label in metrics.items():
                result = model_rows[model_rows["metric_key"].eq(metric_key)].iloc[0]
                if result["result_type"] == "mean_sd":
                    text = (
                        f"{result['mean']:.{decimals}f} ± "
                        f"{result['sd']:.{decimals}f}"
                    )
                else:
                    text = f"{result['mean']:.{decimals}f}"
                display_row[metric_label] = text
                if " ± " in text:
                    mean_text, sd_text = text.split(" ± ", maxsplit=1)
                    latex_text = (
                        rf"\makecell[tc]{{{mean_text}\\$\pm$ {sd_text}}}"
                    )
                else:
                    latex_text = rf"\makecell[tc]{{{text}}}"
                is_best = np.isclose(
                    result["mean"], best_lookup[(target, metric_key)],
                    rtol=0.0, atol=10 ** (-(decimals + 2)),
                )
                latex_row[metric_label] = (
                    f"\\textbf{{{latex_text}}}" if is_best else latex_text
                )
            display_rows.append(display_row)
            latex_rows.append(latex_row)

    return pd.DataFrame(display_rows), pd.DataFrame(latex_rows), audit


def plot_metric_by_split(
    seed_summaries: Mapping[str, pd.DataFrame],
    *,
    metric_label: str,
    targets: Sequence[str],
    random_point_estimates: Mapping[str, Mapping[str, float]] | None = None,
    point_estimates_by_split: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ] | None = None,
    model_order: Sequence[str] = (
        "ResNet18(w/ Aug)", "ImageMol", "Fingerprint-based (ECFP6)",
        "Fingerprint-based (ECFP6 + descriptors)", "D-MPNN", "ChemAutoVision",
    ),
    model_colors: Mapping[str, str] | None = None,
    y_limits: tuple[float, float] | None = None,
    scaffold_split_seed: int = 42,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Draw random/scaffold panels side by side in one figure."""
    colors = {
        "ResNet18(w/ Aug)": "#4D4D4D",
        "ImageMol": "#4C72B0",
        "Fingerprint-based (ECFP6)": "#79B98A",
        "Fingerprint-based (ECFP6 + descriptors)": "#3B7F5A",
        "D-MPNN": "#CCB974",
        "ChemAutoVision": "#E24A33",
    }
    if model_colors is not None:
        colors.update(model_colors)
    point_estimates = dict(point_estimates_by_split or {})
    if random_point_estimates:
        point_estimates.setdefault("random", {}).update(random_point_estimates)
    expected_n = len(expected_seeds)
    seed_text = ", ".join(str(seed) for seed in expected_seeds)

    fig, axes = plt.subplots(
        1,
        len(DEFAULT_SPLITS),
        figsize=(20, 8),
        sharey=True,
    )
    axes_by_split = dict(zip(DEFAULT_SPLITS, np.atleast_1d(axes)))
    repeated_counts: list[int] = []
    has_point_estimates = False

    for split_index, split_name in enumerate(DEFAULT_SPLITS):
        ax = axes_by_split[split_name]
        prepared_summaries = {}
        for model_name, summary in seed_summaries.items():
            split_summary = (
                summary[summary["data_split"].eq(split_name)]
                .set_index("target")
                .reindex(targets)
            )
            if model_name == "D-MPNN":
                complete = split_summary["n"].eq(expected_n)
                split_summary.loc[~complete, ["mean", "sd"]] = np.nan
            if split_summary["mean"].notna().any():
                prepared_summaries[model_name] = split_summary

        model_names = [
            model_name
            for model_name in model_order
            if model_name in prepared_summaries
            or model_name in point_estimates.get(split_name, {})
        ]
        if not model_names:
            warnings.warn(f"No plottable results for {split_name}.", stacklevel=2)
            continue

        x = np.arange(len(targets), dtype=float)
        width = min(0.7 / len(model_names), 0.22)

        for model_index, model_name in enumerate(model_names):
            offset = (model_index - (len(model_names) - 1) / 2) * width
            positions = x + offset
            is_repeated = model_name in prepared_summaries
            if is_repeated:
                model_summary = prepared_summaries[model_name]
                values = model_summary["mean"].to_numpy(dtype=float)
                raw_sd = model_summary["sd"].to_numpy(dtype=float)
                errors = np.nan_to_num(raw_sd, nan=0.0)
                counts = model_summary["n"].fillna(0).astype(int).tolist()
                repeated_counts.extend(count for count in counts if count > 0)
            else:
                values = np.array(
                    [
                        float(point_estimates[split_name][model_name][target])
                        for target in targets
                    ],
                    dtype=float,
                )
                errors = np.zeros_like(values)

            bars = ax.bar(
                positions,
                values,
                width,
                yerr=errors if is_repeated else None,
                capsize=5 if is_repeated else 0,
                error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2},
                label=model_name,
                color=colors.get(model_name, "#777777"),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )
            for bar, value, error in zip(bars, values, errors):
                if not np.isfinite(value):
                    continue
                ax.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value + error),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        split_title = "Random split"
        if split_name == "balanced_scaffold":
            split_title = "Balanced Scaffold split"
        ax.set_title(f"{metric_label} — {split_title}")
        ax.set_ylabel(metric_label if split_index == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        has_point_estimates = has_point_estimates or bool(
            point_estimates.get(split_name)
        )

    legend_entries = {}
    for ax in axes_by_split.values():
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles))
    ordered_labels = [
        model_name for model_name in model_order if model_name in legend_entries
    ]
    fig.legend(
        [legend_entries[label] for label in ordered_labels],
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    if repeated_counts and all(count == expected_n for count in repeated_counts):
        caption = (
            f"Repeated-model bars show the mean over n={expected_n} seeds "
            f"({seed_text}); error bars indicate ±1 sample SD."
        )
    else:
        caption = (
            f"Repeated-model bars target n={expected_n} seeds ({seed_text}) and show "
            "the mean ±1 sample SD of currently available runs; missing runs are pending."
        )
    if has_point_estimates:
        caption += " Models without error bars are single-run point estimates."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    plt.show()
    return fig, axes_by_split
