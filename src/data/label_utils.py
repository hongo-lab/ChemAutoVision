"""Build derived binary labels from continuous molecular-property targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TERTILE_EXTREMES = "tertile_extremes"
QUARTILE_EXTREMES = "quartile_extremes"
LABEL_SCHEMES = (TERTILE_EXTREMES, QUARTILE_EXTREMES)

DERIVED_CLASSIFICATION_DIRECTIONS = {
    "FreeSolv": "bottom",
    "ESOL": "top",
    "Lipo": "top",
}

_SCHEME_QUANTILES = {
    TERTILE_EXTREMES: (1 / 3, 2 / 3),
    QUARTILE_EXTREMES: (0.25, 0.75),
}


@dataclass(frozen=True)
class QuantileThresholds:
    lower: float
    upper: float
    lower_quantile: float
    upper_quantile: float
    direction: str


def fit_quantile_thresholds(
    train_values: pd.Series,
    task_name: str,
    label_scheme: str,
) -> QuantileThresholds:
    """Fit label thresholds using training targets only."""
    if task_name not in DERIVED_CLASSIFICATION_DIRECTIONS:
        raise ValueError(f"derived classification is not supported for {task_name!r}")
    if label_scheme not in _SCHEME_QUANTILES:
        raise ValueError(f"unsupported label scheme: {label_scheme!r}")

    numeric = pd.to_numeric(train_values, errors="raise")
    if numeric.empty:
        raise ValueError("cannot fit label thresholds from an empty training fold")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("training targets must be finite and non-missing")

    lower_quantile, upper_quantile = _SCHEME_QUANTILES[label_scheme]
    lower = float(numeric.quantile(lower_quantile))
    upper = float(numeric.quantile(upper_quantile))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(
            "quantile thresholds must be finite and strictly ordered; "
            f"found lower={lower}, upper={upper}"
        )

    return QuantileThresholds(
        lower=lower,
        upper=upper,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        direction=DERIVED_CLASSIFICATION_DIRECTIONS[task_name],
    )


def apply_quantile_labels(
    values: pd.Series, thresholds: QuantileThresholds
) -> pd.Series:
    """Apply fitted thresholds, using -1 for the excluded middle band."""
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("targets must be finite and non-missing")

    labels = pd.Series(-1, index=values.index, dtype="int8")
    if thresholds.direction == "top":
        labels.loc[numeric <= thresholds.lower] = 0
        labels.loc[numeric >= thresholds.upper] = 1
    elif thresholds.direction == "bottom":
        labels.loc[numeric <= thresholds.lower] = 1
        labels.loc[numeric >= thresholds.upper] = 0
    else:
        raise ValueError(f"unsupported label direction: {thresholds.direction!r}")
    return labels


def build_quantile_classification(
    df: pd.DataFrame,
    task_name: str,
    label_scheme: str,
) -> tuple[pd.DataFrame, QuantileThresholds]:
    """Fit on ``group == 'train'`` and label every fold without re-splitting."""
    required = {"group", task_name}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing columns for derived classification: {missing}")

    thresholds = fit_quantile_thresholds(
        df.loc[df["group"] == "train", task_name], task_name, label_scheme
    )
    result = df.copy()
    continuous_column = f"{task_name}_continuous"
    result[continuous_column] = result[task_name]
    result[task_name] = apply_quantile_labels(result[continuous_column], thresholds)
    return result, thresholds
