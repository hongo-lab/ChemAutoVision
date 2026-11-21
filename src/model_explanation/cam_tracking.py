"""Small MLflow adapter shared by CAM generation and evaluation CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


def log_cam_run(
    *,
    experiment_name: str,
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float | int],
    tags: Mapping[str, object] | None = None,
    artifacts: Iterable[Path] = (),
    tracking_uri: str | None = None,
) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or f"file:{REPO_ROOT / 'mlruns'}")
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=run_name
    ) as run:
        mlflow.log_params(
            {key: value for key, value in params.items() if value is not None}
        )
        mlflow.log_metrics(dict(metrics))
        if tags:
            mlflow.set_tags(
                {key: value for key, value in tags.items() if value is not None}
            )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return run.info.run_id
