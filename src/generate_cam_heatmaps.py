"""Generate GradCAM, ScoreCAM, or LayerCAM heatmaps for split datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset_registry import dataset_name_for_task, target_column_for_task
from model_explanation.cam_generation import (
    generate_cam_heatmap,
    load_image_model,
    prepare_cam_models,
    resolve_target_class,
    validate_heatmap,
)
from model_explanation.cam_io import (
    atomic_save_npy,
    heatmap_path,
    preprocess_image,
    resolve_image_path,
)
from model_explanation.cam_tracking import log_cam_run
from settings import IMG_SIZE
from utils.split import make_split_csv_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source_run_id", required=True)
    parser.add_argument("--model_family", choices=["autokeras", "keras"], required=True)
    parser.add_argument(
        "--task_type", choices=["classification", "regression"], required=True
    )
    parser.add_argument(
        "--cam_method",
        choices=["gradcam", "scorecam", "layercam"],
        required=True,
    )
    parser.add_argument(
        "--target_class",
        choices=["positive", "negative", "predicted", "ground_truth"],
        default="positive",
        help="classification score explained by CAM; ignored for regression",
    )
    parser.add_argument(
        "--subset", choices=["all", "train", "val", "test"], default="all"
    )
    parser.add_argument(
        "--split_type", choices=["random", "balanced_scaffold"], default="random"
    )
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument(
        "--label_scheme",
        choices=["tertile_extremes", "quartile_extremes"],
        default=None,
    )
    parser.add_argument("--data_dir", type=Path, default=Path("../data"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--scorecam_batch_size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tracking_uri", default=None)
    parser.add_argument("--no_mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_type == "regression" and args.target_class != "positive":
        raise ValueError("--target_class is only meaningful for classification")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset_name = dataset_name_for_task(args.task_name)
    target_column = target_column_for_task(args.task_name)
    csv_paths = make_split_csv_paths(
        args.data_dir,
        dataset_name,
        args.split_type,
        args.split_seed,
        args.label_scheme,
    )
    subsets = ["train", "val", "test"] if args.subset == "all" else [args.subset]
    frames: dict[str, pd.DataFrame] = {}
    image_paths: dict[str, list[Path]] = {}
    for subset in subsets:
        csv_path = csv_paths[subset]
        if not csv_path.is_file():
            raise FileNotFoundError(f"split CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        required = {"smiles", "img_path", target_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        frames[subset] = frame
        image_paths[subset] = [
            resolve_image_path(value, csv_path) for value in frame["img_path"]
        ]

    model = load_image_model(args.model_path, args.model_family)
    models = prepare_cam_models(model, args.model_family, input_shape=tuple(IMG_SIZE))
    generated = skipped = 0
    heatmap_values: list[float] = []
    output_directories: set[str] = set()
    heatmap_target = "output" if args.task_type == "regression" else args.target_class
    for subset in subsets:
        frame = frames[subset]
        for row_number, ((_, row), image_path) in enumerate(
            zip(frame.iterrows(), image_paths[subset])
        ):
            output_path = heatmap_path(
                image_path, args.source_run_id, args.cam_method, heatmap_target
            )
            output_directories.add(str(output_path.parent))
            if output_path.is_file() and not args.overwrite:
                validate_heatmap(
                    np.load(output_path, allow_pickle=False),
                    expected_shape=(IMG_SIZE[0], IMG_SIZE[1]),
                )
                skipped += 1
                continue
            inputs = preprocess_image(image_path)
            target_class = None
            if args.task_type == "classification":
                positive_probability = float(
                    np.asarray(models.prediction_model(inputs, training=False)).reshape(
                        -1
                    )[0]
                )
                target_class = resolve_target_class(
                    args.target_class,
                    positive_probability,
                    row[target_column],
                )
            heatmap = generate_cam_heatmap(
                models,
                inputs,
                method=args.cam_method,
                task_type=args.task_type,
                target_class=target_class,
                scorecam_batch_size=args.scorecam_batch_size,
            )
            atomic_save_npy(output_path, heatmap)
            generated += 1
            heatmap_values.append(float(heatmap.mean()))
            if (row_number + 1) % 100 == 0:
                print(f"{subset}: processed {row_number + 1}/{len(frame)}")

    mean_heatmap = float(np.mean(heatmap_values)) if heatmap_values else None
    print(
        f"CAM generation complete: generated={generated}, skipped={skipped}, "
        f"mean_heatmap={mean_heatmap if mean_heatmap is not None else float('nan'):.6f}"
    )
    if not args.no_mlflow:
        run_id = log_cam_run(
            experiment_name="cam_generation",
            run_name=f"{args.cam_method}_{args.task_name}_{heatmap_target}",
            params={
                "source_run_id": args.source_run_id,
                "dataset": dataset_name,
                "target": target_column,
                "task_type": args.task_type,
                "split_type": args.split_type,
                "split_seed": args.split_seed,
                "label_scheme": args.label_scheme,
                "cam_method": args.cam_method,
                "target_class": heatmap_target,
                "subset": args.subset,
                "scorecam_batch_size": args.scorecam_batch_size,
            },
            metrics={
                "n_generated": generated,
                "n_skipped": skipped,
                **(
                    {"mean_heatmap_value": mean_heatmap}
                    if mean_heatmap is not None
                    else {}
                ),
            },
            tags={
                "run_type": "cam_generation",
                "model_family": args.model_family,
                "heatmap_dirs": "|".join(sorted(output_directories)),
            },
            tracking_uri=args.tracking_uri,
        )
        print(f"CAM generation MLflow run: {run_id}")


if __name__ == "__main__":
    main()
