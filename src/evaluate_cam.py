"""Evaluate CAM heatmaps with deletion/insertion faithfulness or top-k IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset_registry import dataset_name_for_task, target_column_for_task
from model_explanation.cam_faithfulness import evaluate_faithfulness
from model_explanation.cam_generation import (
    load_image_model,
    prepare_cam_models,
    resolve_target_class,
    validate_heatmap,
)
from model_explanation.cam_io import (
    heatmap_path,
    load_image_uint8,
    resolve_image_path,
)
from model_explanation.cam_tracking import log_cam_run
from model_explanation.cam_iou import evaluate_iou
from settings import IMG_SIZE
from utils.split import make_split_csv_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_type", choices=["faithfulness", "iou"], required=True)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--source_run_id", default=None)
    parser.add_argument("--cam_method", choices=["gradcam", "scorecam", "layercam"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_family", choices=["autokeras", "keras"])
    parser.add_argument("--task_type", choices=["classification", "regression"])
    parser.add_argument(
        "--target_class",
        choices=["positive", "negative", "predicted", "ground_truth"],
        default="positive",
    )
    parser.add_argument("--subset", choices=["train", "val", "test"], default="val")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--heatmap_dirs", nargs="+", type=Path)
    parser.add_argument("--dir_labels", nargs="+")
    parser.add_argument("--k_values", nargs="+", type=float, default=[5.0, 10.0, 20.0])
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
    parser.add_argument("--output_dir", type=Path, default=Path("../results/cam"))
    parser.add_argument("--tracking_uri", default=None)
    parser.add_argument("--no_mlflow", action="store_true")
    return parser.parse_args()


def _require(args: argparse.Namespace, names: list[str]) -> None:
    missing = [name for name in names if getattr(args, name) in (None, [])]
    if missing:
        raise ValueError(
            f"--eval_type {args.eval_type} requires: "
            + ", ".join(f"--{name}" for name in missing)
        )


def run_faithfulness(args: argparse.Namespace) -> None:
    _require(
        args,
        [
            "task_name",
            "source_run_id",
            "cam_method",
            "model_path",
            "model_family",
            "task_type",
        ],
    )
    dataset_name = dataset_name_for_task(args.task_name)
    target_column = target_column_for_task(args.task_name)
    csv_path = make_split_csv_paths(
        args.data_dir,
        dataset_name,
        args.split_type,
        args.split_seed,
        args.label_scheme,
    )[args.subset]
    if not csv_path.is_file():
        raise FileNotFoundError(f"split CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"img_path", target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    image_paths = [resolve_image_path(value, csv_path) for value in frame["img_path"]]
    heatmap_target = "output" if args.task_type == "regression" else args.target_class
    heatmap_paths = [
        heatmap_path(path, args.source_run_id, args.cam_method, heatmap_target)
        for path in image_paths
    ]
    missing_heatmaps = [str(path) for path in heatmap_paths if not path.is_file()]
    if missing_heatmaps:
        raise FileNotFoundError(
            "faithfulness requires complete heatmaps; missing: "
            + ", ".join(missing_heatmaps[:20])
        )

    model = load_image_model(args.model_path, args.model_family)
    models = prepare_cam_models(model, args.model_family, tuple(IMG_SIZE))
    records: list[dict[str, object]] = []
    deletion_scores: list[np.ndarray] = []
    insertion_scores: list[np.ndarray] = []
    ratios = None
    for row_number, ((_, row), image_path, cam_path) in enumerate(
        zip(frame.iterrows(), image_paths, heatmap_paths)
    ):
        image = load_image_uint8(image_path)
        heatmap = validate_heatmap(
            np.load(cam_path, allow_pickle=False), expected_shape=image.shape[:2]
        )
        target_class = None
        if args.task_type == "classification":
            probability = float(
                np.asarray(
                    models.prediction_model(image[None] / 255.0, training=False)
                ).reshape(-1)[0]
            )
            target_class = resolve_target_class(
                args.target_class, probability, row[target_column]
            )
        result = evaluate_faithfulness(
            models.prediction_model,
            image,
            heatmap,
            steps=args.steps,
            task_type=args.task_type,
            target_class=target_class,
        )
        records.append(
            {
                "row_number": row_number,
                "image_key": image_path.stem,
                "target_class": target_class,
                "deletion_auc": result["deletion_auc"],
                "insertion_auc": result["insertion_auc"],
            }
        )
        deletion_scores.append(result["deletion_scores"])
        insertion_scores.append(result["insertion_scores"])
        ratios = result["deletion_ks"]

    output = args.output_dir / "faithfulness" / args.source_run_id / args.cam_method
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(output / f"{args.subset}_results.csv", index=False)
    np.save(output / f"{args.subset}_ratios.npy", ratios, allow_pickle=False)
    np.save(
        output / f"{args.subset}_deletion_scores.npy",
        np.stack(deletion_scores),
        allow_pickle=False,
    )
    if not args.no_mlflow:
        run_id = log_cam_run(
            experiment_name="cam_evaluation",
            run_name=f"faithfulness_{args.cam_method}_{args.task_name}_{args.subset}",
            params={
                "source_run_id": args.source_run_id,
                "dataset": dataset_name,
                "target": target_column,
                "task_type": args.task_type,
                "split_type": args.split_type,
                "split_seed": args.split_seed,
                "label_scheme": args.label_scheme,
                "subset": args.subset,
                "cam_method": args.cam_method,
                "target_class": heatmap_target,
                "steps": args.steps,
            },
            metrics={
                "mean_deletion_auc": float(results.deletion_auc.mean()),
                "mean_insertion_auc": float(results.insertion_auc.mean()),
                "n_molecules": len(results),
            },
            tags={"run_type": "cam_faithfulness"},
            artifacts=output.glob(f"{args.subset}_*"),
            tracking_uri=args.tracking_uri,
        )
        print(f"CAM evaluation MLflow run: {run_id}")
    np.save(
        output / f"{args.subset}_insertion_scores.npy",
        np.stack(insertion_scores),
        allow_pickle=False,
    )
    print(
        f"mean deletion AUC={results.deletion_auc.mean():.6f}, "
        f"mean insertion AUC={results.insertion_auc.mean():.6f}"
    )


def run_iou(args: argparse.Namespace) -> None:
    _require(args, ["heatmap_dirs"])
    directories = [path.resolve() for path in args.heatmap_dirs]
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"heatmap directory not found: {directory}")
    labels = args.dir_labels or [directory.name for directory in directories]
    if len(labels) != len(directories):
        raise ValueError("--dir_labels must match --heatmap_dirs")
    key_sets = [
        {path.stem for path in directory.glob("*.npy")} for directory in directories
    ]
    if not key_sets:
        raise ValueError("no heatmap directories supplied")
    all_keys = set.union(*key_sets)
    common_keys = set.intersection(*key_sets)
    if common_keys != all_keys:
        missing_report = {
            labels[index]: sorted(all_keys - keys)
            for index, keys in enumerate(key_sets)
            if keys != all_keys
        }
        raise FileNotFoundError(
            "IoU requires identical heatmap sets; missing files: "
            + json.dumps(missing_report, ensure_ascii=False)
        )
    if not common_keys:
        raise ValueError("heatmap directories contain no .npy files")
    detail, summary = evaluate_iou(
        directories, labels, args.k_values, sorted(common_keys)
    )
    output = args.output_dir / "iou"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "iou_detail.csv", index=False)
    summary.to_csv(output / "iou_summary.csv", index=False)
    if not args.no_mlflow:
        metrics = {"n_molecules": len(common_keys)}
        for _, row in summary.iterrows():
            pair = str(row["pair_label"]).replace(" ", "_")
            k = f"{float(row['k']):g}".replace(".", "p")
            metrics[f"mean_iou_{pair}_k{k}"] = float(row["mean_iou"])
        run_id = log_cam_run(
            experiment_name="cam_evaluation",
            run_name="iou_" + "_".join(labels),
            params={
                "heatmap_dirs": "|".join(str(path) for path in directories),
                "dir_labels": "|".join(labels),
                "k_values": ",".join(str(value) for value in args.k_values),
            },
            metrics=metrics,
            tags={"run_type": "cam_iou"},
            artifacts=[output / "iou_detail.csv", output / "iou_summary.csv"],
            tracking_uri=args.tracking_uri,
        )
        print(f"CAM evaluation MLflow run: {run_id}")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.eval_type == "faithfulness":
        run_faithfulness(args)
    else:
        run_iou(args)


if __name__ == "__main__":
    main()
