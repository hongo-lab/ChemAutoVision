"""Generate strict, provenance-checked Chemprop atom features from CAM files."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from data.dataset_registry import dataset_name_for_task
from data.graph_features import (
    build_atom_descriptors,
    get_full_feature_names,
    heatmap_set_sha256,
    validate_heatmap_paths,
)
from model_explanation.cam_io import heatmap_path, resolve_image_path
from settings import IMG_SIZE
from utils.atom_descriptor_metadata import write_atom_descriptor_metadata
from utils.split import make_split_csv_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--source_run_id", required=True)
    parser.add_argument(
        "--task_type", choices=["classification", "regression"], required=True
    )
    parser.add_argument(
        "--cam_methods",
        nargs="+",
        choices=["gradcam", "scorecam", "layercam"],
        required=True,
    )
    parser.add_argument(
        "--target_class",
        choices=["positive", "negative", "predicted"],
        default="positive",
        help="ground_truth is intentionally forbidden to prevent target leakage",
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
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument(
        "--aggregation", choices=["mean", "max", "gaussian"], default="mean"
    )
    parser.add_argument("--include_bonds", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=Path("../data"))
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.cam_methods)) != len(args.cam_methods):
        raise ValueError("--cam_methods contains duplicates")
    if args.task_type == "regression" and args.target_class != "positive":
        raise ValueError("--target_class is only meaningful for classification")
    heatmap_target = "output" if args.task_type == "regression" else args.target_class
    dataset_name = dataset_name_for_task(args.task_name)
    csv_paths = make_split_csv_paths(
        args.data_dir,
        dataset_name,
        args.split_type,
        args.split_seed,
        args.label_scheme,
    )
    output_dir = args.output_dir or args.data_dir / "atom_descriptors"
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, tuple[pd.DataFrame, dict[str, list[Path]]]] = {}
    for subset, csv_path in csv_paths.items():
        if not csv_path.is_file():
            raise FileNotFoundError(f"split CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        missing_columns = {"smiles", "img_path"}.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing columns: {sorted(missing_columns)}"
            )
        image_paths = [
            resolve_image_path(value, csv_path) for value in frame["img_path"]
        ]
        paths_by_method = {
            method: [
                heatmap_path(image_path, args.source_run_id, method, heatmap_target)
                for image_path in image_paths
            ]
            for method in args.cam_methods
        }
        validate_heatmap_paths(
            paths_by_method,
            expected_count=len(frame),
            expected_shape=(IMG_SIZE[0], IMG_SIZE[1]),
        )
        inputs[subset] = (frame, paths_by_method)

    # Compute all subsets before writing any final artifact.
    descriptors = {
        subset: build_atom_descriptors(
            frame["smiles"].tolist(),
            paths_by_method,
            radius=args.radius,
            aggregation=args.aggregation,
            include_bonds=args.include_bonds,
            img_size=(IMG_SIZE[0], IMG_SIZE[1]),
        )
        for subset, (frame, paths_by_method) in inputs.items()
    }

    methods = "-".join(args.cam_methods)
    base = (
        f"{args.task_name}_{args.source_run_id}_{methods}_{heatmap_target}_"
        f"r{args.radius}_{args.aggregation}_atom_desc"
    )
    feature_names = get_full_feature_names(args.cam_methods)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_directory:
        staging = Path(temporary_directory)
        staged: list[tuple[Path, Path, Path, Path]] = []
        for subset, csv_path in csv_paths.items():
            staged_pkl = staging / f"{base}_{subset}.pkl"
            pd.DataFrame({"atom_features": descriptors[subset]}).to_pickle(staged_pkl)
            _, paths_by_method = inputs[subset]
            ordered_heatmaps = [
                path
                for molecule_index in range(len(descriptors[subset]))
                for method in args.cam_methods
                for path in [paths_by_method[method][molecule_index]]
            ]
            staged_meta = write_atom_descriptor_metadata(
                staged_pkl,
                csv_path,
                split_type=args.split_type,
                split_seed=args.split_seed,
                subset=subset,
                extra={
                    "source_run_id": args.source_run_id,
                    "cam_methods": args.cam_methods,
                    "target_class": heatmap_target,
                    "radius": args.radius,
                    "aggregation": args.aggregation,
                    "include_bonds": args.include_bonds,
                    "n_cam_features": len(args.cam_methods),
                    "n_total_features": len(feature_names),
                    "feature_names": feature_names,
                    "heatmap_set_sha256": heatmap_set_sha256(ordered_heatmaps),
                },
            )
            final_pkl = output_dir / staged_pkl.name
            final_meta = output_dir / final_pkl.with_suffix(".meta.json").name
            staged.append((staged_pkl, staged_meta, final_pkl, final_meta))

        for staged_pkl, staged_meta, final_pkl, final_meta in staged:
            os.replace(staged_pkl, final_pkl)
            os.replace(staged_meta, final_meta)
            print(f"saved {final_pkl}")


if __name__ == "__main__":
    main()
