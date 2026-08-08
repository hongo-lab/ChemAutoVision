import argparse
import gc
import time
from functools import partial

import chemprop.hyperparameter_optimization as chemprop_hyperopt_module
from chemprop.train import make_predictions
from chemprop.args import TrainArgs, HyperoptArgs, PredictArgs
from chemprop.data import get_data
from chemprop.features import set_extra_atom_fdim
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import tensorflow as tf
import torch
from utils.utils import list_gpu_names
import os
from recording.record_mlflow import record_exp_result
from training.graph_training import run_training_with_early_stopping
from training.early_stopping import get_best_checkpoint_path

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument("--task_name", type=str, help="the task name")
    parser.add_argument("--batch_size", type=str, help="batch_size")
    parser.add_argument("--train_csv_path", type=str, help="train csv path")
    parser.add_argument("--valid_csv_path", type=str, help="valid csv path")
    parser.add_argument("--test_csv_path", type=str, help="test csv path")
    parser.add_argument("--num_iter", type=str, help="graph hyperopt iter num")
    parser.add_argument("--epochs", type=int, default=200, help="maximum epochs")
    parser.add_argument("--features_generator", type=str, help="graph feature generator")
    parser.add_argument("--dataset_type", type=str, choices=["classification", "regression"])
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")
    parser.add_argument(
        "--min_delta",
        type=float,
        default=None,
        help="minimum validation metric change; defaults to 0.005 for classification and 0.0 for regression",
    )
    parser.add_argument("--gpu", type=str, help="gpu")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--atom_descriptors_path", type=str, default=None,
                        help="train 用の原子特徴量 pkl パス（generate_cam_graph_features.py の出力）")
    parser.add_argument("--atom_descriptors", type=str, default="feature",
                        choices=["feature", "descriptor"],
                        help="原子特徴量の注入方法: feature=メッセージパッシング前, descriptor=readout後 (デフォルト: feature)")
    parser.add_argument("--radius", type=int, default=None,
                        help="CAM原子特徴量生成時のradius [px]。指定すると atom_descriptors_path 内の"
                             "_atom_desc_ を _r{radius}_{aggregation}_atom_desc_ に置換して対応する pkl を使用する")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "max", "gaussian"],
                        help="CAM集計方法。--radius と合わせて使用（デフォルト: mean）")
    args = parser.parse_args()

    if args.patience < 0:
        parser.error("--patience must be greater than or equal to 0")
    if args.epochs < 1:
        parser.error("--epochs must be greater than or equal to 1")
    early_stopping_min_delta = (
        args.min_delta
        if args.min_delta is not None
        else (0.005 if args.dataset_type == "classification" else 0.0)
    )

    task_name = args.task_name
    base_task_name = task_name[2:] if task_name[:2] in ("t_", "q_") else task_name
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    selected_gpu_indices = list(map(int, args.gpu.split(",")))
    gpu_names = list_gpu_names()
    selected_gpu_names = [gpu_names[i] for i in selected_gpu_indices]
    print(selected_gpu_names)

    # hyper parameter tuning
    hp_save_dir = f"./graph_hyperopt/hp/dmpnn_{task_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config_save_path = f"./{hp_save_dir}/{task_name}_b{args.batch_size}_dmpnn_best_hp_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

    # atom_descriptors_path が指定された場合、_train/_val/_test のパスを導出する
    # 命名規則: {base}_atom_desc_train.pkl → {base}_atom_desc_val.pkl / _test.pkl
    atom_desc_train_path = args.atom_descriptors_path
    atom_desc_val_path = None
    atom_desc_test_path = None
    if atom_desc_train_path is not None:
        if args.radius is not None:
            atom_desc_train_path = atom_desc_train_path.replace(
                "_atom_desc_", f"_r{args.radius}_{args.aggregation}_atom_desc_"
            )
        _base = atom_desc_train_path.replace("_train.pkl", "")
        atom_desc_val_path = f"{_base}_val.pkl"
        atom_desc_test_path = f"{_base}_test.pkl"

    validation_metric = 'auc' if args.dataset_type == 'classification' else 'mse'

    hp_args = [
        '--data_path', f"../data/train_{task_name}_img.csv",
        '--separate_val_path', f"../data/val_{task_name}_img.csv",
        '--dataset_type', args.dataset_type,
        '--metric', validation_metric,
        # '--features_generator', args.features_generator,
        '--batch_size', args.batch_size,
        '--save_dir', hp_save_dir,
        '--num_iters', str(args.num_iter),
        '--config_save_path', config_save_path,
        '--epochs', str(args.epochs),
        '--smiles_columns', 'smiles',
        '--target_columns', base_task_name,
        '--gpu', args.gpu,
        '--seed', str(args.seed)
    ]
    if args.dataset_type == 'classification':
        hp_args.append('--class_balance')
    if atom_desc_train_path is not None:
        hp_args += ['--atom_descriptors_path', atom_desc_train_path,
                    '--atom_descriptors', args.atom_descriptors,
                    '--separate_val_atom_descriptors_path', atom_desc_val_path]
    
    hyperopt_args = HyperoptArgs().parse_args(hp_args)
    # Chemprop 1.x の hyperopt はモジュール変数 run_training を
    # cross_validate に渡すため、Early Stopping 対応版へ差し替える。
    chemprop_hyperopt_module.run_training = partial(
        run_training_with_early_stopping,
        patience=args.patience,
        min_delta=early_stopping_min_delta,
    )
    chemprop_hyperopt_module.hyperopt(hyperopt_args)

    # training
    model_save_dir = f"./graph_hyperopt/trained/dmpnn_{task_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 
    tr_args = [
        '--data_path', f"../data/train_{task_name}_img.csv",
        '--separate_val_path', f"../data/val_{task_name}_img.csv",
        '--separate_test_path',f"../data/test_{task_name}_img.csv",
        '--dataset_type', args.dataset_type,
        '--config_path', config_save_path,
        '--metric', validation_metric,
        # '--features_generator', args.features_generator,
        '--batch_size', args.batch_size,
        # '--features_size', '200',
        '--epochs', str(args.epochs),
        '--save_dir', model_save_dir ,
        '--smiles_columns', 'smiles',
        '--target_columns', base_task_name,
        '--gpu', args.gpu,
        '--seed', str(args.seed)
    ]
    if args.dataset_type == 'classification':
        tr_args.append('--class_balance')
    if atom_desc_train_path is not None:
        tr_args += ['--atom_descriptors_path', atom_desc_train_path,
                    '--atom_descriptors', args.atom_descriptors,
                    '--separate_val_atom_descriptors_path', atom_desc_val_path,
                    '--separate_test_atom_descriptors_path', atom_desc_test_path]
    train_args = TrainArgs().parse_args(tr_args)
    data = get_data(
        path=train_args.data_path,
        smiles_columns=train_args.smiles_columns,
        target_columns=train_args.target_columns,
        atom_descriptors_path=atom_desc_train_path,
        args=train_args
    )

    # if len(data) > 0:
    #     print(f"First data sample type: {type(data[0])}")
    #     if hasattr(data[0], '__dict__'):
    #         print(f"First data sample attributes: {vars(data[0])}")
    # breakpoint()
    # train_args.task_names = data.target_names
    if not hasattr(train_args, 'task_names') or train_args.task_names is None:
        train_args.task_names = [base_task_name] if isinstance(base_task_name, str) else base_task_name
        print(f"Set task_names to: {train_args.task_names}")

    # cross_validate.py と同等の処理: atom_features_size を train_args に保存し
    # EXTRA_ATOM_FDIM を設定する。run_training は直接呼び出すためこの初期化が必要。
    # make_predictions がチェックポイントから train_args.atom_features_size を読んで
    # set_extra_atom_fdim() を呼び直すため、ここで正しい値を保存しておかないと
    # 予測時に EXTRA_ATOM_FDIM がリセットされてしまう。
    if train_args.atom_descriptors == 'descriptor':
        train_args.atom_descriptors_size = data.atom_descriptors_size()
    elif train_args.atom_descriptors == 'feature':
        train_args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(train_args.atom_features_size)

    if atom_desc_train_path is not None:
        import json as _json
        n_cam = train_args.atom_features_size if train_args.atom_descriptors == 'feature' else 0
        meta_path = atom_desc_train_path.replace("_train.pkl", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = _json.load(f)
            if n_cam != meta["n_cam_features"]:
                raise ValueError(
                    f"[atom features] 列数不一致: pkl から読み取った n_cam={n_cam}, "
                    f"meta.json の n_cam_features={meta['n_cam_features']}. "
                    f"pkl と meta.json が対応していない可能性があります。"
                )
            print(f"\n[atom features] total={meta['n_total_features']}  "
                  f"(default={meta['n_total_features'] - n_cam}, cam={n_cam}, mode={train_args.atom_descriptors})")
            print(f"  feature_names (last 5): {meta['feature_names'][-5:]}")
        else:
            from data.graph_features import get_atom_feature_names
            default_names = get_atom_feature_names()
            print(f"\n[atom features] total={len(default_names) + n_cam}  "
                  f"(default={len(default_names)}, cam={n_cam}, mode={train_args.atom_descriptors})")
            print(f"  default (last 3): {default_names[-3:]}")
            if n_cam > 0:
                print(f"  CAM descriptors : {n_cam} column(s) from {atom_desc_train_path}")
            print(f"  (meta.json not found at {meta_path} — feature names unavailable)")

    # if features_gen == 'rdkit_2d':
    # train_args.features_size = 200

    # print(f"train_args.features_generator: {getattr(train_args, 'features_generator', 'None')}")
    # print(f"train_args.features_size: {getattr(train_args, 'features_size', 'None')}")
    # try:
    run_training_with_early_stopping(
        train_args,
        data,
        patience=args.patience,
        min_delta=early_stopping_min_delta,
    )
    # except Exception as e:
    #     print(f"Training error: {e}")
    #     print(f"Error type: {type(e)}")
    #     import traceback
    #     traceback.print_exc()
    # train_model(train_args)

    # predict
    predict_output_path = f"{model_save_dir}/prediction_{task_name}.csv"
    predict_args_list = [
        '--test_path', f"../data/test_{task_name}_img.csv",
        '--checkpoint_dir', model_save_dir,
        '--preds_path', predict_output_path,
        # '--features_generator', args.features_generator,
        '--smiles_column', 'smiles',
        # '--target_columns', task_name,
        '--gpu', args.gpu
    ]
    if atom_desc_test_path is not None:
        predict_args_list += ['--atom_descriptors_path', atom_desc_test_path,
                              '--atom_descriptors', args.atom_descriptors]
    predict_args = PredictArgs().parse_args(predict_args_list)

    make_predictions(predict_args)

    y_score = pd.read_csv(predict_output_path)[base_task_name]
    y_preds = np.where(y_score > 0.5, 1, 0)
    y_test = pd.read_csv(f"../data/test_{task_name}_img.csv")[base_task_name]

    record_exp_result(
        '576013465360263177' if args.dataset_type == "regression" else '570837897253197098',
        # '0',
        # metrics
        {
            "rmse": np.sqrt(mean_squared_error(y_test, y_score)),
            "mse": mean_squared_error(y_test, y_score),
            "mae": mean_absolute_error(y_test, y_score),
            "r2": r2_score(y_test, y_score),
        } if args.dataset_type == "regression" else {
            "acc": accuracy_score(y_test, y_preds),
            "recall": recall_score(y_test, y_preds),
            "precision": precision_score(y_test, y_preds),
            "roc_auc": roc_auc_score(y_test, y_score),
            "mcc": matthews_corrcoef(y_test, y_preds),
            "f1": f1_score(y_test, y_preds),
        },
        # params
        {
            "epochs": args.epochs,
            "patience": args.patience,
            "min_delta": early_stopping_min_delta,
            "early_stopping_metric": validation_metric,
            "best_epoch": train_args.early_stopping_results[0]["best_epoch"],
            "epochs_ran": train_args.early_stopping_results[0]["epochs_ran"],
            "stopped_early": train_args.early_stopping_results[0]["stopped_early"],
            "num_iters": args.num_iter,
            "batch_size": args.batch_size,
            "features_generator": None if not hasattr(args, 'features_generator') else args.features_generator,
            "features_size": None if not hasattr(args, 'features_size') else args.features_size,
            "atom_descriptors_path": atom_desc_train_path,
            "atom_descriptors": args.atom_descriptors if atom_desc_train_path is not None else None,
            "cam_aggregation": args.aggregation if atom_desc_train_path is not None else None,
            "cam_radius": args.radius if atom_desc_train_path is not None else None,
            "seed": args.seed,
        },
        # tags
        {
            "target": task_name,
            "explanatory_val": "graph",
            "gpu_names": selected_gpu_names,
            "model_name": 'chemprops',
            "hp_result_path": hp_save_dir,
            "model_path": get_best_checkpoint_path(train_args.early_stopping_results),
            "result_csv_path": predict_output_path,
        },
    )
