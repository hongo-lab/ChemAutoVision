"""Chemprop 1.x training loop with patience-based early stopping.

This module mirrors Chemprop's ``run_training`` flow so the installed
Chemprop version does not need to be modified. The only behavioral change is
that each ensemble member stops after the configured number of consecutive
non-improving validation epochs.
"""

import json
from logging import Logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from callbacks import GraphEarlyStopping
from training.early_stopping import (
    restore_best_checkpoint,
    update_early_stopping_and_save,
)
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import (
    MoleculeDataLoader,
    MoleculeDataset,
    get_class_sizes,
    get_data,
    set_cache_graph,
    split_data,
)
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.spectra_utils import load_phase_mask, normalize_spectra
from chemprop.train.evaluate import evaluate, evaluate_predictions
from chemprop.train.loss_functions import get_loss_func
from chemprop.train.predict import predict
from chemprop.train.train import train
from chemprop.utils import (
    build_lr_scheduler,
    build_optimizer,
    load_checkpoint,
    load_frzn_model,
    makedirs,
    multitask_mean,
    save_checkpoint,
    save_smiles_splits,
)


def run_training_with_early_stopping(
    args: TrainArgs,
    data: MoleculeDataset,
    logger: Logger = None,
    *,
    patience: int,
    min_delta: float,
) -> Dict[str, List[float]]:
    """Train Chemprop models and stop on a validation-score plateau."""
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    torch.manual_seed(args.pytorch_seed)

    debug(f"Splitting data with seed {args.seed}")
    if args.separate_test_path:
        test_data = get_data(
            path=args.separate_test_path,
            args=args,
            features_path=args.separate_test_features_path,
            atom_descriptors_path=args.separate_test_atom_descriptors_path,
            bond_features_path=args.separate_test_bond_features_path,
            phase_features_path=args.separate_test_phase_features_path,
            smiles_columns=args.smiles_columns,
            loss_function=args.loss_function,
            logger=logger,
        )
    if args.separate_val_path:
        val_data = get_data(
            path=args.separate_val_path,
            args=args,
            features_path=args.separate_val_features_path,
            atom_descriptors_path=args.separate_val_atom_descriptors_path,
            bond_features_path=args.separate_val_bond_features_path,
            phase_features_path=args.separate_val_phase_features_path,
            smiles_columns=args.smiles_columns,
            loss_function=args.loss_function,
            logger=logger,
        )

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
    else:
        train_data, val_data, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )

    if args.dataset_type == "classification":
        class_sizes = get_class_sizes(data)
        debug("Class sizes")
        for i, task_class_sizes in enumerate(class_sizes):
            debug(
                f'{args.task_names[i]} '
                f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}'
            )
        args.train_class_sizes = get_class_sizes(train_data, proportion=False)

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_atom_descriptors=True
        )
        val_data.normalize_features(
            atom_descriptor_scaler, scale_atom_descriptors=True
        )
        test_data.normalize_features(
            atom_descriptor_scaler, scale_atom_descriptors=True
        )
    else:
        atom_descriptor_scaler = None

    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_bond_features=True
        )
        val_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
        test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None

    args.train_data_size = len(train_data)
    debug(
        f"Total size = {len(data):,} | train size = {len(train_data):,} | "
        f"val size = {len(val_data):,} | test size = {len(test_data):,}"
    )
    if len(val_data) == 0:
        raise ValueError("The validation data split is empty; early stopping requires validation data.")
    empty_test_set = len(test_data) == 0
    if empty_test_set:
        debug("The test data split is empty. Test metrics will be NaN.")

    if args.dataset_type == "regression":
        debug("Fitting scaler")
        scaler = train_data.normalize_targets()
        args.spectra_phase_mask = None
    elif args.dataset_type == "spectra":
        args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
        for dataset in [train_data, test_data, val_data]:
            data_targets = normalize_spectra(
                spectra=dataset.targets(),
                phase_features=dataset.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=None,
                threshold=args.spectra_target_floor,
            )
            dataset.set_targets(data_targets)
        scaler = None
    else:
        args.spectra_phase_mask = None
        scaler = None

    loss_func = get_loss_func(args)
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == "multiclass":
        sum_test_preds = np.zeros(
            (len(test_smiles), args.num_tasks, args.multiclass_num_classes)
        )
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed,
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data, batch_size=args.batch_size, num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data, batch_size=args.batch_size, num_workers=num_workers
    )
    if args.class_balance:
        debug(f"With class_balance, effective train size = {train_data_loader.iter_size:,}")

    early_stopping_results = []
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except TypeError:
            writer = SummaryWriter(logdir=save_dir)

        if args.checkpoint_paths is not None:
            debug(f"Loading model {model_idx} from {args.checkpoint_paths[model_idx]}")
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f"Building model {model_idx}")
            model = MoleculeModel(args)

        if args.checkpoint_frzn is not None:
            debug(f"Loading and freezing parameters from {args.checkpoint_frzn}.")
            model = load_frzn_model(
                model=model,
                path=args.checkpoint_frzn,
                current_args=args,
                logger=logger,
            )

        debug(model)
        if args.checkpoint_frzn is not None:
            debug(f"Number of unfrozen parameters = {param_count(model):,}")
            debug(f"Total number of parameters = {param_count_all(model):,}")
        else:
            debug(f"Number of parameters = {param_count_all(model):,}")

        if args.cuda:
            debug("Moving model to cuda")
            model = model.to(args.device)

        checkpoint_path = os.path.join(save_dir, MODEL_FILE_NAME)
        save_checkpoint(
            checkpoint_path,
            model,
            scaler,
            features_scaler,
            atom_descriptor_scaler,
            bond_feature_scaler,
            args,
        )
        optimizer = build_optimizer(model, args)
        scheduler = build_lr_scheduler(optimizer, args)

        early_stopping = GraphEarlyStopping(
            patience=patience,
            mode="min" if args.minimize_score else "max",
            min_delta=min_delta,
        )
        n_iter = 0
        for epoch in trange(args.epochs):
            debug(f"Epoch {epoch}")
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger,
            )
            for metric, scores in val_scores.items():
                mean_val_score = multitask_mean(scores, metric=metric)
                debug(f"Validation {metric} = {mean_val_score:.6f}")
                writer.add_scalar(f"validation_{metric}", mean_val_score, n_iter)
                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f"Validation {task_name} {metric} = {val_score:.6f}")
                        writer.add_scalar(
                            f"validation_{task_name}_{metric}", val_score, n_iter
                        )

            monitored_score = multitask_mean(
                val_scores[args.metric], metric=args.metric
            )
            def save_best_checkpoint() -> None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    scaler,
                    features_scaler,
                    atom_descriptor_scaler,
                    bond_feature_scaler,
                    args,
                )

            should_stop = update_early_stopping_and_save(
                early_stopping=early_stopping,
                current_score=monitored_score,
                epoch=epoch,
                save_best=save_best_checkpoint,
            )

            if should_stop:
                info(
                    f"Model {model_idx} early stopping on epoch {epoch}: "
                    f"validation {args.metric} did not improve by more than "
                    f"{min_delta} for {patience} epochs"
                )
                break

        best_score = early_stopping.best_score
        best_epoch = early_stopping.best_epoch
        epochs_ran = epoch + 1 if args.epochs > 0 else 0
        early_stopping_results.append(
            {
                "model_idx": model_idx,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "epochs_ran": epochs_ran,
                "stopped_early": early_stopping.should_stop,
                "stopped_epoch": early_stopping.stopped_epoch,
                "checkpoint_path": checkpoint_path,
            }
        )

        if best_score is None:
            info(f"Model {model_idx} did not produce a finite validation {args.metric}.")
        else:
            info(
                f"Model {model_idx} best validation {args.metric} = "
                f"{best_score:.6f} on epoch {best_epoch}"
            )

        model = restore_best_checkpoint(
            checkpoint_path,
            lambda path: load_checkpoint(path, device=args.device, logger=logger),
        )
        if empty_test_set:
            info(f"Model {model_idx} provided with no test set; skipping test evaluation.")
        else:
            test_preds = predict(
                model=model, data_loader=test_data_loader, scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                gt_targets=test_data.gt_targets(),
                lt_targets=test_data.lt_targets(),
                logger=logger,
            )
            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                info(f"Model {model_idx} test {metric} = {avg_test_score:.6f}")
                writer.add_scalar(f"test_{metric}", avg_test_score, 0)
                if args.show_individual_scores and args.dataset_type != "spectra":
                    for task_name, test_score in zip(args.task_names, scores):
                        info(f"Model {model_idx} test {task_name} {metric} = {test_score:.6f}")
                        writer.add_scalar(
                            f"test_{task_name}_{metric}", test_score, n_iter
                        )
        writer.close()

    args.early_stopping_results = early_stopping_results

    if empty_test_set:
        ensemble_scores = {
            metric: [np.nan for _ in args.task_names] for metric in args.metrics
        }
    else:
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            gt_targets=test_data.gt_targets(),
            lt_targets=test_data.lt_targets(),
            logger=logger,
        )

    for metric, scores in ensemble_scores.items():
        mean_ensemble_test_score = multitask_mean(scores, metric=metric)
        info(f"Ensemble test {metric} = {mean_ensemble_test_score:.6f}")
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f"Ensemble test {task_name} {metric} = {ensemble_score:.6f}")

    with open(os.path.join(args.save_dir, "test_scores.json"), "w") as file:
        json.dump(ensemble_scores, file, indent=4, sort_keys=True)

    with open(os.path.join(args.save_dir, "early_stopping.json"), "w") as file:
        json.dump(early_stopping_results, file, indent=4)

    if args.save_preds and not empty_test_set:
        test_preds_dataframe = pd.DataFrame(data={"smiles": test_data.smiles()})
        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]
        test_preds_dataframe.to_csv(
            os.path.join(args.save_dir, "test_preds.csv"), index=False
        )

    return ensemble_scores
