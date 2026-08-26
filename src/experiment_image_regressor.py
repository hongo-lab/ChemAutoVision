import argparse
import gc
import time
from datetime import datetime
from functools import partial

import autokeras as ak
import numpy as np
import tensorflow as tf
from seed import set_seeds
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
from keras_tuner import Objective
from keras_tuner.tuners import BayesianOptimization

from data.images import create_regression_img_data
from modeling.modeling import (
    create_vgg16_model,
    create_vgg19_model,
    create_densenet121_model,
    create_densenet201_model,
    create_resnet18_model,
    create_resnet50_model,
    create_resnet152_model,
    create_efficientnetb0_model,
    create_convnextbase_model,
    create_vit_model
) 
from recording.record_mlflow import record_exp_result
import os
from settings import IMG_SIZE, IS_DOWNSCALE_BEFORE_DAUG
from utils.split import (
    BALANCED_SCAFFOLD_SPLIT_TYPE,
    make_split_prefix,
    split_method_name,
)
from utils.utils import create_result_csv, list_gpu_names
import json
import os

EXP_ID = 576013465360263177

LOSS = "mean_squared_error"
AK_SERCH_MODEL = None

create_model_functions = {
    "vgg16": create_vgg16_model,
    "vgg19": create_vgg19_model,
    "densenet121": create_densenet121_model,
    "densenet201": create_densenet201_model,
    "resnet18": create_resnet18_model,
    "resnet50": create_resnet50_model,
    "resnet152": create_resnet152_model,
    "efficientnetb0": create_efficientnetb0_model,
    "convnextbase": create_convnextbase_model,
    "vit": create_vit_model,
}

BATCH_SIZES = [8, 16, 32, 64, 128]
PARAM_GRID_VIT = {
    "patch_size": [16, 32, 64],
    "hidden_size": [256, 512, 768],
    "num_layers": [4, 8, 12],
    "num_heads": [4, 8, 12],
    "mlp_dim": [512, 1024, 2048, 3072],
    "attention_do_rate": [0.0, 0.1, 0.2],
    "do_rate": [0.0, 0.1, 0.2],
    "representation_size": [0, 128, 256, 384]
}
PARAM_GRID = {
    "lr": [5e-4, 5e-3, 5e-2, 5e-1],
}

DATA_PATHS = {
    "ESOL": "../data/ESOL_img.pkl",
    "Lipo": "../data/Lipo_img.pkl",
    "FreeSolv": "../data/FreeSolv_img.pkl",
}

def build_model_for_kerastuner(hp, is_execute_data_aug):
    lr = hp.Float(
        "lr",
        min_value=1e-5,
        max_value=1e-1,
        sampling="log"
    )

    inputs = tf.keras.layers.Input(shape=IMG_SIZE)
    x = inputs
    if is_execute_data_aug:
        rotation_factor = hp.Choice("rotation_factor", [0.0, 0.1])
        translation_factor = hp.Choice("translation_factor", [0.0, 0.1])
        zoom_factor = hp.Choice("zoom_factor", [0.0, 0.1])
        if rotation_factor > 0:
            x = tf.keras.layers.RandomRotation(rotation_factor)(x)
        if translation_factor > 0:
            x = tf.keras.layers.RandomTranslation(translation_factor, translation_factor)(x)
        if zoom_factor > 0:
            x = tf.keras.layers.RandomZoom(zoom_factor, zoom_factor)(x)

    kwargs = dict(
        input_shape=IMG_SIZE,
        loss_alg=LOSS,
        metrics=[LOSS],
        is_regression=True,
    )

    base_model = create_model_functions[args.model_name](**kwargs)

    outputs = base_model(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=LOSS,
        metrics=[LOSS]
    )
    return model

def create_callbacks():
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    ]
    
    return callbacks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument("--task_name", type=str, help="the task name")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--max_trials", type=int, default=50, help="autokeras maxtrials")
    parser.add_argument('--hp_tuning', choices=['keras_tuner'], help='Hyperparameter tuning method')
    parser.add_argument("--gpu", type=str, help="gpu")
    parser.add_argument("--model_name", choices=["autokeras", "vgg16", "vgg19", "densenet121", "densenet201", "resnet18", "resnet50", "resnet152", "efficientnetb0", "convnextbase", "vit"], type=str, help="model_name")
    parser.add_argument('--execute_data_aug', action='store_true', help='execute data augmentation')
    parser.add_argument("--seed", type=int, default=99, help="seed")
    parser.add_argument(
        "--split_type",
        choices=["random", BALANCED_SCAFFOLD_SPLIT_TYPE],
        default="random",
        help="data split strategy (generate_data.py と一致させる)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="seed for balanced scaffold assignment (required for balanced_scaffold)",
    )

    args = parser.parse_args()

    set_seeds(42)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    gpu_names = list_gpu_names()
    print(gpu_names)

    task_name = args.task_name
    batch_size = int(args.batch_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if task_name not in DATA_PATHS:
        raise ValueError(f"Invalid task name: {task_name}")
    
    data_path = DATA_PATHS[task_name]
    split_prefix = make_split_prefix(args.split_type, args.split_seed)
    split_method = split_method_name(args.split_type)
    split_run_name = (
        f"{split_method}_s{args.split_seed}"
        if args.split_type == BALANCED_SCAFFOLD_SPLIT_TYPE
        else split_method
    )
    data_aug_suffix = "" if args.execute_data_aug else "_noaug"
    data_path = data_path.replace("../data/", f"../data/{split_prefix}")

    train_ds, val_ds, test_ds = create_regression_img_data(
        data_path,
        task_name,
        0.2,
        IMG_SIZE
    )

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_model_path = (
        f"../models/{task_name}_regression_{args.model_name}_{split_run_name}"
        f"{data_aug_suffix}_{timestamp}.h5"
    )
    call_backs = create_callbacks()
    if args.model_name == "autokeras":

        input_node = ak.ImageInput()
        x = input_node

        if args.execute_data_aug:
            x = ak.ImageAugmentation(
                horizontal_flip=False,
                vertical_flip=False,
                contrast_factor=0.0
            )(x)
        
        x = ak.ImageBlock(
            block_type=AK_SERCH_MODEL, 
            normalize=False,
            augment=False
        )(x)
        output_node = ak.RegressionHead()(x)
        model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            directory=f"./auto_model/{task_name}_bs{args.batch_size}_seed{args.seed}_{split_run_name}",
            max_trials=args.max_trials,
            seed=args.seed,
        )
        ak_settings = model.tuner.hypermodel.get_config()
        ak_model_name = AK_SERCH_MODEL.name if AK_SERCH_MODEL is not None else None
        ak_model_version = (
            AK_SERCH_MODEL.version if AK_SERCH_MODEL is not None else None
        )

        batched_train_ds, batched_val_ds = (
            train_ds.batch(batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE
            ),
            val_ds.batch(batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE
            )
        )
        model.fit(
            batched_train_ds,
            epochs=args.epochs,
            callbacks=call_backs,
            validation_data=batched_val_ds,
        )
        best_model = model.export_model()
    else:
        if args.hp_tuning == "keras_tuner":
            batched_train_ds, batched_val_ds = (
                train_ds.batch(batch_size).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE
                ),
                val_ds.batch(batch_size).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE
                )
            )
            tuner = BayesianOptimization(
                        lambda hp: build_model_for_kerastuner(hp, args.execute_data_aug),
                        objective=Objective(
                            "val_mean_squared_error",
                            direction="min",
                        ),
                        max_trials=args.max_trials,
                        executions_per_trial=1,
                        directory=(
                            f"keras_tuner/{task_name}_seed{args.seed}_"
                            f"{split_run_name}{data_aug_suffix}"
                        ),
                        project_name=f"{args.model_name}_bs{batch_size}_bayesian",
                        overwrite=False,
                        seed=args.seed
                    )
            tuner.search(
                    batched_train_ds,
                    validation_data=batched_val_ds,
                    epochs=args.epochs,
                    callbacks=call_backs,
                )
            best_hp = tuner.get_best_hyperparameters(1)[0]
            best_model = build_model_for_kerastuner(best_hp, args.execute_data_aug)
    
    train_ds, val_ds, test_ds = (
        train_ds.batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        ),
        val_ds.batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        ),
        test_ds.batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        ),
    )
    
    time_training_start = time.perf_counter()
    history = best_model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=call_backs,
        validation_data=val_ds,
    )
    time_training_end = time.perf_counter()
    time_pred_start = time.perf_counter()
    y_pred = best_model.predict(test_ds)
    time_pred_end = time.perf_counter()

    y_test = np.concatenate([y for _, y in test_ds], axis=0)
    training_time = time_training_end - time_training_start
    pred_time = time_pred_end - time_pred_start

    result_csv_path = (
        f"../results/{task_name}_regression_{args.model_name}_{split_run_name}"
        f"{data_aug_suffix}_{timestamp}.csv"
    )
    create_result_csv(result_csv_path, y_pred.flatten(), y_test)

    save_model(best_model, save_model_path)

    mse = mean_squared_error(y_test, y_pred)

    record_exp_result(
        EXP_ID,
        # metrics
        {
            "rmse": np.sqrt(mse),
            "mse": mse,
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "training_time": training_time,
            "pred_time": pred_time,
        },
        # params
        {
            "epochs": args.epochs,
            "max_trials": args.max_trials,
            "batch_size": batch_size,
            "callbacks": str(
                [
                    f"{callback.__class__.__name__}[monitor={callback.monitor}]"
                    for callback in call_backs
                ]
            )
            if call_backs is not None
            else None,
            "optimizer": None
            if args.model_name == "autokeras"
            else best_model.optimizer.get_config()["name"],
            "loss_func": LOSS,
            "learning_rate": None
            if args.model_name == "autokeras"
            else best_hp.values.get("lr", None) if best_hp 
            else lr,
            "seed": args.seed,
        },
        # tags
        {
            "target": task_name,
            "gpu_names": gpu_names,
            "csv_path": data_path,
            "data_split": split_method,
            "split_seed": args.split_seed
            if args.split_type == BALANCED_SCAFFOLD_SPLIT_TYPE
            else None,
            "ak_settings": ak_settings if args.model_name == "autokeras" else None,
            "model_name": args.model_name,
            "serch_ak_model": f"{ak_model_name}"
            if args.model_name == "autokeras"
            else None,
            "serch_ak_model_version": f"{ak_model_version}"
            if args.model_name == "autokeras"
            else None,
            "model_path": save_model_path,
            "explanatory_val": "img",
            "result_csv_path": result_csv_path,
            "img_size": str(IMG_SIZE),
            "execute_data_aug": args.execute_data_aug,
            "hp_tuning": args.hp_tuning,
            "history": history.history if not history is None else None,
        },
    )
