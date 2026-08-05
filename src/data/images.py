import logging
import math
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
from rdkit import Chem
from rdkit.Chem import Draw
from tensorflow.keras.utils import img_to_array

from settings import IMG_SIZE, IS_DOWNSCALE_BEFORE_DAUG


def smi_to_img(smiles: str, img_size: tuple[int, int]) -> Image:
    try:
        img = Draw.MolToImage(
            Chem.MolFromSmiles(smiles), size=(img_size[0], img_size[1])
        )
        return img
    except Exception as e:
        logging.error((f"Error processing SMILES to img'{smiles}': {e}"))
        raise

def create_cls_image_data(
    data_path: str,
    task_name: str,
    test_size: int,
    img_size: tuple[int, int],
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    _df = pd.read_pickle(data_path)
    smiles_column = "smiles"

    _df = _df[_df[task_name] != -1]

    _df["image"] = _df["img_path"].apply(preprocess_image)

    df_train = _df[_df["group"] == "train"]
    df_val = _df[_df["group"] == "val"]
    df_test = _df[_df["group"] == "test"]

    df_train = df_train[["image", task_name, smiles_column]]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (np.stack(df_train["image"]), df_train[task_name])
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    val_ds = tf.data.Dataset.from_tensor_slices(
        (np.stack(df_val["image"].values), df_val[task_name])
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, int(y)))

    test_ds = tf.data.Dataset.from_tensor_slices(
        (np.stack(df_test["image"].values), df_test[task_name])
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, int(y)))
    return train_ds, val_ds, test_ds


def create_regression_img_data(
    data_path: str,
    task_name: str,
    test_size: int | None,
    img_size: tuple[int, int],
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    _df = pd.read_pickle(data_path)

    smiles_column = "smiles"
    ob_column = task_name

    _df["image"] = _df["img_path"].apply(preprocess_image)

    df_train = _df[_df["group"] == "train"]
    df_val = _df[_df["group"] == "val"]
    df_test = _df[_df["group"] == "test"]

    df_train = df_train[["image", ob_column, smiles_column]]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (np.stack(df_train["image"]), df_train[ob_column])
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    val_ds = tf.data.Dataset.from_tensor_slices(
        (np.stack(df_val["image"]), df_val[ob_column])
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    test_ds = tf.data.Dataset.from_tensor_slices(
        (np.stack(df_test["image"]), df_test[ob_column])
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    return train_ds, val_ds, test_ds


def padding_image(original_img: Image, img_size: tuple[int, int]) -> Image:
    img = Image.new("RGB", (img_size[0], img_size[1]), "white")
    x = (img.size[0] - original_img.size[0]) // 2
    y = (img.size[1] - original_img.size[1]) // 2
    img.paste(original_img, (x, y))
    return img


def preprocess_image(file_path: str) -> tf.Tensor:
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
    image = tf.cast(image, tf.float32)

    return image


