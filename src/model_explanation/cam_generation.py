"""CAM model adapters and heatmap generation primitives.

The public functions in this module are intentionally independent from the
command-line entry points.  This keeps model/heatmap behaviour testable and
ensures faithfulness evaluation uses exactly the same target score as CAM
generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf


CamMethod = Literal["gradcam", "scorecam", "layercam"]
TargetClass = Literal["positive", "negative", "predicted", "ground_truth"]


@dataclass(frozen=True)
class CamModels:
    """Models needed by the three supported CAM implementations."""

    prediction_model: tf.keras.Model
    cam_model: tf.keras.Model
    backbone: tf.keras.Model | None
    layercam_model: tf.keras.Model


def load_image_model(model_path: str | Path, model_family: str) -> tf.keras.Model:
    """Load an AutoKeras or ordinary Keras image model without compiling it."""
    if model_family == "autokeras":
        import autokeras as ak

        return tf.keras.models.load_model(
            str(model_path), custom_objects=ak.CUSTOM_OBJECTS, compile=False
        )
    if model_family == "keras":
        return tf.keras.models.load_model(str(model_path), compile=False)
    raise ValueError(f"unsupported model_family: {model_family!r}")


def build_cam_model_autokeras(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int] = (256, 256, 3),
) -> tuple[tf.keras.Model, tf.keras.Model | None, tf.keras.Model]:
    """Rebuild an exported AutoKeras model around its nested CNN backbone.

    AutoKeras exports preprocessing and a nested Functional backbone.  Feeding
    the outer graph directly to tf-keras-vis can make its convolution-layer
    search select a disconnected layer.  Rebuilding from a cloned backbone
    gives CAM a flat, fixed-size input while preserving learned weights.
    """
    backbone = None
    layers_after_backbone: list[str] = []
    found_backbone = False
    for layer in model.layers:
        nested_layers = getattr(layer, "layers", ())
        if not found_backbone and len(nested_layers) > 5:
            backbone = layer
            found_backbone = True
        elif found_backbone:
            layers_after_backbone.append(layer.name)

    if backbone is None:
        return model, None, model

    inputs = tf.keras.Input(shape=input_shape)
    cloned_backbone = tf.keras.models.clone_model(backbone, input_tensors=inputs)
    cloned_backbone.set_weights(backbone.get_weights())
    output = cloned_backbone.output
    for layer_name in layers_after_backbone:
        output = model.get_layer(layer_name)(output)
    cam_model = tf.keras.Model(inputs=inputs, outputs=output)
    return cam_model, cloned_backbone, cam_model


def prepare_cam_models(
    model: tf.keras.Model,
    model_family: str,
    input_shape: tuple[int, int, int] = (256, 256, 3),
) -> CamModels:
    """Return prediction and CAM views for a loaded model."""
    if model_family == "autokeras":
        cam_model, backbone, layercam_model = build_cam_model_autokeras(
            model, input_shape=input_shape
        )
        prediction_model = cam_model
    elif model_family == "keras":
        cam_model, layercam_model = model, model
        backbone = _build_activation_model(model)
        prediction_model = model
    else:
        raise ValueError(f"unsupported model_family: {model_family!r}")
    return CamModels(prediction_model, cam_model, backbone, layercam_model)


def _build_activation_model(model: tf.keras.Model) -> tf.keras.Model | None:
    """Return a model ending at the deepest spatial feature layer."""
    for layer in reversed(model.layers):
        try:
            output_shape = tuple(layer.output_shape)
        except (AttributeError, TypeError):
            continue
        if len(output_shape) == 4:
            try:
                return tf.keras.Model(model.inputs, layer.output)
            except ValueError:
                continue
    return None


def binary_target_score(
    predictions: np.ndarray | tf.Tensor, target_class: int
) -> float:
    """Return P(y=target_class) from a one-output sigmoid model."""
    value = float(np.asarray(predictions).reshape(-1)[0])
    if target_class == 1:
        return value
    if target_class == 0:
        return 1.0 - value
    raise ValueError(f"target_class must be 0 or 1, got {target_class}")


class BinaryTargetScore:
    """tf-keras-vis score callable for a binary sigmoid output."""

    def __init__(self, target_class: int):
        if target_class not in (0, 1):
            raise ValueError(f"target_class must be 0 or 1, got {target_class}")
        self.target_class = target_class

    def __call__(self, output: tf.Tensor) -> tf.Tensor:
        positive = output[:, 0]
        return positive if self.target_class == 1 else 1.0 - positive


class RegressionTargetScore:
    """tf-keras-vis score callable for a scalar regression output."""

    def __call__(self, output: tf.Tensor) -> tf.Tensor:
        return output[:, 0]


def resolve_target_class(
    target_policy: TargetClass,
    positive_probability: float,
    ground_truth: object | None = None,
) -> int:
    """Resolve a binary target class without silently using labels."""
    if target_policy == "positive":
        return 1
    if target_policy == "negative":
        return 0
    if target_policy == "predicted":
        return int(positive_probability >= 0.5)
    if target_policy == "ground_truth":
        if ground_truth is None or bool(np.isnan(ground_truth)):
            raise ValueError("ground_truth target requires a non-missing label")
        target = int(ground_truth)
        if target not in (0, 1):
            raise ValueError(f"ground-truth label must be 0 or 1, got {ground_truth!r}")
        return target
    raise ValueError(f"unsupported target_policy: {target_policy!r}")


def validate_heatmap(
    heatmap: np.ndarray,
    expected_shape: tuple[int, int] | None = None,
    *,
    reject_constant: bool = True,
) -> np.ndarray:
    """Validate and return a float32 CAM heatmap.

    Constant maps are rejected by default because they carry no ranking signal
    for faithfulness, IoU, or per-atom aggregation and commonly indicate a CAM
    generation failure.
    """
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"heatmap must be 2-D, got shape {array.shape}")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"heatmap shape {array.shape} does not match expected {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("heatmap contains NaN or infinity")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"heatmap values must be in [0, 1], got min={minimum}, max={maximum}"
        )
    if reject_constant and maximum - minimum <= 1e-8:
        raise ValueError("heatmap is constant and contains no attribution signal")
    return array


def scorecam_white_baseline(
    cam_model: tf.keras.Model,
    backbone: tf.keras.Model,
    inputs: np.ndarray,
    *,
    target_class: int | None,
    task_type: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute Score-CAM using the white molecular-image background."""
    if backbone is None:
        raise ValueError(
            "ScoreCAM requires an explicit convolutional backbone; "
            "the selected model adapter did not expose one"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    height, width = inputs.shape[1:3]
    image = inputs[0]
    activations = np.asarray(backbone(inputs, training=False))[0]
    if activations.ndim != 3:
        raise ValueError(
            f"backbone must return (H, W, C) activations, got {activations.shape}"
        )

    def scores(batch: np.ndarray) -> np.ndarray:
        predictions = np.asarray(cam_model(batch, training=False)).reshape(-1)
        if task_type == "regression":
            return predictions.astype(np.float32)
        if target_class == 1:
            return predictions.astype(np.float32)
        if target_class == 0:
            return (1.0 - predictions).astype(np.float32)
        raise ValueError("classification ScoreCAM requires target_class 0 or 1")

    white = np.ones((1, height, width, 3), dtype=np.float32)
    baseline_score = float(scores(white)[0])
    cam = np.zeros((height, width), dtype=np.float32)

    channel_count = activations.shape[-1]
    for start in range(0, channel_count, batch_size):
        end = min(start + batch_size, channel_count)
        masked = np.empty((end - start, height, width, 3), dtype=np.float32)
        resized_maps = np.zeros((end - start, height, width), dtype=np.float32)
        for offset, channel in enumerate(range(start, end)):
            activation = activations[:, :, channel]
            low = float(activation.min())
            high = float(activation.max())
            if high - low <= 1e-8:
                masked[offset] = white[0]
                continue
            normalized = (activation - low) / (high - low)
            resized = tf.image.resize(
                normalized[None, :, :, None],
                [height, width],
                method="bilinear",
            ).numpy()[0, :, :, 0]
            resized_maps[offset] = resized
            mask = resized[:, :, None]
            masked[offset] = mask * image + (1.0 - mask)

        weights = np.maximum(scores(masked) - baseline_score, 0.0)
        cam += np.einsum("k,khw->hw", weights, resized_maps)

    cam = np.maximum(cam, 0.0)
    maximum = float(cam.max())
    if maximum <= 1e-8:
        raise ValueError("ScoreCAM produced a constant-zero heatmap")
    cam = (cam - float(cam.min())) / (maximum - float(cam.min()) + 1e-8)
    return validate_heatmap(cam, (height, width))


def generate_cam_heatmap(
    models: CamModels,
    inputs: np.ndarray | tf.Tensor,
    *,
    method: CamMethod,
    task_type: str,
    target_class: int | None = None,
    scorecam_batch_size: int = 32,
) -> np.ndarray:
    """Generate one normalized CAM heatmap."""
    batch = tf.convert_to_tensor(inputs, dtype=tf.float32)
    if tuple(batch.shape[:1]) != (1,):
        raise ValueError(f"CAM generation expects a single image, got {batch.shape}")
    if task_type == "classification":
        if target_class not in (0, 1):
            raise ValueError("classification CAM requires target_class 0 or 1")
        score = BinaryTargetScore(int(target_class))
    elif task_type == "regression":
        score = RegressionTargetScore()
    else:
        raise ValueError(f"unsupported task_type: {task_type!r}")

    if method == "scorecam":
        return scorecam_white_baseline(
            models.cam_model,
            models.backbone,
            batch.numpy(),
            target_class=target_class,
            task_type=task_type,
            batch_size=scorecam_batch_size,
        )

    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.layercam import Layercam
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    if method == "gradcam":
        cam = Gradcam(models.cam_model, model_modifier=ReplaceToLinear(), clone=True)
    elif method == "layercam":
        cam = Layercam(
            models.layercam_model, model_modifier=ReplaceToLinear(), clone=True
        )
    else:
        raise ValueError(f"unsupported CAM method: {method!r}")
    result = cam(
        score,
        batch,
        penultimate_layer=-1,
        seek_penultimate_conv_layer=True,
        training=False,
        expand_cam=True,
        normalize_cam=True,
    )
    return validate_heatmap(
        np.asarray(result[0], dtype=np.float32), tuple(batch.shape[1:3])
    )
