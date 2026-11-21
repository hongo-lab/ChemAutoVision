import numpy as np
import pytest
import tensorflow as tf

from model_explanation.cam_generation import (
    binary_target_score,
    resolve_target_class,
    validate_heatmap,
    generate_cam_heatmap,
    prepare_cam_models,
    scorecam_white_baseline,
)


def test_binary_target_score_supports_both_classes():
    prediction = np.array([[0.2]], dtype=np.float32)
    assert binary_target_score(prediction, 1) == pytest.approx(0.2)
    assert binary_target_score(prediction, 0) == pytest.approx(0.8)


def test_resolve_target_class_does_not_use_label_for_fixed_target():
    assert resolve_target_class("positive", 0.1, 0) == 1
    assert resolve_target_class("negative", 0.9, 1) == 0
    assert resolve_target_class("predicted", 0.49, 1) == 0
    assert resolve_target_class("ground_truth", 0.9, 0) == 0


def test_validate_heatmap_rejects_constant_nan_and_wrong_shape():
    with pytest.raises(ValueError, match="constant"):
        validate_heatmap(np.zeros((4, 4), dtype=np.float32))
    invalid = np.arange(16, dtype=np.float32).reshape(4, 4)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_heatmap(invalid)
    with pytest.raises(ValueError, match="does not match"):
        validate_heatmap(
            np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0,
            expected_shape=(8, 8),
        )


@pytest.mark.parametrize("method", ["gradcam", "layercam"])
def test_gradient_cam_methods_run_for_an_ordinary_keras_cnn(method):
    inputs = tf.keras.Input(shape=(8, 8, 3))
    features = tf.keras.layers.Conv2D(
        2,
        3,
        activation="relu",
        use_bias=False,
        kernel_initializer="ones",
    )(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", use_bias=False, kernel_initializer="ones"
    )(pooled)
    model = tf.keras.Model(inputs, outputs)
    models = prepare_cam_models(model, "keras", input_shape=(8, 8, 3))
    assert models.backbone is not None
    image = np.random.default_rng(1).random((1, 8, 8, 3), dtype=np.float32)
    heatmap = generate_cam_heatmap(
        models,
        image,
        method=method,
        task_type="classification",
        target_class=1,
    )
    assert heatmap.shape == (8, 8)
    assert 0.0 <= heatmap.min() < heatmap.max() <= 1.0


def test_scorecam_white_baseline_returns_nonconstant_map():
    class DarknessModel:
        def __call__(self, inputs, training=False):
            values = 1.0 - np.asarray(inputs).mean(axis=(1, 2, 3))
            return values[:, None].astype(np.float32)

    class FixedBackbone:
        def __call__(self, inputs, training=False):
            activation = np.array([[[1.0], [0.0]], [[1.0], [0.0]]], dtype=np.float32)
            return np.repeat(activation[None], len(inputs), axis=0)

    image = np.ones((1, 4, 4, 3), dtype=np.float32)
    image[:, :, :2, :] = 0.0
    heatmap = scorecam_white_baseline(
        DarknessModel(),
        FixedBackbone(),
        image,
        target_class=1,
        task_type="classification",
        batch_size=1,
    )
    assert heatmap.shape == (4, 4)
    assert heatmap[:, :2].mean() > heatmap[:, 2:].mean()
