import logging
from pathlib import Path
import keras

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"

def _merge_two_feature_maps_if_needed(model: keras.Model) -> keras.Model:
    """If the loaded model has TWO 4D outputs feeding a Dense, merge them first."""
    outs = model.outputs if isinstance(model.outputs, (list, tuple)) else [model.outputs]
    if len(outs) == 2 and getattr(outs[0].shape, "rank", None) == 4:
        merged = keras.layers.Average(name="merge_avg")(outs)  # or Concatenate(axis=-1)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(merged)
        x = keras.layers.Dense(128, activation="relu", name="dense")(x)
        x = keras.layers.BatchNormalization(name="bn1")(x)
        x = keras.layers.Dense(64, activation="relu", name="dense_1")(x)
        x = keras.layers.BatchNormalization(name="bn2")(x)
        x = keras.layers.Dense(32, activation="relu", name="dense_2")(x)
        x = keras.layers.BatchNormalization(name="bn3")(x)
        x = keras.layers.Dropout(0.3, name="dropout")(x)
        out = keras.layers.Dense(5, activation="softmax", name="dense_3")(x)
        return keras.Model(inputs=model.inputs, outputs=out, name="patched_model")
    return model

def _build_fallback_model(input_shape=(256, 256, 3)) -> keras.Model:
    """
    Build a sane default: InceptionV3 (imagenet) + the exact head you expect.
    This bypasses any weird serialization bugs in the .keras file.
    """
    base = keras.applications.InceptionV3(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    x = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = keras.layers.Dense(128, activation="relu", name="dense")(x)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    out = keras.layers.Dense(5, activation="softmax", name="dense_3")(x)
    m = keras.Model(inputs=base.input, outputs=out, name="fallback_embedder")
    LOGGER.warning("Using FALLBACK embedder (built from code) — saved model failed to load")
    return m

def load_embedder() -> keras.Model:
    """
    Try to load the saved model; if it fails during deserialization
    (e.g., 'Dense got 2 tensors'), build a fallback model in code.
    """
    try:
        m = keras.saving.load_model(str(MODEL_PATH), compile=False, safe_mode=False)
        m = _merge_two_feature_maps_if_needed(m)
        LOGGER.warning("Embedder loaded from file: %s | outs=%s", MODEL_PATH.name, len(m.outputs))
        return m
    except Exception as e:
        LOGGER.error("Failed to load %s: %s", MODEL_PATH.name, e)
        return _build_fallback_model()
