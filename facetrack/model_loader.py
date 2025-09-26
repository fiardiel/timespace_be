# facetrack/model_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import keras
from keras import Model, layers
import tensorflow as tf

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"

def _l2norm(x):
    return tf.math.l2_normalize(x, axis=-1, name="l2norm")

def _merge_two_feature_maps_if_needed(base: keras.Model) -> keras.Model:
    """
    If the loaded model outputs TWO 4D feature maps (N,H,W,C),
    merge them so downstream layers see a single tensor.
    """
    outs = base.outputs if isinstance(base.outputs, (list, tuple)) else [base.outputs]
    if len(outs) == 2 and getattr(outs[0].shape, "rank", None) == 4:
        merged = layers.Average(name="merge_avg")(outs)
        return Model(inputs=base.inputs, outputs=merged, name="merged_backbone")
    return base

def _to_embedding_model(backbone: keras.Model, emb_dim: Optional[int] = None) -> keras.Model:
    """
    Convert ANY backbone into an embedding model (no softmax):
      - If output is 4D -> GlobalAveragePooling2D
      - If last layer is softmax classifier -> tap penultimate tensor
      - Optionally project to emb_dim
      - Always L2-normalize at the end
    """
    out = backbone.outputs
    out = out if isinstance(out, (list, tuple)) else [out]
    y = out[0]

    # If still a feature map (4D), pool it.
    if getattr(y.shape, "rank", None) == 4:
        y = layers.GlobalAveragePooling2D(name="gap")(y)

    # If backbone ends with a Dense softmax classifier, back up one layer.
    last = backbone.layers[-1]
    if isinstance(last, layers.Dense):
        act = getattr(last.activation, "__name__", "")
        if act == "softmax":
            # Penultimate output tensor:
            penult = backbone.layers[-2].output
            y = penult
            if getattr(y.shape, "rank", None) == 4:
                y = layers.GlobalAveragePooling2D(name="gap_from_penult")(y)

    # Ensure 2D
    if getattr(y.shape, "rank", None) and y.shape.rank > 2:
        y = layers.GlobalAveragePooling1D(name="gap1d")(y)

    # Optional projection to a tidy embedding size
    if emb_dim is not None:
        y = layers.Dense(emb_dim, activation=None, name="emb_dense")(y)
        y = layers.BatchNormalization(name="emb_bn")(y)

    y = layers.Activation("linear", name="emb_id")(y)
    y = _l2norm(y)

    return Model(inputs=backbone.inputs, outputs=y, name="embedder")

def _build_fallback_backbone(input_shape=(256, 256, 3)) -> keras.Model:
    """
    Minimal fallback backbone (no internet/downloads). This is NOT a classifier.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    # Output is a feature map; we will convert to embedding with _to_embedding_model.
    return Model(inp, x, name="fallback_backbone")

def load_embedder(emb_dim: Optional[int] = None) -> keras.Model:
    """
    Always return an **embedding model** (L2-normalized vector), never a softmax.
    Handles:
      - Saved models with two 4D outputs (merges them first)
      - Classifier models (taps penultimate tensor)
      - Fallback backbone (no downloads)
    """
    try:
        base = keras.saving.load_model(str(MODEL_PATH), compile=False, safe_mode=False)
        base = _merge_two_feature_maps_if_needed(base)
        emb = _to_embedding_model(base, emb_dim=emb_dim)
        LOGGER.warning("Embedder loaded: %s -> output_shape=%s", MODEL_PATH.name, emb.output_shape)
        return emb
    except Exception as e:
        LOGGER.error("Failed to load %s: %s", MODEL_PATH.name, e)
        LOGGER.warning("Using FALLBACK backbone (no softmax head).")
        back = _build_fallback_backbone()
        emb = _to_embedding_model(back, emb_dim=emb_dim or 256)
        LOGGER.warning("Fallback embedder output_shape=%s", emb.output_shape)
        return emb
