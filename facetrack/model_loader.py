# facetrack/model_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import keras
from keras import Model, layers

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"

def _merge_two_feature_maps_if_needed(base: keras.Model) -> keras.Model:
    outs = base.outputs if isinstance(base.outputs, (list, tuple)) else [base.outputs]
    if len(outs) == 2 and getattr(outs[0].shape, "rank", None) == 4:
        merged = layers.Average(name="merge_avg")(outs)
        return Model(inputs=base.inputs, outputs=merged, name="merged_backbone")
    return base

def _ensure_vector_embedding_tensor(t) -> layers.Layer:
    """
    Convert any tensor to a 2D vector, then L2-normalize.
    This is intentionally defensive: we check shape length as well as rank.
    """
    # Pool 4D feature maps -> 2D
    if hasattr(t, "shape"):
        # KerasTensor uses .shape with length; also supports .shape.rank in K3
        rank = getattr(t.shape, "rank", None)
        if rank is None:
            # fallback: use len(t.shape) if available
            try:
                rank = len(t.shape)
            except Exception:
                rank = None
        if rank == 4 or (hasattr(t, "shape") and len(t.shape) == 4):
            t = layers.GlobalAveragePooling2D(name="gap")(t)
        elif rank and rank > 2:
            t = layers.GlobalAveragePooling1D(name="gap1d")(t)

    # Make sure it's 2D flat
    t = layers.Flatten(name="flatten_if_needed")(t)

    # L2 normalize via Keras layer (no raw tf ops on KerasTensor)
    t = layers.UnitNormalization(axis=-1, name="l2norm")(t)
    return t

def _to_embedding_model(backbone: keras.Model, emb_dim: Optional[int] = None) -> keras.Model:
    out = backbone.outputs
    out = out if isinstance(out, (list, tuple)) else [out]
    y = out[0]

    # If backbone ends with a Dense softmax classifier, tap penultimate tensor
    last = backbone.layers[-1]
    if isinstance(last, layers.Dense):
        act = getattr(last.activation, "__name__", "")
        if act == "softmax":
            y = backbone.layers[-2].output

    # Optional projection to tidy size (after pooling/flattening)
    y = _ensure_vector_embedding_tensor(y)
    if emb_dim is not None:
        y = layers.Dense(emb_dim, activation=None, name="emb_dense")(y)
        y = layers.BatchNormalization(name="emb_bn")(y)
        y = layers.UnitNormalization(axis=-1, name="l2norm_proj")(y)

    emb = Model(inputs=backbone.inputs, outputs=y, name="embedder")
    return emb

def _build_fallback_backbone(input_shape=(256, 256, 3)) -> keras.Model:
    # Simple conv backbone (no classifier head), output is 4D feature map
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    return Model(inp, x, name="fallback_backbone")

def load_embedder(emb_dim: Optional[int] = 256) -> keras.Model:
    """
    Always return a 2D, L2-normalized embedding vector (never 4D, never softmax).
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
        emb = _to_embedding_model(back, emb_dim=emb_dim)
        LOGGER.warning("Fallback embedder output_shape=%s", emb.output_shape)
        return emb
