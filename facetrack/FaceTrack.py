#!/usr/bin/env python3
"""
FaceTrack.py

1) Extract faces from a group image using MTCNN.
2) Upscale any tiny faces to 112×112.
3) For each person image in Person/, verify all extracted faces via your custom Inception model.
"""
import json
import time
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from mtcnn import MTCNN
import keras

# --- Configuration ---
MIN_SIZE  = 112      # smallest face side we'll accept
THRESHOLD = 0.4      # max cosine distance for a “match”
MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"  # your saved Keras 3 model

# ---------- Model loading (Keras 3) ----------
def _merge_two_feature_maps_if_needed(base: keras.Model) -> keras.Model:
    """If the loaded model outputs two 4D tensors (N,H,W,C),
    merge them -> GAP -> Dense head."""
    outs = base.outputs if isinstance(base.outputs, (list, tuple)) else [base.outputs]
    if len(outs) == 2 and getattr(outs[0].shape, "rank", None) == 4:
        # Merge choice: Average (blend) or Concatenate (stack channels)
        merged = keras.layers.Average(name="merge_avg")(outs)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(merged)
        # Recreate a simple head (matches your earlier config)
        x = keras.layers.Dense(128, activation="relu", name="dense")(x)
        x = keras.layers.BatchNormalization(name="bn1")(x)
        x = keras.layers.Dense(64, activation="relu", name="dense_1")(x)
        x = keras.layers.BatchNormalization(name="bn2")(x)
        x = keras.layers.Dense(32, activation="relu", name="dense_2")(x)
        x = keras.layers.BatchNormalization(name="bn3")(x)
        x = keras.layers.Dropout(0.3, name="dropout")(x)
        out = keras.layers.Dense(5, activation="softmax", name="dense_3")(x)
        return keras.Model(inputs=base.inputs, outputs=out, name="patched_model")
    return base

def load_face_model() -> keras.Model:
    base = keras.saving.load_model(str(MODEL_PATH), compile=False, safe_mode=False)
    return _merge_two_feature_maps_if_needed(base)

# Load your embedding model once
from .model_loader import load_embedder

# Load your embedding model once
embedder: keras.Model = load_embedder()


# Build the face detector once
detector = MTCNN()


def extract_faces(group_path: Path,
                  out_dir: Path,
                  min_size: int = MIN_SIZE) -> list[Path]:
    """
    - Runs MTCNN on the group image
    - Saves each crop to out_dir/face_{i}.jpg
    - Upscales any face below min_size
    - Returns a list of saved face Paths
    """
    out_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(str(group_path))
    if img_bgr is None:
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(img_rgb)
    saved: list[Path] = []

    for i, det in enumerate(detections):
        x, y, w, h = det.get("box", [0, 0, 0, 0])
        x, y = max(0, x), max(0, y)
        face = img_rgb[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # Upscale if too small
        if face.shape[0] < min_size or face.shape[1] < min_size:
            face = cv2.resize(face, (min_size, min_size), interpolation=cv2.INTER_CUBIC)

        out_path = out_dir / f"face_{i}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def _model_input_hw(model: keras.Model) -> tuple[int, int]:
    """Return (H, W) for the model input, supporting list/tuple input shapes."""
    ishape = model.input_shape
    if isinstance(ishape, (list, tuple)) and isinstance(ishape[0], (list, tuple)):
        # e.g. [(None, 256, 256, 3), ...]
        H, W = ishape[0][1], ishape[0][2]
    elif isinstance(ishape, (list, tuple)):
        # e.g. (None, 256, 256, 3)
        H, W = ishape[1], ishape[2]
    else:
        # Fallback
        H = W = 256
    return int(H), int(W)

def get_embedding(img_path: Path) -> np.ndarray:
    """
    - Loads image, resizes to model's input size,
    - Returns embedding vector (1D)
    NOTE: keep values as float32 in 0..255 (matches augmentation value_range).
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to load image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = _model_input_hw(embedder)
    face = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    arr = face.astype("float32")  # keep 0..255
    arr = np.expand_dims(arr, axis=0)
    emb = embedder.predict(arr, verbose=0)
    return np.ravel(emb[0])  # ensure 1D


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 1 - cosine_similarity(a, b)."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return 1.0 - float(np.dot(a, b) / denom)


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD) -> tuple[int, int]:
    """For each extracted face, compare embedding to the person's embedding."""
    person_emb = get_embedding(person_path)
    matches = 0
    misses  = 0

    for p in face_paths:
        start = time.time()
        face_emb = get_embedding(p)
        dist     = cosine_distance(face_emb, person_emb)
        ok       = (dist <= threshold)
        elapsed  = time.time() - start
        print(f"  {p.name}: dist={dist:.3f}, match={ok} ({elapsed:.2f}s)")
        matches += int(ok)
        misses  += int(not ok)

    return matches, misses


def find_people_in_group_simple(
    group_img: Path,
    person_directory: Path,
    *,
    verbose: bool = True,
) -> str:
    """
    Simplified version:
    - Requires only the group image path and person directory path.
    - Uses a fixed extract_dir ("extracted_faces" folder near this script).
    - Returns a JSON string.
    """
    extract_dir = Path(__file__).resolve().parent / "extracted_faces"

    # Sanity checks
    if not group_img.is_file():
        return json.dumps({"status": False, "message": f"Missing group image: {group_img}"})
    if not person_directory.is_dir():
        return json.dumps({"status": False, "message": f"Missing person directory: {person_directory}"})

    # Clear extracts
    if extract_dir.exists():
        for f in list(extract_dir.iterdir()):
            if f.is_file():
                f.unlink()
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract faces
    faces = extract_faces(group_img, extract_dir, MIN_SIZE)

    if verbose:
        print("\n=== Checking each person image ===")

    # 2) Verify each person
    found_people: List[str] = []
    for person_file in sorted(os.listdir(person_directory)):
        person_path = person_directory / person_file
        if not person_path.is_file():
            continue
        if verbose:
            print(f"\n-- {person_file} --")
        t, f = verify_faces(faces, person_path, threshold=THRESHOLD)
        if t > 0:
            if verbose:
                print(f"=> **{person_file} FOUND** ({t} of {len(faces)} faces matched)")
            found_people.append(person_path.stem)
        else:
            if verbose:
                print(f"=> {person_file} NOT found")

    return json.dumps({"status": True, "found": found_people, "total_faces": len(faces)}, indent=2)


def first_image_in_dir(dir_path: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> Optional[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    files = sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts)
    return files[0] if files else None


if __name__ == "__main__":
    base_dir   = Path(__file__).resolve().parent
    group_dir  = base_dir / "GroupPhotoFromFE"
    person_dir = base_dir / "PeopleFromDataBase"

    group_img = first_image_in_dir(group_dir)
    if group_img is None:
        print(f"[!] No group photo found in: {group_dir}")
        print("    Make sure you ran the endpoint that downloads the latest group photo.")
        raise SystemExit(1)

    print(f"Using group image: {group_img}")
    result_json = find_people_in_group_simple(group_img, person_dir)
    result = json.loads(result_json)

    print("\n=== Summary: People detected in group photo ===")
    if result.get("status") and result.get("found"):
        for name in result["found"]:
            print(f"- {name}")
    else:
        print("No known persons detected.")

