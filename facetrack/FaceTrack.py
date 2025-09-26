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
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from mtcnn import MTCNN
import keras

# --- Configuration ---
MIN_SIZE   = 112      # smallest face side we'll accept
THRESHOLD  = 0.6      # max cosine distance for a “match”
MARGIN_GAP = 0.05     # optional runner-up margin; set to 0 to disable
MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"  # your saved Keras 3 model

# ---------- Model loading (Keras 3) ----------
def _merge_two_feature_maps_if_needed(base: keras.Model) -> keras.Model:
    """If the loaded model outputs two 4D tensors (N,H,W,C),
    merge them -> GAP -> Dense head."""
    outs = base.outputs if isinstance(base.outputs, (list, tuple)) else [base.outputs]
    if len(outs) == 2 and getattr(outs[0].shape, "rank", None) == 4:
        merged = keras.layers.Average(name="merge_avg")(outs)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(merged)
        # Simple head (example only)
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
        H = W = 256
    return int(H), int(W)


def get_embedding(img_path: Path) -> np.ndarray:
    """
    Load image, resize to model's input size, return embedding vector (1D).
    Keep values as float32 in 0..255 (matches augmentation value_range).
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


# ---------- Single-person verifier (kept for your one-person checks) ----------
def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD) -> Tuple[int, int, float]:
    """
    For each extracted face, compare embedding to the person's embedding.
    Returns (matches, misses, min_dist).
    """
    person_emb = get_embedding(person_path)
    matches = 0
    misses  = 0
    min_dist = float("inf")

    for p in face_paths:
        start = time.time()
        face_emb = get_embedding(p)
        dist     = cosine_distance(face_emb, person_emb)
        ok       = (dist <= threshold)
        elapsed  = time.time() - start
        print(f"  {p.name}: dist={dist:.3f}, match={ok} ({elapsed:.2f}s)")
        matches += int(ok)
        misses  += int(not ok)
        if dist < min_dist:
            min_dist = dist

    return matches, misses, min_dist


# ---------- New: group classification (multi-person) ----------
def load_person_embeddings(person_directory: Path) -> List[Tuple[str, np.ndarray]]:
    """
    Preload embeddings for all person images in the directory.
    Returns list of (name_stem, embedding).
    """
    people: List[Tuple[str, np.ndarray]] = []
    for person_file in sorted(os.listdir(person_directory)):
        person_path = person_directory / person_file
        if not person_path.is_file():
            continue
        name = person_path.stem
        emb = get_embedding(person_path)
        people.append((name, emb))
    return people


def classify_faces_against_people(face_paths: list[Path],
                                  people_embs: List[Tuple[str, np.ndarray]],
                                  threshold: float = THRESHOLD,
                                  margin_gap: float = MARGIN_GAP) -> Dict[str, int]:
    """
    For each face, pick the best-matching person (lowest distance).
    Assign that face to the person if: best_dist <= threshold and
    (optional) the runner-up is at least margin_gap worse.
    Returns a dict of counts {person_name: votes}.
    """
    votes: Dict[str, int] = {}

    # Cache face embeddings so we don't re-run the model N_people times per face
    face_emb_cache: Dict[Path, np.ndarray] = {}
    for fp in face_paths:
        face_emb_cache[fp] = get_embedding(fp)

    for fp, femb in face_emb_cache.items():
        # Compute distances to all known people
        dists: List[Tuple[str, float]] = []
        for name, pemb in people_embs:
            d = cosine_distance(femb, pemb)
            dists.append((name, d))

        if not dists:
            continue

        dists.sort(key=lambda x: x[1])  # ascending distance: best first
        best_name, best_dist = dists[0]

        # Runner-up margin (optional)
        margin_ok = True
        if margin_gap > 0 and len(dists) > 1:
            second_best_dist = dists[1][1]
            margin_ok = (second_best_dist - best_dist) >= margin_gap

        print(f"Face {fp.name}: best={best_name} (dist={best_dist:.3f})"
              + (f", second={dists[1][0]} (Δ={second_best_dist - best_dist:.3f})" if len(dists) > 1 else ""))

        if best_dist <= threshold and margin_ok:
            votes[best_name] = votes.get(best_name, 0) + 1
        else:
            print(f"  -> Face {fp.name} unassigned (dist too high or ambiguous).")

    return votes


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
    - Returns a JSON string with potentially multiple people.
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

    # 2) Load all person embeddings once
    people_embs = load_person_embeddings(person_directory)

    # 3) Assign each face to at most one person (top-1 with threshold + optional margin)
    votes = classify_faces_against_people(faces, people_embs, threshold=THRESHOLD, margin_gap=MARGIN_GAP)

    # 4) Collate found names (anyone with ≥1 assigned face)
    found_people = sorted([name for name, count in votes.items() if count >= 1])

    if verbose:
        if found_people:
            print("\n=> Group result:")
            for name in found_people:
                print(f"  - {name}: {votes[name]} face(s)")
        else:
            print("\n=> No confident matches in group photo.")

    return json.dumps({
        "status": True,
        "found": found_people,
        "counts": votes,
        "total_faces": len(faces)
    }, indent=2)


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
        raise SystemExit(1)#!/usr/bin/env python3
"""
FaceTrack.py

1) Extract faces from a group image using MTCNN.
2) Upscale any tiny faces to 112×112.
3) Classify each extracted face against gallery people using your embedder.
"""
import json
import time
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from mtcnn import MTCNN
import keras

# --- Configuration ---
MIN_SIZE   = 112      # smallest face side we'll accept
THRESHOLD  = 0.30     # stricter match threshold (cosine distance)
MARGIN_GAP = 0.12     # require this gap to runner-up to accept a match
MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"  # your saved Keras 3 model

# Input preprocessing expected by the embedder:
#   "raw255"   -> keep 0..255
#   "scale01"  -> scale to 0..1
#   "inception"-> scale to [-1,1] (typical for Inception-family models)
PREPROC    = "inception"

# ---------- Model loading (Keras 3) ----------
def _merge_two_feature_maps_if_needed(base: keras.Model) -> keras.Model:
    """If the loaded model outputs two 4D tensors (N,H,W,C),
    merge them -> GAP -> Dense head. (Kept for compatibility)"""
    outs = base.outputs if isinstance(base.outputs, (list, tuple)) else [base.outputs]
    if len(outs) == 2 and getattr(outs[0].shape, "rank", None) == 4:
        merged = keras.layers.Average(name="merge_avg")(outs)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(merged)
        # Example head (won't be used by the embedder, but we keep this for parity)
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

# Load your embedding model once (must output an embedding vector, not softmax!)
from .model_loader import load_embedder
embedder: keras.Model = load_embedder()

# Build the face detector once
detector = MTCNN()

# ---------- Preprocess helper ----------
def _preprocess_for_embedder(x: np.ndarray) -> np.ndarray:
    if PREPROC == "raw255":
        return x.astype("float32")
    if PREPROC == "scale01":
        return (x.astype("float32")) / 255.0
    # default: Inception-style [-1, 1]
    return (x.astype("float32") / 127.5) - 1.0


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
        H = W = 256
    return int(H), int(W)


def get_embedding(img_path: Path) -> np.ndarray:
    """
    Load image, resize to model's input size, return L2-normalized embedding vector (1D).
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to load image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = _model_input_hw(embedder)
    face = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    arr = _preprocess_for_embedder(face)
    arr = np.expand_dims(arr, axis=0)
    emb = embedder.predict(arr, verbose=0)

    v = np.ravel(emb[0]).astype("float32")
    # L2-normalize (important for cosine distance)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 1 - cosine_similarity(a, b) assuming L2-normalized vectors."""
    # After L2-normalization, cosine_similarity = dot(a,b)
    return 1.0 - float(np.dot(a, b))


# ---------- Single-person verifier (useful for one-person checks) ----------
def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD) -> Tuple[int, int, float]:
    """
    For each extracted face, compare embedding to the person's embedding.
    Returns (matches, misses, min_dist).
    """
    person_emb = get_embedding(person_path)
    matches = 0
    misses  = 0
    min_dist = float("inf")

    for p in face_paths:
        start = time.time()
        face_emb = get_embedding(p)
        dist     = cosine_distance(face_emb, person_emb)
        ok       = (dist <= threshold)
        elapsed  = time.time() - start
        print(f"  {p.name}: dist={dist:.3f}, match={ok} ({elapsed:.2f}s)")
        matches += int(ok)
        misses  += int(not ok)
        if dist < min_dist:
            min_dist = dist

    return matches, misses, min_dist


# ---------- Group classification (multi-person) ----------
def load_person_embeddings(person_directory: Path) -> List[Tuple[str, np.ndarray]]:
    """
    Preload embeddings for all person images in the directory.
    Returns list of (name_stem, embedding).
    """
    people: List[Tuple[str, np.ndarray]] = []
    for person_file in sorted(os.listdir(person_directory)):
        person_path = person_directory / person_file
        if not person_path.is_file():
            continue
        name = person_path.stem
        emb = get_embedding(person_path)
        people.append((name, emb))
    return people


def classify_faces_against_people(face_paths: list[Path],
                                  people_embs: List[Tuple[str, np.ndarray]],
                                  threshold: float = THRESHOLD,
                                  margin_gap: float = MARGIN_GAP) -> Dict[str, int]:
    """
    For each face, pick the best-matching person (lowest distance).
    Assign that face to the person if: best_dist <= threshold and
    (optional) the runner-up is at least margin_gap worse.
    Returns a dict of counts {person_name: votes}.
    """
    votes: Dict[str, int] = {}
    if not people_embs or not face_paths:
        return votes

    # Cache face embeddings (avoid N_people * N_faces predictions)
    face_emb_cache: Dict[Path, np.ndarray] = {}
    for fp in face_paths:
        face_emb_cache[fp] = get_embedding(fp)

    for fp, femb in face_emb_cache.items():
        dists: List[Tuple[str, float]] = []
        for name, pemb in people_embs:
            d = cosine_distance(femb, pemb)
            dists.append((name, d))

        dists.sort(key=lambda x: x[1])  # ascending distance: best first
        best_name, best_dist = dists[0]

        # Runner-up margin (optional)
        margin_ok = True
        if margin_gap > 0 and len(dists) > 1:
            second_best_dist = dists[1][1]
            margin_ok = (second_best_dist - best_dist) >= margin_gap

        # DEBUG: show top-3 per face
        top_show = min(3, len(dists))
        print(f"[DIST] {fp.name}: " + ", ".join([f"{nm}:{dist:.3f}" for nm, dist in dists[:top_show]]))

        if best_dist <= threshold and margin_ok:
            votes[best_name] = votes.get(best_name, 0) + 1
        else:
            print(f"  -> Face {fp.name} unassigned (dist={best_dist:.3f}, margin_ok={margin_ok}).")

    return votes


def find_people_in_group_simple(
    group_img: Path,
    person_directory: Path,
    *,
    verbose: bool = True,
) -> str:
    """
    Simplified version:
    - Only needs the group image path and person directory path.
    - Uses a fixed extract_dir ("extracted_faces" folder near this script).
    - Returns a JSON string with potentially multiple people.
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

    # 2) Load all person embeddings once
    people_embs = load_person_embeddings(person_directory)

    # 3) Assign each face to at most one person (top-1 with threshold + optional margin)
    votes = classify_faces_against_people(faces, people_embs, threshold=THRESHOLD, margin_gap=MARGIN_GAP)

    # 4) Collate found names (anyone with ≥1 assigned face)
    found_people = sorted([name for name, count in votes.items() if count >= 1])

    if verbose:
        if found_people:
            print("\n=> Group result:")
            for name in found_people:
                print(f"  - {name}: {votes[name]} face(s)")
        else:
            print("\n=> No confident matches in group photo.")

    return json.dumps({
        "status": True,
        "found": found_people,
        "counts": votes,
        "total_faces": len(faces)
    }, indent=2)


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
            print(f"- {name} (faces: {result['counts'][name]})")
    else:
        print("No known persons detected.")


    print(f"Using group image: {group_img}")
    result_json = find_people_in_group_simple(group_img, person_dir)
    result = json.loads(result_json)

    print("\n=== Summary: People detected in group photo ===")
    if result.get("status") and result.get("found"):
        for name in result["found"]:
            print(f"- {name} (faces: {result['counts'][name]})")
    else:
        print("No known persons detected.")
