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
from typing import List

import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# --- Configuration ---
MIN_SIZE    = 112                    # smallest face side we'll accept
THRESHOLD   = 0.4                    # max cosine distance for a “match”
MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"  # path to your saved Keras model

# Load your embedding model once
embedder = load_model(MODEL_PATH)

# Build the face detector once
detector = MTCNN()


def extract_faces(group_path: Path,
                  out_dir:    Path,
                  min_size:   int = MIN_SIZE) -> list[Path]:
    """
    - Runs MTCNN on the group image
    - Saves each crop to out_dir/face_{i}.jpg
    - Upscales any face below min_size
    - Returns a list of the saved face file Paths
    """
    out_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(str(group_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(img_rgb)
    saved = []

    for i, det in enumerate(detections):
        x, y, w, h = det["box"]
        x, y = max(0, x), max(0, y)
        face = img_rgb[y : y + h, x : x + w]

        # Upscale if too small
        if face.shape[0] < min_size or face.shape[1] < min_size:
            face = cv2.resize(face, (min_size, min_size),
                              interpolation=cv2.INTER_CUBIC)

        out_path = out_dir / f"face_{i}.jpg"
        # save as BGR for cv2.imwrite
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def get_embedding(img_path: Path) -> np.ndarray:
    """
    - Loads an image, resizes to the model's input size,
      normalizes pixels, and returns the embedding vector.
    """

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # model expects square inputs
    _, H, W, _ = embedder.input_shape
    face = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    emb  = embedder.predict(face)
    return emb[0]  # return 1D vector


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 1 - cosine_similarity(a, b)."""
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD) -> tuple[int, int]:
    """
    For each extracted face, compute its embedding vs. the person's embedding.
    Returns (match_count, non_match_count).
    """
    person_emb = get_embedding(person_path)
    matches = 0
    misses  = 0

    for p in face_paths:
        start     = time.time()
        face_emb  = get_embedding(p)
        dist      = cosine_distance(face_emb, person_emb)
        is_match  = (dist <= threshold)
        elapsed   = time.time() - start

        print(f"  {p.name}: dist={dist:.3f}, match={is_match} ({elapsed:.2f}s)")

        if is_match:
            matches += 1
        else:
            misses  += 1

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
    - Uses a fixed extract_dir ("extracted_faces" folder in script directory).
    - Keeps min_size and threshold constants.
    - Returns a JSON string.
    """
    # Use a constant directory for extracted faces
    extract_dir = Path(__file__).resolve().parent / "extracted_faces"

    # --- Sanity checks ---
    if not group_img.is_file():
        return json.dumps({
            "status": "error",
            "message": f"Missing group image: {group_img}"
        })
    if not person_directory.is_dir():
        return json.dumps({
            "status": "error",
            "message": f"Missing person directory: {person_directory}"
        })

    # --- Clear extract_dir ---
    if extract_dir.exists():
        for f in extract_dir.iterdir():
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

        name_out = person_path.stem

        if verbose:
            print(f"\n-- {person_file} --")
        t, f = verify_faces(faces, person_path, threshold=THRESHOLD)

        if t > 0:
            if verbose:
                print(f"=> **{person_file} FOUND** ({t} of {len(faces)} faces matched)")
            found_people.append(name_out)
        else:
            if verbose:
                print(f"=> {person_file} NOT found")

    # Build JSON response
    response = {
        "status": True,
        "found": found_people,
        "total_faces": len(faces)
    }
    return json.dumps(response, indent=2)


def first_image_in_dir(dir_path: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> Path | None:
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




