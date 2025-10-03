#!/usr/bin/env python3
"""
FaceTrack.py - Face Detection and Recognition Module

This module provides functionality for detecting and recognizing faces in group photos
by comparing them against known individuals. It uses MTCNN for face detection and
deep learning models for face recognition/verification.

Main Features:
1) Extract faces from group images using MTCNN face detector
2) Automatically upscale small faces to ensure minimum resolution (112×112)
3) Generate face embeddings using pre-trained deep learning models
4) Verify extracted faces against known individuals using cosine similarity
5) Return JSON results indicating which people were found in the group photo

Typical Usage:
    from pathlib import Path
    import FaceTrack

    group_image = Path("group_photo.jpg")
    person_directory = Path("known_people/")

    result = FaceTrack.find_people_in_group_simple(group_image, person_directory)
    print(result)  # JSON string with identification results

Author: TIME-Space Machine Learning Team
Dependencies: opencv-python, numpy, tensorflow, mtcnn
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

# --- Configuration Constants ---
MIN_SIZE = 112  # Minimum face size in pixels (faces smaller than this are upscaled)
THRESHOLD = 0.4  # Cosine distance threshold for face matching (lower = stricter)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "inception_model.keras"

# Global model cache to avoid reloading models multiple times
_model_cache = {}

# Initialize face detector (MTCNN) once at module level for efficiency
detector = MTCNN()


def get_model(model_name: str = "Inception"):
    """
    Lazy-load and return the requested face embedding model.

    Models are cached after first load to improve performance. This function
    supports loading different types of face recognition models for generating
    face embeddings used in face verification.

    Args:
        model_name (str): Name of the model to load. Currently supports:
            - "Inception" (default): Uses the default inception_model.keras
            - "Facenet": Alternative model (currently maps to same file)

    Returns:
        tensorflow.keras.Model: Loaded and compiled face embedding model

    Raises:
        FileNotFoundError: If the specified model file doesn't exist

    Example:
        >>> model = get_model("Inception")
        >>> # Model is now cached for subsequent calls
        >>> same_model = get_model("Inception")  # Returns cached instance
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    # Map model names to file paths
    if model_name.lower() == "facenet":
        model_path = Path(__file__).resolve().parent / "inception_model.keras"
    else:
        model_path = DEFAULT_MODEL_PATH

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Embedding model not found at {model_path}")

    model = load_model(str(model_path))
    _model_cache[model_name] = model
    return model


def extract_faces(group_path: Path,
                  out_dir: Path,
                  min_size: int = MIN_SIZE) -> list[Path]:
    """
    Extract all faces from a group image using MTCNN face detection.

    This function detects faces in the input image and saves each detected face
    as a separate image file. Small faces are automatically upscaled to meet
    the minimum size requirement for better recognition accuracy.

    Args:
        group_path (Path): Path to the input group image file
        out_dir (Path): Directory where extracted face images will be saved
        min_size (int): Minimum face size in pixels. Faces smaller than this
                       will be upscaled using cubic interpolation

    Returns:
        list[Path]: List of paths to the saved face image files

    Raises:
        ValueError: If the input image cannot be read

    Example:
        >>> faces = extract_faces(Path("group.jpg"), Path("faces/"), min_size=112)
        >>> print(f"Extracted {len(faces)} faces")
        Extracted 3 faces → faces/
    """
    out_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(str(group_path))

    if img_bgr is None:
        raise ValueError(f"Could not read image from {group_path}")

    # Convert BGR to RGB for MTCNN (which expects RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    detections = detector.detect_faces(img_rgb)
    saved = []

    for i, det in enumerate(detections):
        # Extract bounding box coordinates
        x, y, w, h = det["box"]
        x, y = max(0, x), max(0, y)  # Ensure coordinates are not negative

        # Crop the face from the image
        face = img_rgb[y: y + h, x: x + w]

        # Upscale small faces to minimum size for better recognition
        if face.shape[0] < min_size or face.shape[1] < min_size:
            face = cv2.resize(face, (min_size, min_size),
                              interpolation=cv2.INTER_CUBIC)

        # Save the face image (convert back to BGR for saving)
        out_path = out_dir / f"face_{i}.jpg"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    print(f"Extracted {len(saved)} faces → {out_dir}")
    return saved


def get_embedding(img_path: Path, model_name: str = "Inception") -> np.ndarray:
    """
    Generate a face embedding vector from an image using a deep learning model.

    This function loads an image, preprocesses it to match the model's expected
    input format, and generates a numerical embedding that represents the face
    for comparison purposes.

    Args:
        img_path (Path): Path to the face image file
        model_name (str): Name of the model to use for embedding generation

    Returns:
        np.ndarray: Face embedding vector (1D numpy array)

    Raises:
        ValueError: If the image cannot be read

    Example:
        >>> embedding = get_embedding(Path("face.jpg"))
        >>> print(f"Embedding shape: {embedding.shape}")
        Embedding shape: (512,)
    """
    model = get_model(model_name)

    img_bgr = cv2.imread(str(img_path))

    if img_bgr is None:
        raise ValueError(f"Could not read image from {img_path}")

    # Convert to RGB as expected by the model
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model's expected input dimensions (typically square)
    _, H, W, _ = model.input_shape
    face = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    # Normalize pixel values to [0, 1] range
    face = face.astype("float32") / 255.0
    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    # Generate embedding
    emb = model.predict(face, verbose=0)
    return emb[0]  # Return first (and only) embedding from batch


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine distance between two embedding vectors.

    Cosine distance is defined as 1 - cosine_similarity, where cosine similarity
    is the dot product of normalized vectors. This metric is commonly used for
    face verification as it's less sensitive to vector magnitude variations.

    Args:
        a (np.ndarray): First embedding vector
        b (np.ndarray): Second embedding vector

    Returns:
        float: Cosine distance between the vectors (0 = identical, 2 = opposite)

    Note:
        - Distance of 0 means vectors are identical
        - Distance of 1 means vectors are orthogonal (90 degrees apart)
        - Distance of 2 means vectors are opposite (180 degrees apart)

    Example:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([1, 0, 0])
        >>> distance = cosine_distance(emb1, emb2)
        >>> print(f"Distance: {distance}")  # Should be close to 0
    """
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def verify_faces(face_paths: list[Path],
                 person_path: Path,
                 threshold: float = THRESHOLD,
                 model_name: str = "Inception") -> tuple[int, int]:
    """
    Verify extracted faces against a known person's reference image.

    This function compares each detected face against a reference image of a
    known person using face embeddings and cosine distance. Faces with distance
    below the threshold are considered matches.

    Args:
        face_paths (list[Path]): List of paths to extracted face images
        person_path (Path): Path to the reference image of the known person
        threshold (float): Cosine distance threshold for matching (lower = stricter)
        model_name (str): Name of the embedding model to use

    Returns:
        tuple[int, int]: (number_of_matches, number_of_non_matches)

    Example:
        >>> faces = [Path("face_1.jpg"), Path("face_2.jpg")]
        >>> matches, misses = verify_faces(faces, Path("alice.jpg"))
        >>> print(f"Found {matches} matches out of {len(faces)} faces")
    """
    # Get the reference embedding for the known person
    person_emb = get_embedding(person_path, model_name)
    matches = 0
    misses = 0

    for p in face_paths:
        start = time.time()
        # Get embedding for the current detected face
        face_emb = get_embedding(p, model_name)
        # Calculate similarity using cosine distance
        dist = cosine_distance(face_emb, person_emb)
        # Check if the distance is below threshold (indicating a match)
        is_match = (dist <= threshold)
        elapsed = time.time() - start

        print(f"  {p.name}: dist={dist:.3f}, match={is_match} ({elapsed:.2f}s)")

        if is_match:
            matches += 1
        else:
            misses += 1

    return matches, misses


def find_people_in_group_simple(
        group_img: Path,
        person_directory: Path,
        *,
        verbose: bool = True,
        model_name: str = "Inception"
) -> str:
    """
    Main function to identify known people in a group photograph.

    This is the primary interface for the face recognition system. It extracts
    faces from a group image, then compares each face against all known people
    in the person directory to determine who is present in the photo.

    Args:
        group_img (Path): Path to the group photograph to analyze
        person_directory (Path): Directory containing reference images of known people
        verbose (bool): Whether to print detailed progress information
        model_name (str): Name of the embedding model to use for recognition

    Returns:
        str: JSON-formatted string containing the results with structure:
            {
                "status": bool/str,           # True for success, "error" for failure
                "found": list[str],           # Names of people found (filenames without extension)
                "total_faces": int,           # Total number of faces detected
                "message": str                # Error message if status is "error"
            }

    Example:
        >>> result_json = find_people_in_group_simple(
        ...     Path("party.jpg"),
        ...     Path("known_people/")
        ... )
        >>> result = json.loads(result_json)
        >>> print(f"Found: {result['found']}")
        Found: ['alice', 'bob', 'charlie']

    Workflow:
        1. Validate input files and directories
        2. Extract faces from the group image using MTCNN
        3. For each known person in the directory:
           a. Generate their reference embedding
           b. Compare against all detected faces
           c. If any face matches (distance < threshold), mark as found
        4. Return JSON results
    """
    # Create directory for temporarily storing extracted faces
    extract_dir = Path(__file__).resolve().parent / "extracted_faces"

    # Input validation
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

    # Clean up any previous extracted faces
    if extract_dir.exists():
        for f in extract_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract all faces from the group image
    faces = extract_faces(group_img, extract_dir, MIN_SIZE)

    # If no faces found, return early
    if not faces:
        response = {
            "status": True,
            "found": [],
            "total_faces": 0
        }
        return json.dumps(response, indent=2)

    if verbose:
        print("\n=== Checking each person image ===")

    # Step 2: Compare extracted faces against each known person
    found_people: List[str] = []
    for person_file in sorted(os.listdir(person_directory)):
        person_path = person_directory / person_file
        if not person_path.is_file():
            continue

        # Use filename without extension as the person's name
        name_out = person_path.stem

        if verbose:
            print(f"\n-- {person_file} --")

        # Verify if this person appears in any of the extracted faces
        matches, misses = verify_faces(faces, person_path,
                                       threshold=THRESHOLD,
                                       model_name=model_name)

        # If any face matched this person, add them to found list
        if matches > 0:
            if verbose:
                print(f"=> **{person_file} FOUND** ({matches} of {len(faces)} faces matched)")
            found_people.append(name_out)
        else:
            if verbose:
                print(f"=> {person_file} NOT found")

    # Prepare final response
    response = {
        "status": True,
        "found": found_people,
        "total_faces": len(faces)
    }
    return json.dumps(response, indent=2)


if __name__ == "__main__":
    """
    Example usage and demonstration of the FaceTrack module.

    This script demonstrates how to use the face recognition system with
    the provided test images (Avengers characters). It shows the typical
    workflow for identifying people in a group photo.
    """
    # Setup paths for demonstration
    base_dir = Path(__file__).resolve().parent
    group_img = base_dir / "avengersGroup" / "robertTest2.jpg"
    person_dir = base_dir / "avengersTest"

    # Run the face recognition system
    result_json = find_people_in_group_simple(group_img, person_dir)
    result = json.loads(result_json)

    # Display results
    print("\n=== Summary: People detected in group photo ===")
    if result["status"] and result["found"]:
        for name in result["found"]:
            print(f"- {name}")
    else:
        print("No known persons detected.")