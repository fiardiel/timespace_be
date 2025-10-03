# FaceTrack.py - Face Detection and Recognition Module

## Overview

FaceTrack.py is a comprehensive face detection and recognition module designed for identifying known individuals in group photographs. It leverages state-of-the-art computer vision techniques including MTCNN for face detection and deep learning models for face recognition.

## Features

- 🔍 **Face Detection**: Automatically detect faces in group photos using MTCNN
- 🔄 **Smart Upscaling**: Automatically upscale small faces to ensure minimum resolution (112×112)
- 🧠 **Deep Learning Recognition**: Generate face embeddings using pre-trained models
- ✅ **Face Verification**: Match detected faces against known individuals using cosine similarity
- 📊 **JSON Results**: Return structured JSON results with identification information
- ⚡ **Model Caching**: Efficient model loading with caching for improved performance

## Installation

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `opencv-python` (4.10.0.84) - Computer vision operations
- `numpy` (1.26.4) - Numerical computations
- `tensorflow` (2.18.1) - Deep learning framework
- `keras` (3.9.2) - High-level neural networks API
- `mtcnn` (1.0.0) - Multi-task CNN for face detection
- `pillow` (11.0.0) - Image processing

### Model Files

Ensure the following model file is present in the facetrack directory:
- `inception_model.keras` - Pre-trained face embedding model

## Quick Start

### Basic Usage

```python
from pathlib import Path
import json
from FaceTrack import find_people_in_group_simple

# Set up paths
group_image = Path("path/to/group_photo.jpg")
known_people_dir = Path("path/to/known_people_directory/")

# Identify people in the group photo
result_json = find_people_in_group_simple(group_image, known_people_dir)
result = json.loads(result_json)

# Display results
print(f"Found {len(result['found'])} people:")
for person in result['found']:
    print(f"- {person}")
```

### Directory Structure

Organize your files as follows:

```
your_project/
├── group_photos/
│   └── party.jpg
├── known_people/
│   ├── alice.jpg
│   ├── bob.jpg
│   └── charlie.jpg
└── FaceTrack.py
```

## API Reference

### Main Functions

#### `find_people_in_group_simple(group_img, person_directory, *, verbose=True, model_name="Inception")`

The primary interface for face recognition.

**Parameters:**
- `group_img` (Path): Path to the group photograph
- `person_directory` (Path): Directory containing reference images of known people
- `verbose` (bool): Enable detailed progress output (default: True)
- `model_name` (str): Embedding model to use (default: "Inception")

**Returns:**
- `str`: JSON-formatted results

**Example Response:**
```json
{
  "status": true,
  "found": ["alice", "bob"],
  "total_faces": 5
}
```

#### `extract_faces(group_path, out_dir, min_size=112)`

Extract all faces from a group image.

**Parameters:**
- `group_path` (Path): Input group image path
- `out_dir` (Path): Output directory for extracted faces
- `min_size` (int): Minimum face size in pixels (default: 112)

**Returns:**
- `list[Path]`: Paths to extracted face images

#### `get_embedding(img_path, model_name="Inception")`

Generate face embedding from an image.

**Parameters:**
- `img_path` (Path): Path to face image
- `model_name` (str): Model name (default: "Inception")

**Returns:**
- `np.ndarray`: Face embedding vector

#### `verify_faces(face_paths, person_path, threshold=0.4, model_name="Inception")`

Verify extracted faces against a known person.

**Parameters:**
- `face_paths` (list[Path]): Extracted face image paths
- `person_path` (Path): Reference image of known person
- `threshold` (float): Cosine distance threshold (default: 0.4)
- `model_name` (str): Model name (default: "Inception")

**Returns:**
- `tuple[int, int]`: (matches, non_matches)

### Utility Functions

#### `cosine_distance(a, b)`

Calculate cosine distance between two embedding vectors.

#### `get_model(model_name="Inception")`

Load and cache embedding model.

## Configuration

### Constants

You can modify these constants at the top of the file:

```python
MIN_SIZE = 112      # Minimum face size in pixels
THRESHOLD = 0.4     # Cosine distance threshold for matching
```

### Threshold Tuning

- **Lower threshold (0.2-0.3)**: Stricter matching, fewer false positives
- **Higher threshold (0.5-0.6)**: More lenient matching, more potential matches

## Workflow

1. **Input Validation**: Verify group image and person directory exist
2. **Face Detection**: Use MTCNN to detect all faces in the group photo
3. **Face Extraction**: Save detected faces as individual images
4. **Face Upscaling**: Resize small faces to minimum required size
5. **Embedding Generation**: Create numerical representations of faces
6. **Face Verification**: Compare against known people using cosine similarity
7. **Result Compilation**: Return JSON with identification results

## Performance Considerations

- **Model Caching**: Models are cached after first load for better performance
- **Batch Processing**: Process multiple faces efficiently
- **Memory Management**: Automatic cleanup of temporary face extractions
- **GPU Support**: Leverages TensorFlow GPU acceleration when available

## Error Handling

The module includes comprehensive error handling:

- Missing image files
- Invalid directory paths
- Corrupted image data
- Model loading failures

Error responses follow this format:
```json
{
  "status": "error",
  "message": "Descriptive error message"
}
```

## Example Output

```
Extracted 4 faces → extracted_faces/

=== Checking each person image ===

-- alice.jpg --
  face_0.jpg: dist=0.234, match=True (0.15s)
  face_1.jpg: dist=0.567, match=False (0.14s)
  face_2.jpg: dist=0.789, match=False (0.16s)
  face_3.jpg: dist=0.623, match=False (0.15s)
=> **alice.jpg FOUND** (1 of 4 faces matched)

-- bob.jpg --
  face_0.jpg: dist=0.698, match=False (0.14s)
  face_1.jpg: dist=0.312, match=True (0.15s)
  face_2.jpg: dist=0.845, match=False (0.14s)
  face_3.jpg: dist=0.734, match=False (0.16s)
=> **bob.jpg FOUND** (1 of 4 faces matched)

=== Summary: People detected in group photo ===
- alice
- bob
```

## Testing

Run the module directly for testing:

```bash
python FaceTrack.py
```

This will use the test images in:
- `GroupPhotoFromFE/` - Group photos
- `PeopleFromDataBase/` - Known people references

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure all dependencies are installed
2. **Model not found**: Verify `inception_model.keras` exists
3. **No faces detected**: Check image quality and lighting
4. **Poor recognition**: Adjust threshold or improve reference images

### Best Practices

- Use high-quality, well-lit reference images
- Ensure faces are clearly visible and front-facing
- Use consistent image formats (JPEG, PNG)
- Keep reference images recent and representative

## License

This module is part of the TIME-Space Machine Learning project developed for COMP4500.

## Authors

TIME-Space Machine Learning Team

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please contact the development team or create an issue in the project repository.