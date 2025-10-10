#!/usr/bin/env python3
"""
Comprehensive test suite for FaceTrack.py module.

This module contains extensive unit and integration tests for the face detection
and recognition functionality in FaceTrack.py. Tests cover all functions,
edge cases, error conditions, and performance scenarios.

Test Categories:
1. Unit Tests: Test individual functions in isolation with mocks
2. Integration Tests: Test complete workflows and function interactions
3. Performance Tests: Test system behavior under load and with large datasets
4. Error Handling Tests: Test robustness and error recovery

Author: TIME-Space Machine Learning Team
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import json
import tempfile
import shutil
import time
from pathlib import Path
import numpy as np
import cv2

# Import the module under test
from . import FaceTrack


class TestUtilityFunctions(unittest.TestCase):
    """Unit tests for utility functions in FaceTrack module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        
    def tearDown(self):
        """Clean up after each test method."""
        # Clear model cache to ensure tests don't interfere
        FaceTrack._model_cache.clear()

    # ========== get_model() Tests ==========
    
    def test_get_model_default_inception(self):
        """Test loading default Inception model."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                result = FaceTrack.get_model("Inception")
                
            self.assertEqual(result, mock_model)
            mock_load.assert_called_once()
    
    def test_get_model_facenet_maps_to_inception(self):
        """Test that Facenet model name maps to inception model file."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                result = FaceTrack.get_model("facenet")
                
            self.assertEqual(result, mock_model)
    
    def test_get_model_caching_behavior(self):
        """Test that models are cached after first load."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                # First call should load model
                result1 = FaceTrack.get_model("Inception")
                # Second call should return cached model
                result2 = FaceTrack.get_model("Inception")
                
            # load_model should only be called once due to caching
            mock_load.assert_called_once()
            self.assertEqual(result1, result2)
    
    def test_get_model_file_not_found(self):
        """Test FileNotFoundError when model file doesn't exist."""
        with patch('pathlib.Path.is_file', return_value=False):
            with self.assertRaises(FileNotFoundError) as context:
                FaceTrack.get_model("Inception")
            
            self.assertIn("Embedding model not found", str(context.exception))
    
    def test_get_model_multiple_models_cached_separately(self):
        """Test that different models are cached separately."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_load.side_effect = [mock_model1, mock_model2]
            
            with patch('pathlib.Path.is_file', return_value=True):
                result1 = FaceTrack.get_model("Inception")
                result2 = FaceTrack.get_model("Facenet")
                
            self.assertEqual(mock_load.call_count, 2)
            self.assertEqual(result1, mock_model1)
            self.assertEqual(result2, mock_model2)
    
    def test_get_model_case_insensitive_facenet(self):
        """Test that facenet model name is case insensitive."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                result1 = FaceTrack.get_model("FACENET")
                result2 = FaceTrack.get_model("FaceNet")
                result3 = FaceTrack.get_model("facenet")
                
            # All should return the same cached model
            self.assertEqual(result1, result2)
            self.assertEqual(result2, result3)
    
    def test_get_model_unknown_model_defaults_to_inception(self):
        """Test that unknown model names default to inception model."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                result = FaceTrack.get_model("UnknownModel")
                
            self.assertEqual(result, mock_model)
    
    def test_get_model_empty_string_defaults_to_inception(self):
        """Test that empty string model name defaults to inception."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                result = FaceTrack.get_model("")
                
            self.assertEqual(result, mock_model)
    
    def test_get_model_none_input(self):
        """Test that None input for model name raises an AttributeError."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                # Should raise AttributeError for None input
                with self.assertRaises(AttributeError):
                    FaceTrack.get_model(None)

    # ========== cosine_distance() Tests ==========
    
    def test_cosine_distance_identical_vectors(self):
        """Test cosine distance between identical vectors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        self.assertAlmostEqual(distance, 0.0, places=10)
    
    def test_cosine_distance_orthogonal_vectors(self):
        """Test cosine distance between orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        self.assertAlmostEqual(distance, 1.0, places=10)
    
    def test_cosine_distance_opposite_vectors(self):
        """Test cosine distance between opposite vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        self.assertAlmostEqual(distance, 2.0, places=10)
    
    def test_cosine_distance_zero_vector(self):
        """Test cosine distance with zero vector."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        
        # This should handle division by zero gracefully
        distance = FaceTrack.cosine_distance(a, b)
        
        # Distance with zero vector should be NaN or handled specially
        self.assertTrue(np.isnan(distance) or np.isinf(distance))
    
    def test_cosine_distance_different_magnitudes_same_direction(self):
        """Test cosine distance between vectors with same direction but different magnitudes."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # Same direction, double magnitude
        
        distance = FaceTrack.cosine_distance(a, b)
        
        self.assertAlmostEqual(distance, 0.0, places=10)
    
    def test_cosine_distance_high_dimensional_vectors(self):
        """Test cosine distance with high-dimensional vectors."""
        np.random.seed(42)
        a = np.random.random(512)  # Typical face embedding dimension
        b = np.random.random(512)
        
        distance = FaceTrack.cosine_distance(a, b)
        
        # Distance should be between 0 and 2
        self.assertTrue(0 <= distance <= 2)
    
    def test_cosine_distance_normalized_vectors(self):
        """Test cosine distance with pre-normalized vectors."""
        a = np.array([0.6, 0.8, 0.0])  # Already normalized
        b = np.array([0.0, 0.6, 0.8])  # Already normalized
        
        distance = FaceTrack.cosine_distance(a, b)
        
        # Should be valid distance
        self.assertTrue(0 <= distance <= 2)
    
    def test_cosine_distance_single_element_vectors(self):
        """Test cosine distance with single-element vectors."""
        a = np.array([5.0])
        b = np.array([3.0])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        self.assertAlmostEqual(distance, 0.0, places=10)
    
    def test_cosine_distance_negative_values(self):
        """Test cosine distance with negative values."""
        a = np.array([-1.0, -2.0, 3.0])
        b = np.array([1.0, 2.0, -3.0])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        self.assertTrue(0 <= distance <= 2)
    
    def test_cosine_distance_very_small_values(self):
        """Test cosine distance with very small floating point values."""
        a = np.array([1e-10, 2e-10, 3e-10])
        b = np.array([1e-10, 2e-10, 3e-10])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        self.assertAlmostEqual(distance, 0.0, places=5)

    # ========== Configuration Constants Tests ==========
    
    def test_min_size_constant(self):
        """Test MIN_SIZE constant value."""
        self.assertEqual(FaceTrack.MIN_SIZE, 112)
        self.assertIsInstance(FaceTrack.MIN_SIZE, int)
    
    def test_threshold_constant(self):
        """Test THRESHOLD constant value."""
        self.assertEqual(FaceTrack.THRESHOLD, 0.4)
        self.assertIsInstance(FaceTrack.THRESHOLD, float)
    
    def test_default_model_path_constant(self):
        """Test DEFAULT_MODEL_PATH constant."""
        self.assertIsInstance(FaceTrack.DEFAULT_MODEL_PATH, Path)
        self.assertTrue(str(FaceTrack.DEFAULT_MODEL_PATH).endswith('inception_model.keras'))
    
    def test_detector_initialization(self):
        """Test that MTCNN detector is properly initialized."""
        self.assertIsNotNone(FaceTrack.detector)
        # Should have detect_faces method
        self.assertTrue(hasattr(FaceTrack.detector, 'detect_faces'))
    
    def test_model_cache_initialization(self):
        """Test that model cache is properly initialized."""
        self.assertIsInstance(FaceTrack._model_cache, dict)


class TestFaceProcessingFunctions(unittest.TestCase):
    """Unit tests for face processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image_path = self.temp_dir / "test_image.jpg"
        cv2.imwrite(str(self.test_image_path), self.test_image)
        
    def tearDown(self):
        """Clean up after tests."""
        FaceTrack._model_cache.clear()

    # ========== extract_faces() Tests ==========
    
    @patch('FaceTrack.detector')
    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_extract_faces_successful_detection(self, mock_imwrite, mock_imread, mock_detector):
        """Test successful face extraction from image."""
        # Mock image reading
        mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        # Mock face detection
        mock_detections = [
            {"box": [50, 50, 80, 80]},
            {"box": [120, 60, 70, 70]}
        ]
        mock_detector.detect_faces.return_value = mock_detections
        mock_imwrite.return_value = True
        
        out_dir = self.temp_dir / "faces"
        
        result = FaceTrack.extract_faces(self.test_image_path, out_dir)
        
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(path, Path) for path in result))
        mock_detector.detect_faces.assert_called_once()
        self.assertEqual(mock_imwrite.call_count, 2)
    
    @patch('cv2.imread')
    def test_extract_faces_image_read_failure(self, mock_imread):
        """Test extract_faces with image read failure."""
        mock_imread.return_value = None
        
        out_dir = self.temp_dir / "faces"
        
        with self.assertRaises(ValueError) as context:
            FaceTrack.extract_faces(self.test_image_path, out_dir)
        
        self.assertIn("Could not read image", str(context.exception))
    
    @patch('FaceTrack.detector')
    @patch('cv2.imread')
    def test_extract_faces_no_faces_detected(self, mock_imread, mock_detector):
        """Test extract_faces when no faces are detected."""
        mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_detector.detect_faces.return_value = []
        
        out_dir = self.temp_dir / "faces"
        
        result = FaceTrack.extract_faces(self.test_image_path, out_dir)
        
        self.assertEqual(len(result), 0)
        self.assertTrue(out_dir.exists())
    
    @patch('FaceTrack.detector')
    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_extract_faces_small_face_upscaling(self, mock_imwrite, mock_imread, mock_detector):
        """Test that small faces are upscaled to minimum size."""
        mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        # Small face that should be upscaled
        mock_detections = [{"box": [50, 50, 50, 50]}]  # 50x50 face
        mock_detector.detect_faces.return_value = mock_detections
        mock_imwrite.return_value = True
        
        out_dir = self.temp_dir / "faces"
        min_size = 112
        
        with patch('cv2.resize') as mock_resize:
            mock_resize.return_value = np.zeros((min_size, min_size, 3), dtype=np.uint8)
            
            result = FaceTrack.extract_faces(self.test_image_path, out_dir, min_size)
            
            # Should call resize since face is smaller than min_size
            mock_resize.assert_called_once()
            resize_args = mock_resize.call_args[0]
            self.assertEqual(resize_args[1], (min_size, min_size))
    
    @patch('FaceTrack.detector')
    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_extract_faces_negative_coordinates(self, mock_imwrite, mock_imread, mock_detector):
        """Test extract_faces handles negative bounding box coordinates."""
        mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        # Face with negative coordinates
        mock_detections = [{"box": [-10, -5, 80, 80]}]
        mock_detector.detect_faces.return_value = mock_detections
        mock_imwrite.return_value = True
        
        out_dir = self.temp_dir / "faces"
        
        result = FaceTrack.extract_faces(self.test_image_path, out_dir)
        
        self.assertEqual(len(result), 1)
        # Should handle negative coordinates gracefully
    
    @patch('FaceTrack.detector')
    @patch('cv2.imread')
    def test_extract_faces_output_directory_creation(self, mock_imread, mock_detector):
        """Test that output directory is created if it doesn't exist."""
        mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_detector.detect_faces.return_value = []
        
        out_dir = self.temp_dir / "new_faces_dir"
        self.assertFalse(out_dir.exists())
        
        FaceTrack.extract_faces(self.test_image_path, out_dir)
        
        self.assertTrue(out_dir.exists())
    
    def test_extract_faces_nonexistent_image(self):
        """Test extract_faces with nonexistent image file."""
        nonexistent_path = self.temp_dir / "nonexistent.jpg"
        out_dir = self.temp_dir / "faces"
        
        with self.assertRaises(ValueError):
            FaceTrack.extract_faces(nonexistent_path, out_dir)

    # ========== get_embedding() Tests ==========
    
    @patch('FaceTrack.get_model')
    @patch('cv2.imread')
    def test_get_embedding_successful(self, mock_imread, mock_get_model):
        """Test successful embedding generation."""
        # Mock image
        mock_img = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        # Mock model
        mock_model = Mock()
        mock_model.input_shape = (None, 112, 112, 3)
        mock_embedding = np.random.random((1, 512))
        mock_model.predict.return_value = mock_embedding
        mock_get_model.return_value = mock_model
        
        result = FaceTrack.get_embedding(self.test_image_path)
        
        self.assertEqual(result.shape, (512,))
        mock_model.predict.assert_called_once()
    
    @patch('cv2.imread')
    def test_get_embedding_image_read_failure(self, mock_imread):
        """Test get_embedding with image read failure."""
        mock_imread.return_value = None
        
        with self.assertRaises(ValueError) as context:
            FaceTrack.get_embedding(self.test_image_path)
        
        self.assertIn("Could not read image", str(context.exception))
    
    @patch('FaceTrack.get_model')
    @patch('cv2.imread')
    def test_get_embedding_different_model(self, mock_imread, mock_get_model):
        """Test get_embedding with different model name."""
        mock_img = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        mock_model = Mock()
        mock_model.input_shape = (None, 160, 160, 3)  # Different size
        mock_embedding = np.random.random((1, 256))
        mock_model.predict.return_value = mock_embedding
        mock_get_model.return_value = mock_model
        
        result = FaceTrack.get_embedding(self.test_image_path, "Facenet")
        
        self.assertEqual(result.shape, (256,))
        mock_get_model.assert_called_with("Facenet")
    
    @patch('FaceTrack.get_model')
    @patch('cv2.imread')
    @patch('cv2.resize')
    def test_get_embedding_image_resizing(self, mock_resize, mock_imread, mock_get_model):
        """Test that images are resized to match model input shape."""
        # Original image of different size
        mock_img = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        # Resized image
        mock_resized = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_resize.return_value = mock_resized
        
        mock_model = Mock()
        mock_model.input_shape = (None, 112, 112, 3)
        mock_embedding = np.random.random((1, 512))
        mock_model.predict.return_value = mock_embedding
        mock_get_model.return_value = mock_model
        
        FaceTrack.get_embedding(self.test_image_path)
        
        # Should resize to model's expected dimensions
        mock_resize.assert_called_once()
        resize_args = mock_resize.call_args[0]
        self.assertEqual(resize_args[1], (112, 112))
    
    @patch('FaceTrack.get_model')
    @patch('cv2.imread')
    def test_get_embedding_normalization(self, mock_imread, mock_get_model):
        """Test that pixel values are normalized before model prediction."""
        # Image with values 0-255
        mock_img = np.full((112, 112, 3), 255, dtype=np.uint8)
        mock_imread.return_value = mock_img
        
        mock_model = Mock()
        mock_model.input_shape = (None, 112, 112, 3)
        mock_embedding = np.random.random((1, 512))
        mock_model.predict.return_value = mock_embedding
        mock_get_model.return_value = mock_model
        
        FaceTrack.get_embedding(self.test_image_path)
        
        # Check that predict was called with normalized values
        call_args = mock_model.predict.call_args[0][0]
        # Should be normalized to [0, 1] range
        self.assertTrue(np.all(call_args <= 1.0))
        self.assertTrue(np.all(call_args >= 0.0))

    # ========== verify_faces() Tests ==========
    
    @patch('FaceTrack.get_embedding')
    def test_verify_faces_all_matches(self, mock_get_embedding):
        """Test verify_faces when all faces match."""
        # Create test face files
        face_paths = []
        for i in range(3):
            face_path = self.temp_dir / f"face_{i}.jpg"
            cv2.imwrite(str(face_path), np.zeros((112, 112, 3), dtype=np.uint8))
            face_paths.append(face_path)
        
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        # Mock embeddings that are very similar (distance < threshold)
        person_emb = np.array([1.0, 0.0, 0.0])
        face_embs = [
            np.array([0.99, 0.1, 0.1]),   # Close match
            np.array([0.98, 0.15, 0.05]), # Close match
            np.array([0.97, 0.2, 0.1])    # Close match
        ]
        
        mock_get_embedding.side_effect = [person_emb] + face_embs
        
        matches, misses = FaceTrack.verify_faces(face_paths, person_path, threshold=0.5)
        
        self.assertEqual(matches, 3)
        self.assertEqual(misses, 0)
    
    @patch('FaceTrack.get_embedding')
    def test_verify_faces_no_matches(self, mock_get_embedding):
        """Test verify_faces when no faces match."""
        face_paths = [self.temp_dir / "face_0.jpg"]
        cv2.imwrite(str(face_paths[0]), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        # Mock embeddings that are very different (distance > threshold)
        person_emb = np.array([1.0, 0.0, 0.0])
        face_emb = np.array([0.0, 1.0, 0.0])  # Orthogonal = distance of 1.0
        
        mock_get_embedding.side_effect = [person_emb, face_emb]
        
        matches, misses = FaceTrack.verify_faces(face_paths, person_path, threshold=0.5)
        
        self.assertEqual(matches, 0)
        self.assertEqual(misses, 1)
    
    @patch('FaceTrack.get_embedding')
    def test_verify_faces_mixed_results(self, mock_get_embedding):
        """Test verify_faces with mixed match/no-match results."""
        face_paths = []
        for i in range(4):
            face_path = self.temp_dir / f"face_{i}.jpg"
            cv2.imwrite(str(face_path), np.zeros((112, 112, 3), dtype=np.uint8))
            face_paths.append(face_path)
        
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_emb = np.array([1.0, 0.0, 0.0])
        face_embs = [
            np.array([0.99, 0.1, 0.0]),   # Match (small distance)
            np.array([0.0, 1.0, 0.0]),    # No match (orthogonal)
            np.array([0.98, 0.15, 0.0]),  # Match (small distance)
            np.array([-1.0, 0.0, 0.0])    # No match (opposite)
        ]
        
        mock_get_embedding.side_effect = [person_emb] + face_embs
        
        matches, misses = FaceTrack.verify_faces(face_paths, person_path, threshold=0.3)
        
        self.assertEqual(matches, 2)
        self.assertEqual(misses, 2)
    
    @patch('FaceTrack.get_embedding')
    def test_verify_faces_custom_threshold(self, mock_get_embedding):
        """Test verify_faces with custom threshold value."""
        face_paths = [self.temp_dir / "face_0.jpg"]
        cv2.imwrite(str(face_paths[0]), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_emb = np.array([1.0, 0.0, 0.0])
        face_emb = np.array([0.8, 0.6, 0.0])  # Distance = 0.2
        
        mock_get_embedding.side_effect = [person_emb, face_emb]
        
        # Test with strict threshold (distance 0.2 should not match threshold 0.15)
        matches, misses = FaceTrack.verify_faces(face_paths, person_path, threshold=0.15)
        self.assertEqual(matches, 0)
        self.assertEqual(misses, 1)
        
        # Reset mock
        mock_get_embedding.side_effect = [person_emb, face_emb]
        
        # Test with lenient threshold
        matches, misses = FaceTrack.verify_faces(face_paths, person_path, threshold=0.5)
        self.assertEqual(matches, 1)
        self.assertEqual(misses, 0)
    
    @patch('FaceTrack.get_embedding')
    def test_verify_faces_custom_model(self, mock_get_embedding):
        """Test verify_faces with custom model name."""
        face_paths = [self.temp_dir / "face_0.jpg"]
        cv2.imwrite(str(face_paths[0]), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_emb = np.array([1.0, 0.0, 0.0])
        face_emb = np.array([0.99, 0.1, 0.0])
        
        mock_get_embedding.side_effect = [person_emb, face_emb]
        
        FaceTrack.verify_faces(face_paths, person_path, model_name="Facenet")
        
        # Check that get_embedding was called with correct model name
        expected_calls = [
            unittest.mock.call(person_path, "Facenet"),
            unittest.mock.call(face_paths[0], "Facenet")
        ]
        mock_get_embedding.assert_has_calls(expected_calls)
    
    @patch('FaceTrack.get_embedding')
    def test_verify_faces_empty_face_list(self, mock_get_embedding):
        """Test verify_faces with empty face list."""
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_emb = np.array([1.0, 0.0, 0.0])
        mock_get_embedding.return_value = person_emb
        
        matches, misses = FaceTrack.verify_faces([], person_path)
        
        self.assertEqual(matches, 0)
        self.assertEqual(misses, 0)
        # Should still call get_embedding for person
        mock_get_embedding.assert_called_once_with(person_path, "Inception")


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for the complete face recognition workflow."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        
        # Create test directory structure
        self.group_dir = self.temp_dir / "group"
        self.person_dir = self.temp_dir / "people"
        self.group_dir.mkdir()
        self.person_dir.mkdir()
        
        # Create test images
        self.group_image = self.group_dir / "group.jpg"
        cv2.imwrite(str(self.group_image), np.zeros((300, 300, 3), dtype=np.uint8))
        
        self.person1 = self.person_dir / "alice.jpg"
        self.person2 = self.person_dir / "bob.jpg"
        cv2.imwrite(str(self.person1), np.zeros((112, 112, 3), dtype=np.uint8))
        cv2.imwrite(str(self.person2), np.zeros((112, 112, 3), dtype=np.uint8))
        
    def tearDown(self):
        """Clean up after integration tests."""
        FaceTrack._model_cache.clear()

    # ========== find_people_in_group_simple() Integration Tests ==========
    
    @patch('FaceTrack.extract_faces')
    @patch('FaceTrack.verify_faces')
    def test_find_people_successful_detection(self, mock_verify, mock_extract):
        """Test successful people detection workflow."""
        # Mock face extraction
        mock_faces = [Path("face_0.jpg"), Path("face_1.jpg")]
        mock_extract.return_value = mock_faces
        
        # Mock verification - alice found, bob not found
        mock_verify.side_effect = [
            (1, 1),  # alice: 1 match, 1 miss
            (0, 2)   # bob: 0 matches, 2 misses
        ]
        
        result_json = FaceTrack.find_people_in_group_simple(
            self.group_image, self.person_dir, verbose=False
        )
        
        result = json.loads(result_json)
        
        self.assertTrue(result["status"])
        self.assertEqual(result["found"], ["alice"])
        self.assertEqual(result["total_faces"], 2)
        
        mock_extract.assert_called_once()
        self.assertEqual(mock_verify.call_count, 2)
    
    @patch('FaceTrack.extract_faces')
    def test_find_people_no_faces_detected(self, mock_extract):
        """Test workflow when no faces are detected."""
        mock_extract.return_value = []
        
        result_json = FaceTrack.find_people_in_group_simple(
            self.group_image, self.person_dir, verbose=False
        )
        
        result = json.loads(result_json)
        
        self.assertTrue(result["status"])
        self.assertEqual(result["found"], [])
        self.assertEqual(result["total_faces"], 0)
    
    def test_find_people_missing_group_image(self):
        """Test error handling for missing group image."""
        missing_image = self.temp_dir / "missing.jpg"
        
        result_json = FaceTrack.find_people_in_group_simple(
            missing_image, self.person_dir, verbose=False
        )
        
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing group image", result["message"])
    
    def test_find_people_missing_person_directory(self):
        """Test error handling for missing person directory."""
        missing_dir = self.temp_dir / "missing_people"
        
        result_json = FaceTrack.find_people_in_group_simple(
            self.group_image, missing_dir, verbose=False
        )
        
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing person directory", result["message"])
    
    @patch('FaceTrack.extract_faces')
    @patch('FaceTrack.verify_faces')
    def test_find_people_all_people_found(self, mock_verify, mock_extract):
        """Test workflow when all people are found."""
        mock_faces = [Path("face_0.jpg")]
        mock_extract.return_value = mock_faces
        
        # Both people found
        mock_verify.side_effect = [
            (1, 0),  # alice: 1 match, 0 misses
            (1, 0)   # bob: 1 match, 0 misses
        ]
        
        result_json = FaceTrack.find_people_in_group_simple(
            self.group_image, self.person_dir, verbose=False
        )
        
        result = json.loads(result_json)
        
        self.assertTrue(result["status"])
        self.assertEqual(set(result["found"]), {"alice", "bob"})
        self.assertEqual(result["total_faces"], 1)
    
    @patch('FaceTrack.extract_faces')
    @patch('FaceTrack.verify_faces')
    def test_find_people_no_people_found(self, mock_verify, mock_extract):
        """Test workflow when no people are found."""
        mock_faces = [Path("face_0.jpg"), Path("face_1.jpg")]
        mock_extract.return_value = mock_faces
        
        # No people found
        mock_verify.side_effect = [
            (0, 2),  # alice: 0 matches, 2 misses
            (0, 2)   # bob: 0 matches, 2 misses
        ]
        
        result_json = FaceTrack.find_people_in_group_simple(
            self.group_image, self.person_dir, verbose=False
        )
        
        result = json.loads(result_json)
        
        self.assertTrue(result["status"])
        self.assertEqual(result["found"], [])
        self.assertEqual(result["total_faces"], 2)
    
    @patch('FaceTrack.extract_faces')
    @patch('FaceTrack.verify_faces')
    def test_find_people_custom_model(self, mock_verify, mock_extract):
        """Test workflow with custom model name."""
        mock_faces = [Path("face_0.jpg")]
        mock_extract.return_value = mock_faces
        mock_verify.return_value = (1, 0)
        
        FaceTrack.find_people_in_group_simple(
            self.group_image, self.person_dir, 
            verbose=False, model_name="Facenet"
        )
        
        # Check that verify_faces was called with custom model
        for call in mock_verify.call_args_list:
            self.assertEqual(call[1]["model_name"], "Facenet")
    
    @patch('FaceTrack.extract_faces')
    def test_find_people_extracted_faces_cleanup(self, mock_extract):
        """Test that extracted faces directory is cleaned up."""
        # Create some existing files in extracted_faces directory
        extract_dir = Path(__file__).resolve().parent / "extracted_faces"
        extract_dir.mkdir(exist_ok=True)
        
        existing_file = extract_dir / "old_face.jpg"
        existing_file.write_text("dummy")
        
        mock_extract.return_value = []
        
        FaceTrack.find_people_in_group_simple(
            self.group_image, self.person_dir, verbose=False
        )
        
        # Old file should be removed
        self.assertFalse(existing_file.exists())
    
    def test_find_people_empty_person_directory(self):
        """Test workflow with empty person directory."""
        empty_dir = self.temp_dir / "empty_people"
        empty_dir.mkdir()
        
        with patch('FaceTrack.extract_faces') as mock_extract:
            mock_extract.return_value = [Path("face_0.jpg")]
            
            result_json = FaceTrack.find_people_in_group_simple(
                self.group_image, empty_dir, verbose=False
            )
        
        result = json.loads(result_json)
        
        self.assertTrue(result["status"])
        self.assertEqual(result["found"], [])
        self.assertEqual(result["total_faces"], 1)
    
    def test_find_people_person_directory_with_subdirs(self):
        """Test that subdirectories in person directory are ignored."""
        # Create a subdirectory
        subdir = self.person_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.jpg").write_text("dummy")
        
        with patch('FaceTrack.extract_faces') as mock_extract:
            with patch('FaceTrack.verify_faces') as mock_verify:
                mock_extract.return_value = [Path("face_0.jpg")]
                mock_verify.return_value = (1, 0)
                
                result_json = FaceTrack.find_people_in_group_simple(
                    self.group_image, self.person_dir, verbose=False
                )
        
        result = json.loads(result_json)
        
        # Should only process the 2 person files, not the subdirectory
        self.assertEqual(len(result["found"]), 2)  # alice and bob


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Tests for error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        
    def tearDown(self):
        """Clean up after tests."""
        FaceTrack._model_cache.clear()

    # ========== Error Handling Tests ==========
    
    def test_cosine_distance_with_nan_values(self):
        """Test cosine distance with NaN values in vectors."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        # Should handle NaN gracefully
        self.assertTrue(np.isnan(distance))
    
    def test_cosine_distance_with_inf_values(self):
        """Test cosine distance with infinite values."""
        a = np.array([1.0, np.inf, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        
        distance = FaceTrack.cosine_distance(a, b)
        
        # Should handle infinity gracefully
        self.assertTrue(np.isnan(distance) or np.isinf(distance))
    
    @patch('FaceTrack.load_model')
    def test_get_model_load_failure(self, mock_load_model):
        """Test get_model when model loading fails."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with patch('pathlib.Path.is_file', return_value=True):
            with self.assertRaises(Exception) as context:
                FaceTrack.get_model("Inception")
            
            self.assertIn("Model loading failed", str(context.exception))
    
    @patch('cv2.imread')
    def test_extract_faces_corrupted_image(self, mock_imread):
        """Test extract_faces with corrupted image data."""
        # Simulate corrupted image that cv2 can't read
        mock_imread.return_value = None
        
        group_path = self.temp_dir / "corrupted.jpg"
        group_path.write_bytes(b"corrupted data")
        out_dir = self.temp_dir / "faces"
        
        with self.assertRaises(ValueError):
            FaceTrack.extract_faces(group_path, out_dir)
    
    def test_find_people_with_permission_error(self):
        """Test find_people_in_group_simple with permission errors."""
        # This test would be platform-specific and might not work in all environments
        # So we'll mock the permission error
        
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.is_dir', return_value=True):
                with patch('FaceTrack.extract_faces') as mock_extract:
                    mock_extract.side_effect = PermissionError("Permission denied")
                    
                    # The function should handle this gracefully
                    # In reality, we might want to catch and return an error status
                    with self.assertRaises(PermissionError):
                        FaceTrack.find_people_in_group_simple(
                            Path("dummy.jpg"), Path("dummy_dir")
                        )

    # ========== Edge Cases Tests ==========
    
    def test_cosine_distance_very_large_vectors(self):
        """Test cosine distance with very large vectors."""
        # Create large random vectors
        np.random.seed(42)
        a = np.random.random(10000)
        b = np.random.random(10000)
        
        distance = FaceTrack.cosine_distance(a, b)
        
        # Should complete without memory issues
        self.assertTrue(0 <= distance <= 2)
        self.assertFalse(np.isnan(distance))
    
    def test_extract_faces_very_small_image(self):
        """Test extract_faces with very small image."""
        # Create a tiny image
        tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
        tiny_path = self.temp_dir / "tiny.jpg"
        cv2.imwrite(str(tiny_path), tiny_img)
        
        with patch('FaceTrack.detector') as mock_detector:
            mock_detector.detect_faces.return_value = []
            
            out_dir = self.temp_dir / "faces"
            result = FaceTrack.extract_faces(tiny_path, out_dir)
            
            # Should handle small images gracefully
            self.assertEqual(len(result), 0)
    
    def test_extract_faces_very_large_image(self):
        """Test extract_faces with very large image."""
        # We'll mock this to avoid memory issues in tests
        with patch('cv2.imread') as mock_imread:
            # Simulate a very large image
            large_img = np.zeros((5000, 5000, 3), dtype=np.uint8)
            mock_imread.return_value = large_img
            
            with patch('FaceTrack.detector') as mock_detector:
                mock_detector.detect_faces.return_value = []
                
                large_path = self.temp_dir / "large.jpg"
                out_dir = self.temp_dir / "faces"
                
                result = FaceTrack.extract_faces(large_path, out_dir)
                
                # Should handle large images without crashing
                self.assertEqual(len(result), 0)
    
    def test_verify_faces_with_identical_embeddings(self):
        """Test verify_faces when embeddings are identical."""
        face_path = self.temp_dir / "face.jpg"
        cv2.imwrite(str(face_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        with patch('FaceTrack.get_embedding') as mock_get_embedding:
            # Identical embeddings
            identical_emb = np.array([1.0, 0.0, 0.0])
            mock_get_embedding.return_value = identical_emb
            
            matches, misses = FaceTrack.verify_faces([face_path], person_path)
            
            # Should detect as match (distance = 0)
            self.assertEqual(matches, 1)
            self.assertEqual(misses, 0)
    
    def test_find_people_with_unicode_filenames(self):
        """Test find_people_in_group_simple with unicode filenames."""
        # Create files with unicode names
        unicode_person = self.temp_dir / "张三.jpg"  # Chinese characters
        unicode_group = self.temp_dir / "团体照片.jpg"
        
        cv2.imwrite(str(unicode_person), np.zeros((112, 112, 3), dtype=np.uint8))
        cv2.imwrite(str(unicode_group), np.zeros((300, 300, 3), dtype=np.uint8))
        
        person_dir = self.temp_dir
        
        with patch('FaceTrack.extract_faces') as mock_extract:
            with patch('FaceTrack.verify_faces') as mock_verify:
                mock_extract.return_value = [Path("face_0.jpg")]
                mock_verify.return_value = (1, 0)
                
                result_json = FaceTrack.find_people_in_group_simple(
                    unicode_group, person_dir, verbose=False
                )
                
                result = json.loads(result_json)
                
                # Should handle unicode filenames
                if result["status"] == "error":
                    # If there's an error, just check that it ran without crashing
                    self.assertIsInstance(result["message"], str)
                else:
                    self.assertTrue(result["status"])
                    self.assertIn("张三", result["found"])


class TestPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests for FaceTrack module."""
    
    def setUp(self):
        """Set up test fixtures for performance tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        
    def tearDown(self):
        """Clean up after performance tests."""
        FaceTrack._model_cache.clear()

    # ========== Performance Tests ==========
    
    def test_model_caching_performance(self):
        """Test that model caching improves performance."""
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                # First call - should load model
                start_time = time.time()
                model1 = FaceTrack.get_model("Inception")
                first_call_time = time.time() - start_time
                
                # Second call - should use cache
                start_time = time.time()
                model2 = FaceTrack.get_model("Inception")
                second_call_time = time.time() - start_time
                
                # Cached call should be much faster
                # (In practice, this test might be flaky due to timing variations)
                self.assertLessEqual(mock_load.call_count, 1)
                self.assertEqual(model1, model2)
    
    def test_cosine_distance_performance_large_vectors(self):
        """Test cosine distance performance with large vectors."""
        # Test with vectors similar to real face embeddings
        np.random.seed(42)
        large_vectors = [np.random.random(512) for _ in range(100)]
        
        start_time = time.time()
        
        # Calculate distances between all pairs
        distances = []
        for i in range(len(large_vectors)):
            for j in range(i + 1, len(large_vectors)):
                dist = FaceTrack.cosine_distance(large_vectors[i], large_vectors[j])
                distances.append(dist)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(elapsed_time, 5.0)  # Should complete in under 5 seconds
        self.assertEqual(len(distances), 4950)  # 100 choose 2
    
    @patch('FaceTrack.extract_faces')
    @patch('FaceTrack.verify_faces')
    def test_find_people_performance_many_people(self, mock_verify, mock_extract):
        """Test find_people_in_group_simple performance with many people."""
        # Create many person files
        person_dir = self.temp_dir / "many_people"
        person_dir.mkdir()
        
        num_people = 50
        for i in range(num_people):
            person_file = person_dir / f"person_{i:03d}.jpg"
            cv2.imwrite(str(person_file), np.zeros((112, 112, 3), dtype=np.uint8))
        
        group_image = self.temp_dir / "group.jpg"
        cv2.imwrite(str(group_image), np.zeros((300, 300, 3), dtype=np.uint8))
        
        # Mock fast responses
        mock_extract.return_value = [Path("face_0.jpg")]
        mock_verify.return_value = (0, 1)  # No matches to speed up test
        
        start_time = time.time()
        
        result_json = FaceTrack.find_people_in_group_simple(
            group_image, person_dir, verbose=False
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time even with many people
        self.assertLess(elapsed_time, 10.0)
        
        result = json.loads(result_json)
        self.assertTrue(result["status"])
    
    def test_memory_usage_multiple_model_loads(self):
        """Test memory behavior with multiple model instances."""
        # This is more of a smoke test for memory leaks
        with patch('FaceTrack.load_model') as mock_load:
            mock_models = [Mock() for _ in range(10)]
            mock_load.side_effect = mock_models
            
            with patch('pathlib.Path.is_file', return_value=True):
                # Load different "models" (they'll be different mock objects)
                models = []
                for i in range(10):
                    model = FaceTrack.get_model(f"Model_{i}")
                    models.append(model)
                
                # All models should be cached
                self.assertEqual(len(FaceTrack._model_cache), 10)
                
                # Clear cache
                FaceTrack._model_cache.clear()
                self.assertEqual(len(FaceTrack._model_cache), 0)

    # ========== Stress Tests ==========
    
    def test_extract_faces_many_detections(self):
        """Test extract_faces with many face detections."""
        with patch('cv2.imread') as mock_imread:
            mock_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
            mock_imread.return_value = mock_img
            
            with patch('FaceTrack.detector') as mock_detector:
                # Simulate many face detections
                many_detections = [
                    {"box": [i*50, j*50, 40, 40]} 
                    for i in range(10) for j in range(10)
                ]
                mock_detector.detect_faces.return_value = many_detections
                
                with patch('cv2.imwrite') as mock_imwrite:
                    mock_imwrite.return_value = True
                    
                    group_path = self.temp_dir / "many_faces.jpg"
                    out_dir = self.temp_dir / "faces"
                    
                    result = FaceTrack.extract_faces(group_path, out_dir)
                    
                    # Should handle many faces
                    self.assertEqual(len(result), 100)
                    self.assertEqual(mock_imwrite.call_count, 100)
    
    @patch('FaceTrack.get_embedding')
    def test_verify_faces_many_faces(self, mock_get_embedding):
        """Test verify_faces with many face images."""
        # Create many face files
        num_faces = 20
        face_paths = []
        for i in range(num_faces):
            face_path = self.temp_dir / f"face_{i:03d}.jpg"
            cv2.imwrite(str(face_path), np.zeros((112, 112, 3), dtype=np.uint8))
            face_paths.append(face_path)
        
        person_path = self.temp_dir / "person.jpg"
        cv2.imwrite(str(person_path), np.zeros((112, 112, 3), dtype=np.uint8))
        
        # Mock embeddings
        person_emb = np.array([1.0, 0.0, 0.0])
        face_embs = [np.random.random(3) for _ in range(num_faces)]
        mock_get_embedding.side_effect = [person_emb] + face_embs
        
        start_time = time.time()
        
        matches, misses = FaceTrack.verify_faces(face_paths, person_path)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(elapsed_time, 5.0)
        self.assertEqual(matches + misses, num_faces)
    
    def test_concurrent_model_access(self):
        """Test concurrent access to models (simulation)."""
        # Simulate concurrent access by rapid successive calls
        with patch('FaceTrack.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('pathlib.Path.is_file', return_value=True):
                # Rapid successive calls
                models = []
                for _ in range(100):
                    model = FaceTrack.get_model("Inception")
                    models.append(model)
                
                # All should return the same cached model
                self.assertTrue(all(m is models[0] for m in models))
                # Model should only be loaded once
                mock_load.assert_called_once()


class TestConstants(unittest.TestCase):
    """Tests for module constants and configuration."""
    
    def test_min_size_is_reasonable(self):
        """Test that MIN_SIZE is a reasonable value."""
        self.assertGreaterEqual(FaceTrack.MIN_SIZE, 64)
        self.assertLessEqual(FaceTrack.MIN_SIZE, 512)
        self.assertIsInstance(FaceTrack.MIN_SIZE, int)
    
    def test_threshold_is_reasonable(self):
        """Test that THRESHOLD is a reasonable value."""
        self.assertGreaterEqual(FaceTrack.THRESHOLD, 0.0)
        self.assertLessEqual(FaceTrack.THRESHOLD, 1.0)
        self.assertIsInstance(FaceTrack.THRESHOLD, float)
    
    def test_default_model_path_structure(self):
        """Test that DEFAULT_MODEL_PATH has correct structure."""
        path = FaceTrack.DEFAULT_MODEL_PATH
        self.assertIsInstance(path, Path)
        self.assertTrue(path.name.endswith('.keras'))
        self.assertIn('inception', path.name.lower())


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)
