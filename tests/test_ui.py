"""
Unit tests for user interface module.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from user_interface.emotion_detection_app import EmotionDetectionApp


class TestEmotionDetectionApp(unittest.TestCase):
    """Test cases for EmotionDetectionApp."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal test configuration
        self.test_config = {
            "images": {"height": 48, "width": 48},
            "classes": {
                "labels": ["Happy", "Sad", "Angry"],
                "emojis": ["😊", "😔", "😠"]
            },
            "dl_model": {
                "transfer_learning": {"name": "test_transfer_model.h5"},
                "no_transfer_learning": {"name": "test_cnn_model.h5"},
                "directory": self.temp_dir
            }
        }
        
        # Create test config file
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('user_interface.emotion_detection_app.load_config')
    @patch('tensorflow.keras.models.load_model')
    @patch('cv2.CascadeClassifier')
    def test_app_initialization_no_models(self, mock_cascade, mock_load_model, mock_config):
        """Test app initialization when no models exist."""
        mock_config.return_value = self.test_config
        mock_load_model.side_effect = FileNotFoundError("Model not found")
        mock_cascade.return_value.empty.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            EmotionDetectionApp("both")
    
    @patch('user_interface.emotion_detection_app.load_config')
    @patch('tensorflow.keras.models.load_model')
    @patch('cv2.CascadeClassifier')
    @patch('os.path.exists')
    def test_app_initialization_success(self, mock_exists, mock_cascade, mock_load_model, mock_config):
        """Test successful app initialization."""
        mock_config.return_value = self.test_config
        mock_exists.return_value = True
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_cascade_instance = MagicMock()
        mock_cascade_instance.empty.return_value = False
        mock_cascade.return_value = mock_cascade_instance
        
        app = EmotionDetectionApp("transfer")
        
        self.assertIsNotNone(app.config)
        self.assertEqual(app.model_type, "transfer")
        self.assertIn("transfer", app.models)
    
    def test_preprocess_image_transfer(self):
        """Test image preprocessing for transfer learning."""
        with patch('user_interface.emotion_detection_app.load_config') as mock_config, \
             patch('tensorflow.keras.models.load_model'), \
             patch('cv2.CascadeClassifier'), \
             patch('os.path.exists'):
            
            mock_config.return_value = self.test_config
            
            app = EmotionDetectionApp("transfer")
            
            # Create a dummy image
            dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # This should not raise an exception
            processed = app.preprocess_image_transfer(dummy_img)
            
            # Check output shape
            self.assertEqual(processed.shape, (1, 48, 48, 3))
    
    def test_preprocess_image_no_transfer(self):
        """Test image preprocessing for custom CNN."""
        with patch('user_interface.emotion_detection_app.load_config') as mock_config, \
             patch('tensorflow.keras.models.load_model'), \
             patch('cv2.CascadeClassifier'), \
             patch('os.path.exists'):
            
            mock_config.return_value = self.test_config
            
            app = EmotionDetectionApp("no_transfer")
            
            # Create a dummy image
            dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # This should not raise an exception
            processed = app.preprocess_image_no_transfer(dummy_img)
            
            # Check output shape
            self.assertEqual(processed.shape, (1, 48, 48, 1))
    
    @patch('user_interface.emotion_detection_app.load_config')
    @patch('tensorflow.keras.models.load_model')
    @patch('cv2.CascadeClassifier')
    @patch('os.path.exists')
    def test_predict_emotion(self, mock_exists, mock_cascade, mock_load_model, mock_config):
        """Test emotion prediction."""
        mock_config.return_value = self.test_config
        mock_exists.return_value = True
        
        # Mock model prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # High confidence for "Sad"
        mock_load_model.return_value = mock_model
        
        mock_cascade_instance = MagicMock()
        mock_cascade_instance.empty.return_value = False
        mock_cascade.return_value = mock_cascade_instance
        
        app = EmotionDetectionApp("transfer")
        
        # Create a dummy image
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        emotion, emoji, confidence = app.predict_emotion("transfer", dummy_img)
        
        self.assertEqual(emotion, "Sad")
        self.assertEqual(emoji, "😔")
        self.assertAlmostEqual(confidence, 0.8, places=1)
    
    def test_predict_emotion_invalid_model(self):
        """Test prediction with invalid model name."""
        with patch('user_interface.emotion_detection_app.load_config') as mock_config, \
             patch('tensorflow.keras.models.load_model'), \
             patch('cv2.CascadeClassifier'), \
             patch('os.path.exists'):
            
            mock_config.return_value = self.test_config
            
            app = EmotionDetectionApp("transfer")
            
            dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            emotion, emoji, confidence = app.predict_emotion("invalid_model", dummy_img)
            
            self.assertEqual(emotion, "Unknown")
            self.assertEqual(emoji, "❓")
            self.assertEqual(confidence, 0.0)


if __name__ == '__main__':
    unittest.main()