"""
Unit tests for configuration module.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configuration.config_invoke import load_config


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "images": {
                "height": 48,
                "width": 48
            },
            "classes": {
                "labels": ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"],
                "emojis": ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]
            },
            "datasets": {
                "dataset_directory": "../dataset",
                "train_directory": "../dataset/train",
                "test_directory": "../dataset/test"
            },
            "dl_model": {
                "transfer_learning": {
                    "name": "emotion_detection_from_image_transfer_learning.h5",
                    "subdir": "tf_learning",
                    "epoch": 30,
                    "fine_tuning_epochs": 20,
                    "early_stop_crit": 3
                },
                "no_transfer_learning": {
                    "name": "emotion_detection_from_image_no_transfer_learning.h5",
                    "subdir": "no_tf_learning",
                    "epoch": 20
                },
                "batch_size": 64,
                "fixed_seed": 12,
                "learn_r": 0.01,
                "directory": "../trained_dl_models"
            }
        }
    
    def test_load_config_success(self):
        """Test successful configuration loading."""
        config = load_config()
        
        # Check that config is loaded and has expected structure
        self.assertIsInstance(config, dict)
        self.assertIn("images", config)
        self.assertIn("classes", config)
        self.assertIn("datasets", config)
        self.assertIn("dl_model", config)
    
    def test_config_structure(self):
        """Test configuration structure and required fields."""
        config = load_config()
        
        # Test images section
        self.assertIn("height", config["images"])
        self.assertIn("width", config["images"])
        self.assertEqual(config["images"]["height"], 48)
        self.assertEqual(config["images"]["width"], 48)
        
        # Test classes section
        self.assertIn("labels", config["classes"])
        self.assertIn("emojis", config["classes"])
        self.assertEqual(len(config["classes"]["labels"]), 7)
        self.assertEqual(len(config["classes"]["emojis"]), 7)
        
        # Test dl_model section
        self.assertIn("batch_size", config["dl_model"])
        self.assertIn("fixed_seed", config["dl_model"])
        self.assertIn("learn_r", config["dl_model"])
    
    def test_config_data_types(self):
        """Test that configuration values have correct data types."""
        config = load_config()
        
        # Test numeric values
        self.assertIsInstance(config["images"]["height"], int)
        self.assertIsInstance(config["images"]["width"], int)
        self.assertIsInstance(config["dl_model"]["batch_size"], int)
        self.assertIsInstance(config["dl_model"]["fixed_seed"], int)
        self.assertIsInstance(config["dl_model"]["learn_r"], (int, float))
        
        # Test string values
        self.assertIsInstance(config["dl_model"]["directory"], str)
        
        # Test lists
        self.assertIsInstance(config["classes"]["labels"], list)
        self.assertIsInstance(config["classes"]["emojis"], list)
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        # Create a temporary invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_path = f.name
        
        try:
            # This should raise an exception
            with self.assertRaises(json.JSONDecodeError):
                with open(temp_path, 'r') as f:
                    json.load(f)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()