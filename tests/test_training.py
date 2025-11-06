"""
Unit tests for training modules.
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

from dl_scripts.train_transfer_learning import TransferLearningTrainer
from dl_scripts.train_no_transfer_learning import CustomCNNTrainer


class TestTransferLearningTrainer(unittest.TestCase):
    """Test cases for TransferLearningTrainer."""
    
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
            "datasets": {
                "train_directory": os.path.join(self.temp_dir, "train"),
                "test_directory": os.path.join(self.temp_dir, "test")
            },
            "dl_model": {
                "transfer_learning": {
                    "name": "test_transfer_model.h5",
                    "epoch": 2,
                    "fine_tuning_epochs": 1,
                    "early_stop_crit": 1
                },
                "batch_size": 32,
                "fixed_seed": 42,
                "learn_r": 0.001,
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
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = TransferLearningTrainer(self.config_file)
        
        self.assertIsNotNone(trainer.config)
        self.assertEqual(trainer.class_names, ["Happy", "Sad", "Angry"])
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.history)
    
    def test_build_model(self):
        """Test model building."""
        trainer = TransferLearningTrainer(self.config_file)
        
        # Mock TensorFlow to avoid actual model creation
        with patch('tensorflow.keras.applications.DenseNet121') as mock_densenet, \
             patch('tensorflow.keras.models.Model') as mock_model:
            
            mock_base_model = MagicMock()
            mock_base_model.output = MagicMock()
            mock_densenet.return_value = mock_base_model
            
            mock_model_instance = MagicMock()
            mock_model_instance.count_params.return_value = 1000000
            mock_model.return_value = mock_model_instance
            
            model = trainer.build_model()
            
            self.assertIsNotNone(model)
            mock_densenet.assert_called_once()
    
    def test_set_seeds(self):
        """Test random seed setting."""
        trainer = TransferLearningTrainer(self.config_file)
        
        # This should not raise an exception
        trainer._set_seeds()
    
    def test_save_model_no_model(self):
        """Test save_model when no model exists."""
        trainer = TransferLearningTrainer(self.config_file)
        
        with self.assertRaises(ValueError):
            trainer.save_model()


class TestCustomCNNTrainer(unittest.TestCase):
    """Test cases for CustomCNNTrainer."""
    
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
            "datasets": {
                "train_directory": os.path.join(self.temp_dir, "train"),
                "test_directory": os.path.join(self.temp_dir, "test")
            },
            "dl_model": {
                "no_transfer_learning": {
                    "name": "test_cnn_model.h5",
                    "epoch": 2
                },
                "batch_size": 32,
                "fixed_seed": 42,
                "learn_r": 0.001,
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
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = CustomCNNTrainer(self.config_file)
        
        self.assertIsNotNone(trainer.config)
        self.assertEqual(trainer.class_names, ["Happy", "Sad", "Angry"])
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.history)
    
    def test_build_model(self):
        """Test model building."""
        trainer = CustomCNNTrainer(self.config_file)
        
        # Mock TensorFlow to avoid actual model creation
        with patch('tensorflow.keras.models.Sequential') as mock_sequential:
            mock_model = MagicMock()
            mock_model.count_params.return_value = 500000
            mock_sequential.return_value = mock_model
            
            model = trainer.build_model()
            
            self.assertIsNotNone(model)
            mock_sequential.assert_called_once()
    
    def test_save_model_no_model(self):
        """Test save_model when no model exists."""
        trainer = CustomCNNTrainer(self.config_file)
        
        with self.assertRaises(ValueError):
            trainer.save_model()


if __name__ == '__main__':
    unittest.main()