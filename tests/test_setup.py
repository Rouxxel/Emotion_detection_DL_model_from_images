"""
Unit tests for setup modules.
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from set_up.setup_kaggle_and_dependencies import install_requirements, setup_kaggle_credentials
from set_up.download_dataset import download_dataset
from set_up.full_set_up import get_default_kaggle_path


class TestSetupFunctions(unittest.TestCase):
    """Test cases for setup functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_requirements = "numpy==1.21.0\npandas==1.3.0\n"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_default_kaggle_path(self):
        """Test default Kaggle path generation."""
        path = get_default_kaggle_path()
        
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.kaggle/kaggle.json') or path.endswith('.kaggle\\kaggle.json'))
    
    def test_install_requirements_file_not_found(self):
        """Test install_requirements with non-existent file."""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.txt")
        
        with self.assertRaises(FileNotFoundError):
            install_requirements(non_existent_file)
    
    def test_setup_kaggle_credentials_file_not_found(self):
        """Test setup_kaggle_credentials with non-existent file."""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.json")
        
        with self.assertRaises(FileNotFoundError):
            setup_kaggle_credentials(non_existent_file)
    
    @patch('subprocess.run')
    def test_install_requirements_success(self, mock_subprocess):
        """Test successful requirements installation."""
        # Create a temporary requirements file
        req_file = os.path.join(self.temp_dir, "requirements.txt")
        with open(req_file, 'w') as f:
            f.write(self.test_requirements)
        
        # Mock successful subprocess call
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        # This should not raise an exception
        try:
            install_requirements(req_file)
        except Exception as e:
            self.fail(f"install_requirements raised an exception: {e}")
    
    def test_setup_kaggle_credentials_success(self):
        """Test successful Kaggle credentials setup."""
        # Create a temporary kaggle.json file
        kaggle_file = os.path.join(self.temp_dir, "kaggle.json")
        with open(kaggle_file, 'w') as f:
            f.write('{"username": "test", "key": "test_key"}')
        
        # Change to temp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            setup_kaggle_credentials(kaggle_file)
            
            # Check that .kaggle directory was created
            kaggle_dir = os.path.join(self.temp_dir, ".kaggle")
            self.assertTrue(os.path.exists(kaggle_dir))
            
            # Check that kaggle.json was copied
            dest_file = os.path.join(kaggle_dir, "kaggle.json")
            self.assertTrue(os.path.exists(dest_file))
            
        finally:
            os.chdir(original_cwd)


class TestDownloadDataset(unittest.TestCase):
    """Test cases for dataset download functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_download_dataset_already_exists(self, mock_exists, mock_subprocess):
        """Test download when dataset already exists."""
        # Mock that zip file already exists
        mock_exists.side_effect = lambda path: path.endswith('.zip')
        
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            # This should not call subprocess since file "exists"
            download_dataset("test/dataset", "test_dataset")
            
            # Subprocess should not be called for download
            mock_subprocess.assert_not_called()
            
        finally:
            os.chdir(original_cwd)
    
    def test_download_dataset_directory_creation(self):
        """Test that dataset directory is created."""
        dataset_dir = os.path.join(self.temp_dir, "test_dataset")
        
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            # Mock the subprocess and file existence checks
            with patch('subprocess.run') as mock_subprocess, \
                 patch('os.path.exists') as mock_exists, \
                 patch('zipfile.ZipFile'):
                
                # Mock that files don't exist initially
                mock_exists.return_value = False
                mock_subprocess.return_value = MagicMock(returncode=0)
                
                try:
                    download_dataset("test/dataset", "test_dataset")
                except Exception:
                    pass  # We expect this to fail due to mocking, but directory should be created
                
                # Check that directory was created
                self.assertTrue(os.path.exists(dataset_dir))
                
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()