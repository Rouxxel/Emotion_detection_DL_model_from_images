import os
import shutil
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)

def install_requirements(requirements_path="requirements.txt"):
    """Install Python requirements from requirements.txt file."""
    try:
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
            
        logging.info(f"Installing requirements from {requirements_path}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_path
        ], check=True, capture_output=True, text=True)
        logging.info("Requirements installed successfully.")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install requirements: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Error installing requirements: {str(e)}")
        raise

def setup_kaggle_credentials(kaggle_json_path):
    """Set up Kaggle API credentials."""
    try:
        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(f"Kaggle credentials file not found: {kaggle_json_path}")
            
        kaggle_dir = os.path.join(os.path.abspath(os.getcwd()), ".kaggle")
        dest_path = os.path.join(kaggle_dir, "kaggle.json")

        # Create the .kaggle directory if it doesn't exist
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
            logging.info(f"Created directory: {kaggle_dir}")

        # Copy kaggle.json to project .kaggle directory
        shutil.copy(kaggle_json_path, dest_path)
        logging.info(f"Copied {kaggle_json_path} to {dest_path}")

        # Set file permission (Unix-like systems)
        try:
            os.chmod(dest_path, 0o600)
            logging.info(f"Set permissions on {dest_path}")
        except OSError as e:
            logging.warning(f"Could not set file permissions: {e}")
            
    except Exception as e:
        logging.error(f"Error setting up Kaggle credentials: {str(e)}")
        raise

def main(kaggle_json_path, requirements_path="requirements.txt"):
    """Main function to set up Kaggle credentials and install dependencies."""
    try:
        # 1. Install requirements
        install_requirements(requirements_path)

        # 2. Set up Kaggle credentials
        setup_kaggle_credentials(kaggle_json_path)
        
        logging.info("Setup completed successfully.")
        
    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    # If this script is executed directly, call the main function
    main()
