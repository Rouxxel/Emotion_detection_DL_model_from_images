import os
import shutil
import subprocess
import sys
import logging

def install_requirements(requirements_path="requirements.txt"):
    logging.info(f"Installing requirements from {requirements_path}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    logging.info("Requirements installed.")

def setup_kaggle_credentials(kaggle_json_path):
    kaggle_dir = os.path.join(os.path.abspath(os.getcwd()), ".kaggle")
    dest_path = os.path.join(kaggle_dir, "kaggle.json")

    # Create the .kaggle directory if it doesn't exist
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        logging.info(f"Created directory: {kaggle_dir}")

    # Copy kaggle.json to ~/.kaggle
    shutil.copy(kaggle_json_path, dest_path)
    logging.info(f"Copied {kaggle_json_path} to {dest_path}")

    # Set file permission
    os.chmod(dest_path, 0o600)
    logging.info(f"Set permissions on {dest_path}")

def main(kaggle_json_path="C:/Users/Acer/.kaggle/kaggle.json", requirements_path="requirements.txt"):
    # 1. Install requirements
    install_requirements(requirements_path)

    # 2. Set up Kaggle credentials
    setup_kaggle_credentials(kaggle_json_path)

if __name__ == "__main__":
    # If this script is executed directly, call the main function
    main()
