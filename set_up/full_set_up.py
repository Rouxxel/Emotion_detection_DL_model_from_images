import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path (for "python set_up/full_set_up.py" or "python cli.py setup")
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from set_up.setup_kaggle_and_dependencies import main as setup_kaggle_and_dependencies_main
from set_up.download_dataset import main as download_dataset_main
from set_up.augment_minority_classes import main as augment_minority_classes_main

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_default_kaggle_path():
    """Get the default Kaggle credentials path: standard home location first, then project .kaggle/."""
    home_dir = os.path.expanduser("~")
    home_kaggle = os.path.join(home_dir, ".kaggle", "kaggle.json")
    if os.path.isfile(home_kaggle):
        return home_kaggle
    project_kaggle = _project_root / ".kaggle" / "kaggle.json"
    if project_kaggle.is_file():
        return str(project_kaggle)
    return home_kaggle  # so error message points to the standard location

def full_set_up(kaggle_json_path=None, 
                requirements_path="requirements.txt", 
                dataset_name="ananthu017/emotion-detection-fer", 
                dataset_dir="dataset",
                train_subdir="train",
                augmentation_target=7000,
                ):
    try:
        # Use default Kaggle path if none provided
        if kaggle_json_path is None:
            kaggle_json_path = get_default_kaggle_path()
            logging.info(f"Using default Kaggle credentials path: {kaggle_json_path}")
        
        # Verify Kaggle credentials exist
        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(f"Kaggle credentials not found at {kaggle_json_path}")
        
        # If using project-local .kaggle, tell Kaggle CLI to use it for the download step
        kaggle_path_resolved = Path(kaggle_json_path).resolve()
        if kaggle_path_resolved.parent == (_project_root / ".kaggle"):
            os.environ["KAGGLE_CONFIG_DIR"] = str(_project_root / ".kaggle")
        
        # Set up Kaggle and install dependencies
        setup_kaggle_and_dependencies_main(kaggle_json_path, requirements_path)

        # Download the dataset
        download_dataset_main(dataset_name, dataset_dir)

        # Perform augmentation on minority classes in the train dataset folder
        train_dir = os.path.join(dataset_dir, train_subdir)
        logging.info(f"Starting augmentation on training data at {train_dir}...")
        augment_minority_classes_main(train_dir, augmentation_target)
        logging.info("Augmentation completed successfully.")
        
    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Read personal path if it exists, otherwise use default
    personal_path_file = os.path.join(os.path.dirname(__file__), "personal_path.txt")
    kaggle_path = None
    
    if os.path.exists(personal_path_file):
        try:
            with open(personal_path_file, 'r', encoding='utf-8') as f:
                kaggle_path = f.read().strip()
                logging.info(f"Using Kaggle path from personal_path.txt: {kaggle_path}")
        except Exception as e:
            logging.warning(f"Could not read personal_path.txt: {e}")
    
    full_set_up(
        kaggle_json_path=kaggle_path,
        requirements_path="../requirements.txt", 
        dataset_name="ananthu017/emotion-detection-fer", 
        dataset_dir="../dataset",
        train_subdir="train",
        augmentation_target=7000,
    )
