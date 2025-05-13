import logging
from setup_kaggle_and_dependencies import main as setup_kaggle_and_dependencies_main
from download_dataset import main as download_dataset_main

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

def full_set_up(kaggle_json_path="C:/Users/YourName/.kaggle/kaggle.json", requirements_path="requirements.txt", dataset_name="ananthu017/emotion-detection-fer", dataset_dir="dataset"):
    # 1. Set up Kaggle and install dependencies
    setup_kaggle_and_dependencies_main(kaggle_json_path, requirements_path)

    # 2. Download the dataset
    download_dataset_main(dataset_name, dataset_dir)

if __name__ == "__main__":
    # Call the full setup function
    full_set_up(kaggle_json_path="C:/Users/YourName/.kaggle/kaggle.json", #TODO: REPLACE WITH ACTUAL PATH
                requirements_path="requirements.txt", 
                dataset_name="ananthu017/emotion-detection-fer", 
                dataset_dir="dataset")
