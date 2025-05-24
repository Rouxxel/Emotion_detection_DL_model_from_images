import logging
from setup_kaggle_and_dependencies import main as setup_kaggle_and_dependencies_main
from download_dataset import main as download_dataset_main
from augment_minority_classes import main as augment_minority_classes_main

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

def full_set_up(kaggle_json_path="C:/Users/YourName/.kaggle/kaggle.json", 
                requirements_path="requirements.txt", 
                dataset_name="ananthu017/emotion-detection-fer", 
                dataset_dir="dataset",
                train_subdir="train",
                augmentation_target=7000,
                ):
    # Set up Kaggle and install dependencies
    setup_kaggle_and_dependencies_main(kaggle_json_path, requirements_path)

    # Download the dataset
    download_dataset_main(dataset_name, dataset_dir)

    # Perform augmentation on minority classes in the train dataset folder
    train_dir = f"{dataset_dir}/{train_subdir}"
    logging.info(f"Starting augmentation on training data at {train_dir}...")
    augment_minority_classes_main(train_dir, augmentation_target)
    logging.info("Augmentation completed.")

if __name__ == "__main__":
    full_set_up(
        kaggle_json_path="C:/Users/Username/.kaggle/kaggle.json",  #TODO: Replace with your actual path
        requirements_path="requirements.txt", 
        dataset_name="ananthu017/emotion-detection-fer", 
        dataset_dir="dataset",
        train_subdir="train",
        augmentation_target=7000,
    )
