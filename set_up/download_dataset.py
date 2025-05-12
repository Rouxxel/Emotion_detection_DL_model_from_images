import os
import subprocess
import logging
import zipfile

def download_dataset(dataset_name="ananthu017/emotion-detection-fer", dataset_dir="dataset"):
    project_root = os.path.abspath(os.getcwd())  # Get the current working directory (root of the project)
    full_dataset_dir = os.path.join(project_root, dataset_dir)

    # Create the dataset directory if it doesn't exist
    if not os.path.exists(full_dataset_dir):
        os.makedirs(full_dataset_dir)
        logging.info(f"Created directory: {full_dataset_dir}")

    # Path for the downloaded zip file
    zip_path = os.path.join(full_dataset_dir, f"{dataset_name.split('/')[-1]}.zip")

    # Check if the dataset zip is already downloaded
    if not os.path.exists(zip_path):
        logging.info(f"Downloading dataset {dataset_name}...")
        subprocess.run([
            "kaggle", "datasets", "download", dataset_name,
            "--path", full_dataset_dir, "--force"
        ], check=True)
        logging.info(f"Dataset {dataset_name} downloaded to {zip_path}")
    else:
        logging.info(f"Dataset {dataset_name} already exists at {zip_path}")

    # Unzip the dataset if not already unzipped
    if not os.path.exists(os.path.join(full_dataset_dir, "extracted.txt")):
        logging.info(f"Unzipping dataset {dataset_name}...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(full_dataset_dir)

        with open(os.path.join(full_dataset_dir, "extracted.txt"), "w") as f:
            f.write("Dataset extracted")  # Create a flag file to mark that extraction is complete

        logging.info("Dataset unzipped.")
    else:
        logging.info(f"Dataset {dataset_name} already unzipped.")
        
def main(dataset_name="ananthu017/emotion-detection-fer", dataset_dir="dataset"):
    download_dataset(dataset_name, dataset_dir)

if __name__ == "__main__":
    # If this script is executed directly, call the main function
    main()
