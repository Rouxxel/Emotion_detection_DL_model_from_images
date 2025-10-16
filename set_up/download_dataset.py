import os
import subprocess
import logging
import zipfile

logging.basicConfig(level=logging.INFO)

def download_dataset(dataset_name="ananthu017/emotion-detection-fer", dataset_dir="dataset"):
    """Download and extract dataset from Kaggle."""
    try:
        project_root = os.path.abspath(os.getcwd())
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
            try:
                result = subprocess.run([
                    "kaggle", "datasets", "download", dataset_name,
                    "--path", full_dataset_dir, "--force"
                ], check=True, capture_output=True, text=True)
                logging.info(f"Dataset {dataset_name} downloaded to {zip_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to download dataset: {e.stderr}")
                raise
            except FileNotFoundError:
                raise RuntimeError("Kaggle CLI not found. Please ensure Kaggle is installed and configured.")
        else:
            logging.info(f"Dataset {dataset_name} already exists at {zip_path}")

        # Unzip the dataset if not already unzipped
        extracted_flag = os.path.join(full_dataset_dir, "extracted.txt")
        if not os.path.exists(extracted_flag):
            logging.info(f"Unzipping dataset {dataset_name}...")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(full_dataset_dir)

                with open(extracted_flag, "w", encoding='utf-8') as f:
                    f.write("Dataset extracted")

                logging.info("Dataset unzipped successfully.")
            except zipfile.BadZipFile:
                logging.error(f"Downloaded file {zip_path} is not a valid zip file")
                raise
            except Exception as e:
                logging.error(f"Failed to extract dataset: {str(e)}")
                raise
        else:
            logging.info(f"Dataset {dataset_name} already unzipped.")
            
    except Exception as e:
        logging.error(f"Error in download_dataset: {str(e)}")
        raise
        
def main(dataset_name="ananthu017/emotion-detection-fer", dataset_dir="dataset"):
    download_dataset(dataset_name, dataset_dir)

if __name__ == "__main__":
    # If this script is executed directly, call the main function
    main()
