# Emotion_detection_DL_model_from_images

Deep learning project in which 2 Deep learning models can be built, trained and saved for detecting emotions from `.png` images of human faces in gray-scale. It also includes a user interface.

Small website form one of our members: https://awaken-ai.com/impact-of-transfer-learning-when-identifying-human-emotions/

## 📌 Features

- **Emotion Classification**: 7 emotion categories (Anger 😠, Disgust 🤢, Fear 😱, Happy 😊, Neutral 😐, Sadness 😔, Surprise 😲)
- **Dual Model Architecture**: 
  - Transfer learning model using DenseNet121
  - Custom CNN model built from scratch
- **Real-time Detection**: Webcam-based emotion detection with live preview
- **Modular Design**: Clean, testable Python modules (converted from Jupyter notebooks)
- **Comprehensive Testing**: Unit tests with pytest and coverage reporting
- **Data Pipeline**: Automated dataset validation and preprocessing
- **CLI Interface**: Professional command-line interface for all operations
- **Environment Configuration**: Support for environment variable overrides
- **Experiment Tracking**: MLOps-ready experiment tracking and model versioning
- **Advanced Configuration**: Multi-source configuration with validation
- **Data Pipeline**: Automated data validation, preprocessing, and quality checks
- **CI/CD Ready**: GitHub Actions workflow with security scanning and documentation
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Comprehensive Documentation**: Sphinx-generated documentation with examples
- **Performance Monitoring**: Memory profiling and performance benchmarks

---

## 📁 Directory Structure

```
Emotion_detection_DL_model_from_images/
|
├── configuration/
│   ├── __init__.py
│   ├── config_file.json         # JSON configuration file
│   └── config_invoke.py         # Loads config JSON
│
├── dl_scripts/
│   ├── __init__.py
│   ├── dl_model_no_transfer_learning.ipynb # Script to build, train and save model
│   ├── dl_model_transfer_learning.ipynb # Script to build, train and save model
│
├── documentation/                      # all documentation files
│
├── set_up/
│   ├── setup_kaggle_and_dependencies.py 
│   ├── download_dataset.py
│   └── full_set_up.py                   #Run the other 2 scripts in the directory
│
├── user_interface/
│   ├── __init__.py 
│   ├── emotion_detection_UI.ipynb      #Notebook for user interface
│   └── haarcascade_frontalface_default.xml #pre-trained classifier used by OpenCV                   
│
├── .gitignore
├── Dataset_distribution.png
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Emotion_detection_DL_model_from_images.git
cd Emotion_detection_DL_model_from_images
```

### 2. Add your Kaggle API credentials
*You must be registered to kaggle and have your credentials somewhere locally*

Place your `kaggle.json` file in the appropriate location (e.g., `C:/Users/YourName/.kaggle/kaggle.json`).

### 3. Run the setup script

This installs dependencies and downloads the dataset:

**Using the new CLI (recommended):**
```bash
python cli.py setup
```

**Or using the original script:**
```bash
python set_up/full_set_up.py
```

---

## 🧪 Usage

### Using the CLI (Recommended)

**Complete setup:**
```bash
python cli.py setup                       # Full setup with auto-detected Kaggle credentials
python cli.py setup --kaggle-path /path/to/kaggle.json  # Custom Kaggle path
```

**View current configuration:**
```bash
python cli.py config
```

**Validate dataset:**
```bash
python data_pipeline/data_validator.py dataset --output validation_report.json
```

**Train the models:**
```bash
python cli.py train --model both          # Train both models
python cli.py train --model transfer      # Train only transfer learning model
python cli.py train --model no-transfer   # Train only custom CNN model

# With experiment tracking
python cli.py train --track-experiment --experiment-name "my_experiment"
```

**Manage experiments:**
```bash
python cli.py experiments list            # List all experiments
python cli.py experiments show --experiment-id exp_0001  # Show experiment details
python cli.py experiments compare --experiment-ids exp_0001 exp_0002  # Compare experiments
```

**Launch user interface:**
```bash
python cli.py ui                          # Launch with both models (webcam)
python cli.py ui --model-type transfer --mode webcam    # Specific model
python cli.py ui --mode image --input photo.jpg --output result.jpg  # Process single image
```

### Using Make Commands

```bash
make help                    # Show all available commands
make setup                   # Run full setup
make test                    # Run all tests
make train                   # Train both models
make train-tracked           # Train with experiment tracking
make ui                      # Launch user interface
make validate-data           # Validate dataset
make preprocess-data         # Run data preprocessing
make docs                    # Build documentation
make experiments             # List experiments
make clean                   # Clean up generated files
```

### Using Python Modules Directly

**Train transfer learning model:**
```bash
python dl_scripts/train_transfer_learning.py
```

**Train custom CNN model:**
```bash
python dl_scripts/train_no_transfer_learning.py
```

**Launch emotion detection app:**
```bash
python user_interface/emotion_detection_app.py --model both --mode webcam
```

### Environment Variables

You can override configuration using environment variables:
```bash
export EMOTION_DETECTION_DL_MODEL__BATCH_SIZE=32
export EMOTION_DETECTION_DL_MODEL__LEARN_R=0.001
python cli.py train
```

---

## 🧠 Emotion Classes

The model is trained to detect the following emotions:

| Emotion   | Emoji |
|-----------|--------|
| Anger     | 👿     |
| Disgust   | 🤢     |
| Fear      | 😱     |
| Happy     | 😊     |
| Neutral   | 😐     |
| Sadness   | 😔     |
| Surprise  | 😲     |

---

## 🛠 Dependencies

Listed in `requirements.txt`. Example versions:

```
kaggle==1.7.4.5
tensorflow==2.17.0
matplotlib==3.10.3
pillow==10.4.0
pandas==2.2.3
numpy==1.26.4
seaborn==0.13.2
plotly==6.0.1
scikit-learn==1.6.1
opencv-python==4.10.0.84
``` 

Install manually if setup fails:

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

Or use the Makefile:
```bash
make install      # Production dependencies
make install-dev  # Development dependencies
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
make test

# Run with coverage report
make test-coverage

# Run only unit tests
pytest tests/ -m "not integration"

# Run specific test file
pytest tests/test_config.py -v
```

## 📝 Logging

- **Training logs**: `log_history_transfer_learning.log`, `log_history_no_transfer_learning.log`
- **General logs**: `emotion_detection.log`
- **Console output**: Real-time logging to terminal
- **Configurable levels**: INFO, DEBUG, WARNING, ERROR

## 🔧 Development

### Setting up development environment:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
make format

# Run linting
make lint
```

### Project Structure:
```
├── configuration/          # Configuration management
├── data_pipeline/         # Data validation and preprocessing
├── dl_scripts/           # Training scripts (converted from notebooks)
├── set_up/              # Setup and installation scripts
├── tests/               # Unit and integration tests
├── user_interface/      # Real-time detection application
├── cli.py              # Command-line interface
└── Makefile           # Development commands
```

---
