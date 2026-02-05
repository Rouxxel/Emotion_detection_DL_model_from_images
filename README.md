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
- **Model Optimization**: TensorFlow Lite, TensorRT, and ONNX optimization
- **Docker Containerization**: Production-ready containerization with multi-stage builds
- **Resource Management**: Intelligent resource allocation and cleanup

---

## 📁 Directory Structure

```
Emotion_detection_DL_model_from_images/
├── configuration/          # Configuration management (JSON config, loaders)
├── data_pipeline/         # Data validation and preprocessing
├── dl_scripts/             # Training scripts and notebooks (transfer learning + custom CNN)
├── docs/                   # All documentation: Sphinx API/docs source + project report/slides
│   ├── conf.py, index.rst  # Sphinx build (make docs → _build/html)
│   └── project/            # Project deliverables (report PDF, LaTeX, slides) — see docs/PROJECT_ORGANIZATION.md
├── documentation/          # Legacy: project report/slides (to be merged into docs/project/ — see docs/PROJECT_ORGANIZATION.md)
├── experiment_tracking/    # Experiment tracking and versioning
├── performance/            # Model optimization (TFLite, TensorRT, ONNX)
├── set_up/                 # Setup: Kaggle, dependencies, dataset download
├── tests/                  # Unit and integration tests
├── user_interface/         # Single UI package: app, notebook, Haar cascade for webcam/image detection
├── cli.py                  # Command-line interface
├── requirements.txt
└── README.md
```

**Documentation:** There are two folders today: **`docs/`** (Sphinx source for generated API/user docs) and **`documentation/`** (project report PDF, LaTeX, and slides). For a single, clearer layout, see [docs/PROJECT_ORGANIZATION.md](docs/PROJECT_ORGANIZATION.md).

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

## 🎯 Step-by-step: get a trained model and run the camera UI

Do all steps **from the project root** (the folder that contains `cli.py` and `README.md`).

### 1. Environment and credentials

- **Python**: 3.8+ with `pip`.
- **Kaggle**: Create an account at [kaggle.com](https://www.kaggle.com), then get your API key (Account → Create New API Token). The setup script looks for `kaggle.json` in this order:
  1. **Standard location** (checked first):
     - **Windows**: `C:\Users\<YourName>\.kaggle\kaggle.json`
     - **macOS/Linux**: `~/.kaggle/kaggle.json`
  2. **Project root** (fallback): `<project_folder>/.kaggle/kaggle.json`

  As an option, you can put your Kaggle credentials in the project root by creating a `.kaggle` folder and placing `kaggle.json` inside it. This folder is in `.gitignore`, so your credentials are not committed. Use this if you prefer not to use the standard user folder.

### 2. Clone and install

```bash
git clone https://github.com/your-username/Emotion_detection_DL_model_from_images.git
cd Emotion_detection_DL_model_from_images
pip install -r requirements.txt
```

### 3. Setup (dataset and dependencies)

This installs Python dependencies, downloads the FER emotion dataset from Kaggle, and augments the training data.

```bash
python cli.py setup
```

- Dataset is stored under **`dataset/`** (e.g. `dataset/train/`, `dataset/test/`).
- If your `kaggle.json` is elsewhere:  
  `python cli.py setup --kaggle-path "C:\path\to\kaggle.json"`

### 4. Train the model(s)

Train **one** or **both** of:

- **Transfer learning**: DenseNet121 fine-tuned for emotions.
- **Custom CNN**: small CNN built from scratch.

Trained models, training logs, and accuracy/loss plots are saved under **`trained_dl_models/`** in subfolders:
- **`trained_dl_models/tf_learning/`** — transfer learning model (`.h5`), `log_history_transfer_learning.log`, `transfer_learning_history.png`
- **`trained_dl_models/no_tf_learning/`** — custom CNN model (`.h5`), `log_history_no_transfer_learning.log`, `no_transfer_learning_history.png`

**Train both (recommended for the UI):**
```bash
python cli.py train --model both
```

**Train only one:**
```bash
python cli.py train --model transfer      # → trained_dl_models/tf_learning/*.h5
python cli.py train --model no-transfer  # → trained_dl_models/no_tf_learning/*.h5
```

Training can take a long time (epochs and hardware-dependent). When it finishes, each model’s folder will contain the `.h5` file, a `.log` file, and an accuracy/loss `.png` plot.

### 5. Run the camera UI (webcam window)

The **user interface** that opens a window and uses your webcam is in **`user_interface/`**. It loads the trained `.h5` model(s) from `trained_dl_models/` and runs real-time emotion detection on the camera feed.

**Launch with default (both models, webcam):**
```bash
python cli.py ui
```

Or call the app directly:

```bash
python user_interface/emotion_detection_app.py --model both --mode webcam
```

**Options:**

- `--model transfer` or `--model no-transfer`: use only one of the two models.
- `--mode image --input photo.png --output result.png`: run on a single image instead of webcam.

In webcam mode, use **`s`** to switch between models (if both are loaded) and **`q`** to quit.

**Webcam error “The function is not implemented” / no window:** Your OpenCV build has no GUI support (often because `opencv-python-headless` is installed). Fix: `pip uninstall opencv-python-headless` then `pip install opencv-python`. Use **image mode** instead if you only need file-based detection: `python cli.py ui --mode image --input photo.png --output result.png`.

**Summary:** Setup → Train → UI. All from project root; the config points to `dataset/` and `trained_dl_models/` so the UI finds the models you just trained.

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
make optimize                # Optimize trained models
make benchmark               # Run performance benchmarks
make docker-build            # Build Docker images
make docker-dev              # Run development container
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

# Performance optimization
python cli.py optimize trained_dl_models/model.h5 --optimization-type tflite
python cli.py benchmark --duration 60
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

- **Training logs**: inside `trained_dl_models/tf_learning/` and `trained_dl_models/no_tf_learning/` (e.g. `log_history_transfer_learning.log`, `log_history_no_transfer_learning.log`)
- **General logs**: `emotion_detection.log`
- **Console output**: Real-time logging to terminal
- **Configurable levels**: INFO, DEBUG, WARNING, ERROR

## 🐳 Docker Usage

### Quick Start with Docker:
```bash
# Setup Docker environment (creates dataset, trained_dl_models, experiments, logs)
chmod +x scripts/docker-setup.sh
./scripts/docker-setup.sh

# Build images
make docker-build

# Run development environment (interactive shell in container)
make docker-dev

# Run production container (UI; requires trained .h5 models in trained_dl_models/)
make docker-prod

# Train models in container (uses dataset/ and writes to trained_dl_models/)
make docker-train

# Serve Sphinx docs at http://localhost:8080
make docker-docs
```

### Docker Services:
- **Development** (`emotion-detection-dev`): Full dev environment, volumes for code/dataset/models.
- **Production** (`emotion-detection-prod`): Runs `cli.py ui --mode webcam`; mount `trained_dl_models/` with your `.h5` files.
- **Training** (`emotion-detection-train`): Runs `cli.py train` with experiment tracking; mount `dataset/` and `trained_dl_models/`.
- **Documentation** (`docs`): Builds Sphinx HTML and serves at port 8080.

## ⚡ Performance Optimization

### Model Optimization:
```bash
# Optimize for mobile/edge deployment
python cli.py optimize model.h5 --optimization-type tflite

# Optimize for NVIDIA GPUs
python cli.py optimize model.h5 --optimization-type tensorrt

# Convert to ONNX format
python cli.py optimize model.h5 --optimization-type onnx
```

### Performance Monitoring:
```bash
# Run performance benchmark
python cli.py benchmark --duration 60

# Run performance tests
chmod +x scripts/performance-test.sh
./scripts/performance-test.sh
```

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

### Project structure

See the [Directory Structure](#-directory-structure) section above for the full layout.

---
