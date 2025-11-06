# Emotion_detection_DL_model_from_images

Deep learning project in which 2 Deep learning models can be built, trained and saved for detecting emotions from `.png` images of human faces in gray-scale. It also includes a user interface.

Small website form one of our members: https://awaken-ai.com/impact-of-transfer-learning-when-identifying-human-emotions/

## 📌 Features

- Emotion classification into 7 categories:
  - Anger 😠
  - Disgust 🤢
  - Fear 😱
  - Happy 😊
  - Neutral 😐
  - Sadness 😔
  - Surprise 😲
- Transfer learning in a custom CNN model
- No Transfer learning in a custom CNN model
- Real-time webcam emotion prediction (user interface)
- Automatic requirements.txt and dataset download from Kaggle (Must have credentials first)
- Configurable via `config_file.json`
- Logging to file and console
- Modular and reusable code structure

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

**View current configuration:**
```bash
python cli.py config
```

**Train the models:**
```bash
python cli.py train --model both          # Train both models
python cli.py train --model transfer      # Train only transfer learning model
python cli.py train --model no-transfer   # Train only non-transfer learning model
```

**Launch user interface:**
```bash
python cli.py ui
```

### Using Jupyter Notebooks (Original Method)

**Train the models:**
```bash
python dl_scripts/dl_model_no_transfer_learning.ipynb
```

and/or

```bash
python dl_scripts/dl_model_transfer_learning.ipynb
```

This will start building, training and saving the models in the root in a directory named 'trained_dl_models'
You can configure parameters like epochs, model name, dataset path, etc., inside `configuration/config_file.json`.

**User interface:**
Note: For the UI to work, you must first train the models by running one of the training scripts. The trained models will be saved in the trained_dl_models/ directory.

```bash
python user_interface/emotion_detection_UI.ipynb
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

## 📝 Logging

Logs are written to both console and a file (`log_history.log`) in the root directory.

---
