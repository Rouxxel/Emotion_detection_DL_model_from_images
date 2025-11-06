#!/usr/bin/env python3
"""
Emotion Detection User Interface

Real-time emotion detection application using webcam feed.
Supports both transfer learning and custom CNN models.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import argparse

# Computer Vision and UI
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configuration.config_invoke import load_config


class EmotionDetectionApp:
    """Real-time emotion detection application."""
    
    def __init__(self, model_type: str = "both"):
        """
        Initialize the emotion detection app.
        
        Args:
            model_type: Type of model to use ("transfer", "no_transfer", or "both")
        """
        self.config = load_config()
        self.model_type = model_type
        self.models = {}
        
        # Configuration
        self.img_height = self.config["images"]["height"]
        self.img_width = self.config["images"]["width"]
        self.class_labels = self.config["classes"]["labels"]
        self.class_emojis = self.config["classes"]["emojis"]
        
        # Setup logging
        self._setup_logging()
        
        # Load models
        self._load_models()
        
        # Load face cascade
        self._load_face_cascade()
        
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def _load_models(self) -> None:
        """Load the trained emotion detection models."""
        try:
            model_dir = self.config["dl_model"]["directory"]
            
            if self.model_type in ["transfer", "both"]:
                transfer_model_path = os.path.join(
                    model_dir, 
                    self.config["dl_model"]["transfer_learning"]["name"]
                )
                if os.path.exists(transfer_model_path):
                    self.models["transfer"] = tf.keras.models.load_model(transfer_model_path)
                    logging.info(f"Transfer learning model loaded from {transfer_model_path}")
                else:
                    logging.warning(f"Transfer learning model not found at {transfer_model_path}")
            
            if self.model_type in ["no_transfer", "both"]:
                no_transfer_model_path = os.path.join(
                    model_dir,
                    self.config["dl_model"]["no_transfer_learning"]["name"]
                )
                if os.path.exists(no_transfer_model_path):
                    self.models["no_transfer"] = tf.keras.models.load_model(no_transfer_model_path)
                    logging.info(f"Custom CNN model loaded from {no_transfer_model_path}")
                else:
                    logging.warning(f"Custom CNN model not found at {no_transfer_model_path}")
            
            if not self.models:
                raise FileNotFoundError("No trained models found. Please train models first.")
                
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_face_cascade(self) -> None:
        """Load the Haar cascade for face detection."""
        try:
            cascade_path = os.path.join(
                os.path.dirname(__file__), 
                "haarcascade_frontalface_default.xml"
            )
            
            if not os.path.exists(cascade_path):
                # Try to use OpenCV's built-in cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load face cascade classifier")
                
            logging.info("Face cascade classifier loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading face cascade: {str(e)}")
            raise
    
    def preprocess_image_transfer(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for transfer learning model (RGB).
        
        Args:
            img: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Resize image
        img_resized = cv2.resize(img, (self.img_width, self.img_height))
        
        # Convert to RGB (transfer learning models expect RGB)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to array and preprocess
        img_array = img_to_array(img_rgb)
        img_array = preprocess_input(img_array)
        img_array = img_array * (1.0 / 255)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def preprocess_image_no_transfer(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for custom CNN model (grayscale).
        
        Args:
            img: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        img_resized = cv2.resize(img_gray, (self.img_width, self.img_height))
        
        # Normalize and expand dimensions
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
        
        return img_array
    
    def predict_emotion(self, model_name: str, img: np.ndarray) -> Tuple[str, str, float]:
        """
        Predict emotion from image using specified model.
        
        Args:
            model_name: Name of the model to use ("transfer" or "no_transfer")
            img: Input image array
            
        Returns:
            Tuple of (predicted_label, emoji, confidence)
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not loaded")
            
            # Preprocess image based on model type
            if model_name == "transfer":
                preprocessed_img = self.preprocess_image_transfer(img)
            else:
                preprocessed_img = self.preprocess_image_no_transfer(img)
            
            # Make prediction
            prediction = self.models[model_name].predict(preprocessed_img, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            predicted_label = self.class_labels[predicted_class]
            emoji = self.class_emojis[predicted_class]
            
            return predicted_label, emoji, confidence
            
        except Exception as e:
            logging.error(f"Error predicting emotion: {str(e)}")
            return "Unknown", "❓", 0.0
    
    def run_webcam_detection(self) -> None:
        """Run real-time emotion detection using webcam."""
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
            logging.info("Starting webcam emotion detection. Press 'q' to quit, 's' to switch models.")
            
            # Model switching for "both" mode
            current_model = "transfer" if "transfer" in self.models else "no_transfer"
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to capture frame from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    if len(self.models) > 0:
                        emotion, emoji, confidence = self.predict_emotion(current_model, face_roi)
                        
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Display prediction
                        label = f"{emotion} {emoji} ({confidence:.2f})"
                        cv2.putText(frame, label, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display current model info
                model_info = f"Model: {current_model.replace('_', ' ').title()}"
                cv2.putText(frame, model_info, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display instructions
                instructions = "Press 'q' to quit, 's' to switch models"
                cv2.putText(frame, instructions, (10, frame.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Emotion Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and len(self.models) > 1:
                    # Switch between models
                    if current_model == "transfer":
                        current_model = "no_transfer"
                    else:
                        current_model = "transfer"
                    logging.info(f"Switched to {current_model} model")
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Webcam detection stopped")
            
        except Exception as e:
            logging.error(f"Error in webcam detection: {str(e)}")
            raise
    
    def process_image_file(self, image_path: str, output_path: str = None) -> None:
        """
        Process a single image file for emotion detection.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                logging.warning("No faces detected in the image")
                return
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = img[y:y+h, x:x+w]
                
                # Predict with all available models
                for model_name in self.models:
                    emotion, emoji, confidence = self.predict_emotion(model_name, face_roi)
                    
                    # Draw rectangle and label
                    color = (0, 255, 0) if model_name == "transfer" else (255, 0, 0)
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    
                    label = f"{model_name}: {emotion} {emoji} ({confidence:.2f})"
                    y_offset = y - 10 - (len(self.models) - list(self.models.keys()).index(model_name) - 1) * 25
                    cv2.putText(img, label, (x, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save or display result
            if output_path:
                cv2.imwrite(output_path, img)
                logging.info(f"Result saved to {output_path}")
            else:
                cv2.imshow('Emotion Detection Result', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise


def main():
    """Main function for the emotion detection app."""
    parser = argparse.ArgumentParser(description="Emotion Detection Application")
    parser.add_argument(
        '--model', 
        choices=['transfer', 'no_transfer', 'both'], 
        default='both',
        help='Model type to use'
    )
    parser.add_argument(
        '--mode', 
        choices=['webcam', 'image'], 
        default='webcam',
        help='Detection mode'
    )
    parser.add_argument(
        '--input', 
        help='Input image path (for image mode)'
    )
    parser.add_argument(
        '--output', 
        help='Output image path (for image mode)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize app
        app = EmotionDetectionApp(model_type=args.model)
        
        if args.mode == 'webcam':
            app.run_webcam_detection()
        elif args.mode == 'image':
            if not args.input:
                raise ValueError("Input image path required for image mode")
            app.process_image_file(args.input, args.output)
            
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()