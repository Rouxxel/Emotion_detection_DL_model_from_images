#!/usr/bin/env python3
"""
Transfer Learning Model Training Script

This script implements a deep learning model for emotion detection using transfer learning.
It uses a pre-trained model as a base and fine-tunes it for emotion classification.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import json

# Dataset analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Performance metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configuration.config_invoke import load_config


class TransferLearningTrainer:
    """Trainer class for emotion detection using transfer learning."""
    
    def __init__(self, config_path: str = None):
        """Initialize the trainer with configuration."""
        self.config = load_config() if config_path is None else self._load_custom_config(config_path)
        self.model = None
        self.history = None
        self.class_names = self.config["classes"]["labels"]
        
        # Set up logging
        self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
    def _load_custom_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from custom path."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = "log_history_transfer_learning.log"
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info(f"Transfer learning model training started. Logs saved to {log_file}")
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config["dl_model"]["fixed_seed"]
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logging.info(f"Random seeds set to {seed}")
    
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare training and validation datasets."""
        try:
            img_height = self.config["images"]["height"]
            img_width = self.config["images"]["width"]
            batch_size = self.config["dl_model"]["batch_size"]
            train_dir = self.config["datasets"]["train_directory"]
            
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
            
            # Only rescaling for validation
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            # Create generators
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training',
                color_mode='rgb'  # Transfer learning models typically use RGB
            )
            
            val_generator = val_datagen.flow_from_directory(
                train_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                color_mode='rgb'
            )
            
            logging.info(f"Training samples: {train_generator.samples}")
            logging.info(f"Validation samples: {val_generator.samples}")
            logging.info(f"Number of classes: {train_generator.num_classes}")
            
            return train_generator, val_generator
            
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise
    
    def build_model(self) -> Model:
        """Build the transfer learning model."""
        try:
            img_height = self.config["images"]["height"]
            img_width = self.config["images"]["width"]
            num_classes = len(self.class_names)
            
            # Load pre-trained DenseNet121
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(img_height, img_width, 3)
            )
            
            # Freeze base model layers initially
            base_model.trainable = False
            
            # Add custom classification head
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile model
            learning_rate = self.config["dl_model"]["learn_r"]
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logging.info("Transfer learning model built successfully")
            logging.info(f"Total parameters: {model.count_params():,}")
            logging.info(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
            
            self.model = model
            return model
            
        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            raise
    
    def train_model(self, train_generator, val_generator) -> Dict[str, Any]:
        """Train the model with transfer learning approach."""
        try:
            if self.model is None:
                raise ValueError("Model not built. Call build_model() first.")
            
            epochs = self.config["dl_model"]["transfer_learning"]["epoch"]
            fine_tuning_epochs = self.config["dl_model"]["transfer_learning"]["fine_tuning_epochs"]
            early_stop_patience = self.config["dl_model"]["transfer_learning"]["early_stop_crit"]
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=early_stop_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-7
                )
            ]
            
            # Phase 1: Train with frozen base model
            logging.info("Phase 1: Training with frozen base model")
            history1 = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Phase 2: Fine-tuning with unfrozen layers
            logging.info("Phase 2: Fine-tuning with unfrozen layers")
            
            # Unfreeze the base model
            self.model.layers[0].trainable = True
            
            # Use a lower learning rate for fine-tuning
            self.model.compile(
                optimizer=Adam(learning_rate=self.config["dl_model"]["learn_r"] / 10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history2 = self.model.fit(
                train_generator,
                epochs=fine_tuning_epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            self.history = {
                'loss': history1.history['loss'] + history2.history['loss'],
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
            }
            
            logging.info("Model training completed successfully")
            return self.history
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self) -> str:
        """Save the trained model."""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")
            
            # Create directory if it doesn't exist
            model_dir = self.config["dl_model"]["directory"]
            os.makedirs(model_dir, exist_ok=True)
            
            model_name = self.config["dl_model"]["transfer_learning"]["name"]
            model_path = os.path.join(model_dir, model_name)
            
            self.model.save(model_path)
            logging.info(f"Model saved to {model_path}")
            
            return model_path
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def evaluate_model(self, test_generator) -> Dict[str, float]:
        """Evaluate the model on test data."""
        try:
            if self.model is None:
                raise ValueError("No model to evaluate. Train the model first.")
            
            # Evaluate on test data
            test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
            
            # Get predictions
            predictions = self.model.predict(test_generator)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_generator.classes
            
            # Calculate metrics
            metrics = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'precision': precision_score(true_classes, predicted_classes, average='weighted'),
                'recall': recall_score(true_classes, predicted_classes, average='weighted'),
                'f1_score': f1_score(true_classes, predicted_classes, average='weighted')
            }
            
            logging.info("Model evaluation completed")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise
    
    def plot_training_history(self, save_path: str = None) -> None:
        """Plot training history."""
        if self.history is None:
            logging.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history['loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Training history plot saved to {save_path}")
        
        plt.show()


def main():
    """Main training function."""
    try:
        # Initialize trainer
        trainer = TransferLearningTrainer()
        
        # Prepare data
        train_gen, val_gen = trainer.prepare_data()
        
        # Build model
        trainer.build_model()
        
        # Train model
        trainer.train_model(train_gen, val_gen)
        
        # Save model
        model_path = trainer.save_model()
        
        # Plot training history
        trainer.plot_training_history("transfer_learning_history.png")
        
        logging.info("Transfer learning model training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
