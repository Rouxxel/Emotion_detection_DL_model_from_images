#!/usr/bin/env python3
"""
Data Preprocessing Pipeline

This module provides comprehensive data preprocessing functionality
for the emotion detection dataset.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline."""
    
    def __init__(self, config: Dict[str, any]):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Image parameters
        self.img_height = config["images"]["height"]
        self.img_width = config["images"]["width"]
        self.class_labels = config["classes"]["labels"]
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_labels)
        
    def load_dataset(self, dataset_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from directory structure.
        
        Args:
            dataset_path: Path to dataset directory
            split: Dataset split ('train' or 'test')
            
        Returns:
            Tuple of (images, labels)
        """
        dataset_path = Path(dataset_path) / split
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        images = []
        labels = []
        
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir() or class_dir.name not in self.class_labels:
                continue
            
            class_label = class_dir.name
            self.logger.info(f"Loading {class_label} images...")
            
            # Get all image files
            image_files = list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg"))
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_label)
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_path}: {str(e)}")
                    continue
        
        self.logger.info(f"Loaded {len(images)} images from {split} set")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels
    
    def _load_and_preprocess_image(self, img_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image array or None if failed
        """
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize pixel values
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {img_path}: {str(e)}")
            return None
    
    def augment_data(self, images: np.ndarray, labels: np.ndarray, 
                    target_samples_per_class: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data to balance classes.
        
        Args:
            images: Input images
            labels: Input labels
            target_samples_per_class: Target number of samples per class
            
        Returns:
            Augmented images and labels
        """
        self.logger.info("Starting data augmentation...")
        
        # Create data generator for augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        augmented_images = []
        augmented_labels = []
        
        # Process each class
        for class_label in self.class_labels:
            # Get images for this class
            class_mask = labels == class_label
            class_images = images[class_mask]
            class_labels_array = labels[class_mask]
            
            current_count = len(class_images)
            self.logger.info(f"Class {class_label}: {current_count} samples")
            
            # Add original images
            augmented_images.extend(class_images)
            augmented_labels.extend(class_labels_array)
            
            # Generate augmented images if needed
            if current_count < target_samples_per_class:
                needed = target_samples_per_class - current_count
                self.logger.info(f"Generating {needed} augmented samples for {class_label}")
                
                # Generate augmented images
                generated = 0
                while generated < needed:
                    for img in class_images:
                        if generated >= needed:
                            break
                        
                        # Reshape for data generator
                        img_batch = np.expand_dims(img, axis=0)
                        
                        # Generate augmented image
                        aug_iter = datagen.flow(img_batch, batch_size=1)
                        aug_img = next(aug_iter)[0]
                        
                        augmented_images.append(aug_img)
                        augmented_labels.append(class_label)
                        generated += 1
        
        self.logger.info(f"Augmentation complete. Total samples: {len(augmented_images)}")
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def encode_labels(self, labels: np.ndarray, categorical: bool = True) -> np.ndarray:
        """
        Encode string labels to numerical format.
        
        Args:
            labels: String labels
            categorical: Whether to return categorical (one-hot) encoding
            
        Returns:
            Encoded labels
        """
        # Convert to numerical labels
        numerical_labels = self.label_encoder.transform(labels)
        
        if categorical:
            # Convert to categorical (one-hot) encoding
            return to_categorical(numerical_labels, num_classes=len(self.class_labels))
        
        return numerical_labels
    
    def create_data_generators(self, train_images: np.ndarray, train_labels: np.ndarray,
                              val_images: np.ndarray, val_labels: np.ndarray,
                              batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow data generators.
        
        Args:
            train_images: Training images
            train_labels: Training labels
            val_images: Validation images
            val_labels: Validation labels
            batch_size: Batch size
            
        Returns:
            Training and validation datasets
        """
        # Create training dataset with augmentation
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def split_data(self, images: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            images: Input images
            labels: Input labels
            test_size: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Train and validation splits
        """
        return train_test_split(
            images, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
    
    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            labels: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get unique classes and their counts
        unique_labels = np.unique(labels)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        # Convert to dictionary
        class_weight_dict = {}
        for i, label in enumerate(unique_labels):
            # Get numerical index for this label
            label_idx = self.label_encoder.transform([label])[0]
            class_weight_dict[label_idx] = class_weights[i]
        
        return class_weight_dict
    
    def preprocess_for_model(self, images: np.ndarray, model_type: str = "transfer") -> np.ndarray:
        """
        Preprocess images for specific model types.
        
        Args:
            images: Input images
            model_type: Type of model ("transfer" or "custom")
            
        Returns:
            Preprocessed images
        """
        if model_type == "transfer":
            # For transfer learning models (RGB, DenseNet preprocessing)
            if images.shape[-1] == 1:
                # Convert grayscale to RGB
                images = np.repeat(images, 3, axis=-1)
            
            # Apply DenseNet preprocessing
            from tensorflow.keras.applications.densenet import preprocess_input
            return preprocess_input(images * 255.0)
        
        elif model_type == "custom":
            # For custom CNN (grayscale)
            if len(images.shape) == 4 and images.shape[-1] == 3:
                # Convert RGB to grayscale
                images = np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])
                images = np.expand_dims(images, axis=-1)
            
            return images
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def save_preprocessing_info(self, output_path: str, preprocessing_stats: Dict[str, any]) -> None:
        """
        Save preprocessing information for reproducibility.
        
        Args:
            output_path: Path to save preprocessing info
            preprocessing_stats: Statistics and parameters used
        """
        info = {
            "config": self.config,
            "class_labels": self.class_labels,
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
            "preprocessing_stats": preprocessing_stats,
            "image_shape": (self.img_height, self.img_width),
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        self.logger.info(f"Preprocessing info saved to {output_path}")


class DataQualityChecker:
    """Check data quality and provide recommendations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_image_quality(self, images: np.ndarray) -> Dict[str, any]:
        """
        Check image quality metrics.
        
        Args:
            images: Array of images
            
        Returns:
            Quality metrics dictionary
        """
        quality_metrics = {
            "total_images": len(images),
            "mean_brightness": [],
            "mean_contrast": [],
            "blur_scores": [],
            "low_quality_count": 0
        }
        
        for img in images[:min(1000, len(images))]:  # Sample for performance
            # Convert to uint8 for OpenCV
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Brightness (mean pixel value)
            brightness = np.mean(img_uint8)
            quality_metrics["mean_brightness"].append(brightness)
            
            # Contrast (standard deviation)
            contrast = np.std(img_uint8)
            quality_metrics["mean_contrast"].append(contrast)
            
            # Blur detection (Laplacian variance)
            if len(img_uint8.shape) == 3:
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_uint8
            
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics["blur_scores"].append(blur_score)
            
            # Count low quality images
            if brightness < 50 or contrast < 20 or blur_score < 100:
                quality_metrics["low_quality_count"] += 1
        
        # Calculate statistics
        quality_metrics["avg_brightness"] = np.mean(quality_metrics["mean_brightness"])
        quality_metrics["avg_contrast"] = np.mean(quality_metrics["mean_contrast"])
        quality_metrics["avg_blur_score"] = np.mean(quality_metrics["blur_scores"])
        quality_metrics["low_quality_percentage"] = (quality_metrics["low_quality_count"] / 
                                                    len(quality_metrics["mean_brightness"])) * 100
        
        return quality_metrics
    
    def generate_recommendations(self, quality_metrics: Dict[str, any], 
                               class_distribution: Dict[str, int]) -> List[str]:
        """
        Generate data quality recommendations.
        
        Args:
            quality_metrics: Image quality metrics
            class_distribution: Class distribution statistics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check image quality
        if quality_metrics["low_quality_percentage"] > 10:
            recommendations.append(
                f"High percentage of low-quality images ({quality_metrics['low_quality_percentage']:.1f}%). "
                "Consider image enhancement or filtering."
            )
        
        if quality_metrics["avg_brightness"] < 80:
            recommendations.append("Images appear dark. Consider brightness enhancement.")
        
        if quality_metrics["avg_contrast"] < 30:
            recommendations.append("Low contrast detected. Consider contrast enhancement.")
        
        if quality_metrics["avg_blur_score"] < 200:
            recommendations.append("Some images may be blurry. Consider sharpening filters.")
        
        # Check class balance
        counts = list(class_distribution.values())
        if counts:
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = 1 - (min_count / max_count) if max_count > 0 else 0
            
            if imbalance_ratio > 0.3:
                recommendations.append(
                    f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                    "Consider data augmentation or resampling."
                )
        
        # Check dataset size
        total_samples = sum(counts) if counts else 0
        if total_samples < 1000:
            recommendations.append("Small dataset size. Consider data augmentation.")
        
        return recommendations


def main():
    """Example usage of the data preprocessing pipeline."""
    # Load configuration
    from configuration.config_invoke import load_config
    config = load_config()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load dataset
    try:
        train_images, train_labels = preprocessor.load_dataset("dataset", "train")
        
        # Check data quality
        quality_checker = DataQualityChecker()
        quality_metrics = quality_checker.check_image_quality(train_images)
        
        # Get class distribution
        unique, counts = np.unique(train_labels, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        # Generate recommendations
        recommendations = quality_checker.generate_recommendations(quality_metrics, class_distribution)
        
        print("Data Quality Report:")
        print(f"Total images: {quality_metrics['total_images']}")
        print(f"Average brightness: {quality_metrics['avg_brightness']:.1f}")
        print(f"Average contrast: {quality_metrics['avg_contrast']:.1f}")
        print(f"Low quality percentage: {quality_metrics['low_quality_percentage']:.1f}%")
        
        print("\nClass Distribution:")
        for class_name, count in class_distribution.items():
            print(f"  {class_name}: {count}")
        
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please run setup first to download the dataset.")


if __name__ == "__main__":
    main()
