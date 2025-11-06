#!/usr/bin/env python3
"""
Data Validation Module

This module provides functionality to validate dataset structure and integrity.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
from PIL import Image
import pandas as pd


class DatasetValidator:
    """Validator for emotion detection dataset."""
    
    def __init__(self, dataset_path: str, expected_classes: List[str]):
        """
        Initialize the dataset validator.
        
        Args:
            dataset_path: Path to the dataset directory
            expected_classes: List of expected emotion class names
        """
        self.dataset_path = Path(dataset_path)
        self.expected_classes = set(expected_classes)
        self.validation_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_structure(self) -> Dict[str, bool]:
        """
        Validate the basic structure of the dataset.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'dataset_exists': False,
            'train_dir_exists': False,
            'test_dir_exists': False,
            'all_classes_present_train': False,
            'all_classes_present_test': False
        }
        
        try:
            # Check if dataset directory exists
            if self.dataset_path.exists():
                results['dataset_exists'] = True
                self.logger.info(f"Dataset directory found: {self.dataset_path}")
            else:
                self.logger.error(f"Dataset directory not found: {self.dataset_path}")
                return results
            
            # Check train and test directories
            train_dir = self.dataset_path / "train"
            test_dir = self.dataset_path / "test"
            
            if train_dir.exists():
                results['train_dir_exists'] = True
                self.logger.info("Train directory found")
            else:
                self.logger.error("Train directory not found")
            
            if test_dir.exists():
                results['test_dir_exists'] = True
                self.logger.info("Test directory found")
            else:
                self.logger.error("Test directory not found")
            
            # Check class directories
            if results['train_dir_exists']:
                train_classes = set([d.name for d in train_dir.iterdir() if d.is_dir()])
                results['all_classes_present_train'] = self.expected_classes.issubset(train_classes)
                
                if results['all_classes_present_train']:
                    self.logger.info("All expected classes found in train directory")
                else:
                    missing = self.expected_classes - train_classes
                    self.logger.warning(f"Missing classes in train directory: {missing}")
            
            if results['test_dir_exists']:
                test_classes = set([d.name for d in test_dir.iterdir() if d.is_dir()])
                results['all_classes_present_test'] = self.expected_classes.issubset(test_classes)
                
                if results['all_classes_present_test']:
                    self.logger.info("All expected classes found in test directory")
                else:
                    missing = self.expected_classes - test_classes
                    self.logger.warning(f"Missing classes in test directory: {missing}")
            
        except Exception as e:
            self.logger.error(f"Error validating dataset structure: {str(e)}")
        
        self.validation_results.update(results)
        return results
    
    def validate_images(self, sample_size: int = 100) -> Dict[str, any]:
        """
        Validate image files in the dataset.
        
        Args:
            sample_size: Number of images to sample from each class for validation
            
        Returns:
            Dictionary with image validation results
        """
        results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'corrupted_files': [],
            'size_distribution': {},
            'format_distribution': {}
        }
        
        try:
            for split in ['train', 'test']:
                split_dir = self.dataset_path / split
                if not split_dir.exists():
                    continue
                
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir() or class_dir.name not in self.expected_classes:
                        continue
                    
                    # Get image files
                    image_files = list(class_dir.glob('*.png')) + \
                                 list(class_dir.glob('*.jpg')) + \
                                 list(class_dir.glob('*.jpeg'))
                    
                    # Sample images if there are too many
                    if len(image_files) > sample_size:
                        image_files = np.random.choice(image_files, sample_size, replace=False)
                    
                    for img_path in image_files:
                        results['total_images'] += 1
                        
                        try:
                            with Image.open(img_path) as img:
                                # Check if image can be loaded
                                img.verify()
                                
                                # Reopen for size check (verify() closes the image)
                                with Image.open(img_path) as img2:
                                    size = img2.size
                                    format_type = img2.format
                                    
                                    # Update distributions
                                    size_key = f"{size[0]}x{size[1]}"
                                    results['size_distribution'][size_key] = \
                                        results['size_distribution'].get(size_key, 0) + 1
                                    
                                    results['format_distribution'][format_type] = \
                                        results['format_distribution'].get(format_type, 0) + 1
                                
                                results['valid_images'] += 1
                                
                        except Exception as e:
                            results['invalid_images'] += 1
                            results['corrupted_files'].append(str(img_path))
                            self.logger.warning(f"Corrupted image: {img_path} - {str(e)}")
            
            # Log summary
            self.logger.info(f"Image validation complete:")
            self.logger.info(f"  Total images checked: {results['total_images']}")
            self.logger.info(f"  Valid images: {results['valid_images']}")
            self.logger.info(f"  Invalid images: {results['invalid_images']}")
            
        except Exception as e:
            self.logger.error(f"Error validating images: {str(e)}")
        
        self.validation_results.update(results)
        return results
    
    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Get the distribution of images per class.
        
        Returns:
            Dictionary with class distribution for train and test splits
        """
        distribution = {'train': {}, 'test': {}}
        
        try:
            for split in ['train', 'test']:
                split_dir = self.dataset_path / split
                if not split_dir.exists():
                    continue
                
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    
                    class_name = class_dir.name
                    
                    # Count image files
                    image_count = len(list(class_dir.glob('*.png'))) + \
                                 len(list(class_dir.glob('*.jpg'))) + \
                                 len(list(class_dir.glob('*.jpeg')))
                    
                    distribution[split][class_name] = image_count
            
            # Log distribution
            for split, classes in distribution.items():
                self.logger.info(f"{split.capitalize()} set distribution:")
                for class_name, count in classes.items():
                    self.logger.info(f"  {class_name}: {count} images")
                    
        except Exception as e:
            self.logger.error(f"Error getting class distribution: {str(e)}")
        
        return distribution
    
    def check_class_balance(self, imbalance_threshold: float = 0.3) -> Dict[str, any]:
        """
        Check for class imbalance in the dataset.
        
        Args:
            imbalance_threshold: Threshold for considering classes imbalanced
            
        Returns:
            Dictionary with class balance analysis
        """
        results = {
            'is_balanced': True,
            'imbalance_ratio': 0.0,
            'recommendations': []
        }
        
        try:
            distribution = self.get_class_distribution()
            
            for split in ['train', 'test']:
                if not distribution[split]:
                    continue
                
                counts = list(distribution[split].values())
                if not counts:
                    continue
                
                min_count = min(counts)
                max_count = max(counts)
                
                if min_count > 0:
                    imbalance_ratio = 1 - (min_count / max_count)
                    results['imbalance_ratio'] = max(results['imbalance_ratio'], imbalance_ratio)
                    
                    if imbalance_ratio > imbalance_threshold:
                        results['is_balanced'] = False
                        results['recommendations'].append(
                            f"Consider data augmentation for {split} set. "
                            f"Imbalance ratio: {imbalance_ratio:.2f}"
                        )
            
            if results['is_balanced']:
                self.logger.info("Dataset is reasonably balanced")
            else:
                self.logger.warning(f"Dataset imbalance detected. Ratio: {results['imbalance_ratio']:.2f}")
                for rec in results['recommendations']:
                    self.logger.warning(f"Recommendation: {rec}")
                    
        except Exception as e:
            self.logger.error(f"Error checking class balance: {str(e)}")
        
        return results
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            output_path: Optional path to save the report as JSON
            
        Returns:
            Complete validation report
        """
        report = {
            'dataset_path': str(self.dataset_path),
            'expected_classes': list(self.expected_classes),
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'structure_validation': self.validate_structure(),
            'image_validation': self.validate_images(),
            'class_distribution': self.get_class_distribution(),
            'balance_analysis': self.check_class_balance()
        }
        
        # Calculate overall health score
        structure_score = sum(report['structure_validation'].values()) / len(report['structure_validation'])
        image_score = report['image_validation']['valid_images'] / max(1, report['image_validation']['total_images'])
        balance_score = 1.0 if report['balance_analysis']['is_balanced'] else 0.5
        
        report['overall_health_score'] = (structure_score + image_score + balance_score) / 3
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Validation report saved to {output_path}")
        
        return report


def main():
    """Main function for running dataset validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate emotion detection dataset")
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--classes', nargs='+', 
                       default=['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise'],
                       help='Expected emotion classes')
    parser.add_argument('--output', help='Output path for validation report')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of images to sample for validation')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.dataset_path, args.classes)
    report = validator.generate_report(args.output)
    
    print(f"\n📊 Dataset Validation Report")
    print(f"{'='*50}")
    print(f"Dataset Path: {report['dataset_path']}")
    print(f"Overall Health Score: {report['overall_health_score']:.2f}/1.0")
    print(f"Structure Valid: {all(report['structure_validation'].values())}")
    print(f"Images Valid: {report['image_validation']['valid_images']}/{report['image_validation']['total_images']}")
    print(f"Balanced: {report['balance_analysis']['is_balanced']}")


if __name__ == "__main__":
    main()
