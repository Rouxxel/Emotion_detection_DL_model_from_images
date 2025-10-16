#!/usr/bin/env python3
"""
Command Line Interface for Emotion Detection Deep Learning Project

This CLI provides easy access to all project functionality including setup,
training, and running the user interface.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from set_up.full_set_up import full_set_up
from configuration.config_invoke import load_config

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('emotion_detection.log')
        ]
    )

def cmd_setup(args) -> None:
    """Run the full project setup."""
    try:
        full_set_up(
            kaggle_json_path=args.kaggle_path,
            requirements_path=args.requirements,
            dataset_name=args.dataset,
            dataset_dir=args.dataset_dir,
            train_subdir=args.train_subdir,
            augmentation_target=args.augmentation_target
        )
        print("✅ Setup completed successfully!")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

def cmd_config(args) -> None:
    """Show current configuration."""
    try:
        config = load_config()
        print("📋 Current Configuration:")
        print("=" * 50)
        
        for section, values in config.items():
            print(f"\n[{section.upper()}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {values}")
                
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

def cmd_train(args) -> None:
    """Train the emotion detection models."""
    print("🚀 Training functionality will be implemented when notebooks are converted to scripts.")
    print("For now, please run the Jupyter notebooks in dl_scripts/ directory.")

def cmd_ui(args) -> None:
    """Launch the user interface."""
    print("🎯 UI functionality will be implemented when notebooks are converted to scripts.")
    print("For now, please run the Jupyter notebook in user_interface/ directory.")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Emotion Detection Deep Learning Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup                           # Run full setup with defaults
  %(prog)s setup --kaggle-path ~/.kaggle/kaggle.json
  %(prog)s config                          # Show current configuration
  %(prog)s train --model transfer          # Train transfer learning model
  %(prog)s ui                              # Launch user interface
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up the project')
    setup_parser.add_argument(
        '--kaggle-path',
        help='Path to Kaggle credentials JSON file'
    )
    setup_parser.add_argument(
        '--requirements',
        default='requirements.txt',
        help='Path to requirements.txt file'
    )
    setup_parser.add_argument(
        '--dataset',
        default='ananthu017/emotion-detection-fer',
        help='Kaggle dataset name'
    )
    setup_parser.add_argument(
        '--dataset-dir',
        default='dataset',
        help='Directory to store dataset'
    )
    setup_parser.add_argument(
        '--train-subdir',
        default='train',
        help='Training subdirectory name'
    )
    setup_parser.add_argument(
        '--augmentation-target',
        type=int,
        default=7000,
        help='Target number of images per class after augmentation'
    )
    setup_parser.set_defaults(func=cmd_setup)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.set_defaults(func=cmd_config)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument(
        '--model',
        choices=['transfer', 'no-transfer', 'both'],
        default='both',
        help='Which model to train'
    )
    train_parser.set_defaults(func=cmd_train)
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch user interface')
    ui_parser.set_defaults(func=cmd_ui)
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the command
    args.func(args)

if __name__ == '__main__':
    main()