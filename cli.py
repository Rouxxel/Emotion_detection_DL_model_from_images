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
    try:
        # Initialize experiment tracking if requested
        tracker = None
        if args.track_experiment:
            from experiment_tracking.experiment_tracker import ExperimentTracker
            tracker = ExperimentTracker()
            
            exp_name = args.experiment_name or f"{args.model}_training"
            exp_id = tracker.start_experiment(
                name=exp_name,
                description=f"Training {args.model} model(s)",
                tags=[args.model, "cli_training"],
                config={"model_type": args.model, "cli_args": vars(args)}
            )
            print(f"📊 Started experiment tracking: {exp_id}")
        
        if args.model in ['transfer', 'both']:
            print("🚀 Training transfer learning model...")
            from dl_scripts.train_transfer_learning import main as train_transfer
            train_transfer()
            print("✅ Transfer learning model training completed!")
        
        if args.model in ['no-transfer', 'both']:
            print("🚀 Training custom CNN model...")
            from dl_scripts.train_no_transfer_learning import main as train_no_transfer
            train_no_transfer()
            print("✅ Custom CNN model training completed!")
        
        # End experiment tracking
        if tracker:
            tracker.end_experiment("completed")
            print("📊 Experiment tracking completed!")
            
    except Exception as e:
        if tracker:
            tracker.end_experiment("failed")
        print(f"❌ Training failed: {e}")
        sys.exit(1)

def cmd_experiments(args) -> None:
    """Manage experiments."""
    try:
        from experiment_tracking.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        
        if args.action == 'list':
            experiments_df = tracker.list_experiments()
            if experiments_df.empty:
                print("No experiments found.")
            else:
                print("📊 Experiments:")
                print(experiments_df.to_string(index=False))
        
        elif args.action == 'show':
            if not args.experiment_id:
                print("❌ Experiment ID required for 'show' action")
                sys.exit(1)
            
            exp_data = tracker.get_experiment(args.experiment_id)
            print(f"📊 Experiment {args.experiment_id}:")
            print(f"  Name: {exp_data.get('name', 'N/A')}")
            print(f"  Status: {exp_data.get('status', 'N/A')}")
            print(f"  Start Time: {exp_data.get('start_time', 'N/A')}")
            print(f"  Duration: {exp_data.get('duration_seconds', 0):.1f}s")
            
            if exp_data.get('metrics'):
                print("  Metrics:")
                for metric_name, metric_data in exp_data['metrics'].items():
                    if metric_data:
                        final_value = metric_data[-1]['value']
                        print(f"    {metric_name}: {final_value:.4f}")
        
        elif args.action == 'compare':
            if not args.experiment_ids or len(args.experiment_ids) < 2:
                print("❌ At least 2 experiment IDs required for comparison")
                sys.exit(1)
            
            comparison_df = tracker.compare_experiments(args.experiment_ids)
            print("📊 Experiment Comparison:")
            print(comparison_df.to_string(index=False))
        
        elif args.action == 'delete':
            if not args.experiment_id:
                print("❌ Experiment ID required for 'delete' action")
                sys.exit(1)
            
            if not args.confirm:
                print("❌ Use --confirm flag to confirm deletion")
                sys.exit(1)
            
            tracker.delete_experiment(args.experiment_id, confirm=True)
            print(f"✅ Experiment {args.experiment_id} deleted")
            
    except Exception as e:
        print(f"❌ Experiment management failed: {e}")
        sys.exit(1)

def cmd_ui(args) -> None:
    """Launch the user interface."""
    try:
        print("🎯 Launching emotion detection interface...")
        from user_interface.emotion_detection_app import main as ui_main
        
        # Prepare arguments for the UI
        ui_args = [
            '--model', args.model_type,
            '--mode', args.mode
        ]
        
        if args.input:
            ui_args.extend(['--input', args.input])
        if args.output:
            ui_args.extend(['--output', args.output])
        
        # Temporarily replace sys.argv for the UI
        original_argv = sys.argv
        sys.argv = ['emotion_detection_app.py'] + ui_args
        
        try:
            ui_main()
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"❌ UI launch failed: {e}")
        sys.exit(1)

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
    train_parser.add_argument(
        '--track-experiment',
        action='store_true',
        help='Enable experiment tracking'
    )
    train_parser.add_argument(
        '--experiment-name',
        help='Name for the experiment'
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Experiments command
    exp_parser = subparsers.add_parser('experiments', help='Manage experiments')
    exp_parser.add_argument(
        'action',
        choices=['list', 'show', 'compare', 'delete'],
        help='Action to perform'
    )
    exp_parser.add_argument(
        '--experiment-id',
        help='Experiment ID for show/delete actions'
    )
    exp_parser.add_argument(
        '--experiment-ids',
        nargs='+',
        help='Experiment IDs for compare action'
    )
    exp_parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm deletion'
    )
    exp_parser.set_defaults(func=cmd_experiments)
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch user interface')
    ui_parser.add_argument(
        '--model-type',
        choices=['transfer', 'no_transfer', 'both'],
        default='both',
        help='Which model to use for detection'
    )
    ui_parser.add_argument(
        '--mode',
        choices=['webcam', 'image'],
        default='webcam',
        help='Detection mode'
    )
    ui_parser.add_argument(
        '--input',
        help='Input image path (for image mode)'
    )
    ui_parser.add_argument(
        '--output',
        help='Output image path (for image mode)'
    )
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