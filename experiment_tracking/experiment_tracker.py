#!/usr/bin/env python3
"""
Experiment Tracking Module

This module provides functionality to track machine learning experiments,
model versions, and performance metrics.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import hashlib
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentTracker:
    """Track machine learning experiments and model versions."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment data
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "metadata").mkdir(exist_ok=True)
        
        self.current_experiment = None
        self.logger = logging.getLogger(__name__)
        
        # Load or create experiment registry
        self.registry_path = self.experiment_dir / "experiment_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the experiment registry."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"experiments": {}, "next_id": 1}
    
    def _save_registry(self) -> None:
        """Save the experiment registry."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        exp_id = f"exp_{self.registry['next_id']:04d}"
        self.registry['next_id'] += 1
        return exp_id
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def start_experiment(self, 
                        name: str,
                        description: str = "",
                        tags: List[str] = None,
                        config: Dict[str, Any] = None) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: List of tags for categorization
            config: Configuration dictionary
            
        Returns:
            Experiment ID
        """
        exp_id = self._generate_experiment_id()
        timestamp = datetime.now().isoformat()
        
        if config is None:
            config = {}
        
        experiment_data = {
            "id": exp_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "config": config,
            "config_hash": self._calculate_config_hash(config),
            "start_time": timestamp,
            "end_time": None,
            "status": "running",
            "metrics": {},
            "artifacts": [],
            "model_path": None,
            "plots": []
        }
        
        self.registry["experiments"][exp_id] = experiment_data
        self.current_experiment = exp_id
        self._save_registry()
        
        # Create experiment directory
        exp_dir = self.experiment_dir / "metadata" / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Started experiment {exp_id}: {name}")
        return exp_id
    
    def log_metric(self, name: str, value: Union[float, int], step: int = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        exp_data = self.registry["experiments"][self.current_experiment]
        
        if name not in exp_data["metrics"]:
            exp_data["metrics"][name] = []
        
        metric_entry = {
            "value": float(value),
            "timestamp": datetime.now().isoformat()
        }
        
        if step is not None:
            metric_entry["step"] = step
        
        exp_data["metrics"][name].append(metric_entry)
        self._save_registry()
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int = None) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        exp_data = self.registry["experiments"][self.current_experiment]
        exp_data["config"].update(params)
        exp_data["config_hash"] = self._calculate_config_hash(exp_data["config"])
        self._save_registry()
    
    def log_artifact(self, file_path: str, artifact_type: str = "file") -> None:
        """
        Log an artifact (file, plot, etc.).
        
        Args:
            file_path: Path to the artifact file
            artifact_type: Type of artifact
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        
        # Copy artifact to experiment directory
        exp_dir = self.experiment_dir / "metadata" / self.current_experiment
        dest_path = exp_dir / file_path.name
        shutil.copy2(file_path, dest_path)
        
        exp_data = self.registry["experiments"][self.current_experiment]
        artifact_entry = {
            "path": str(dest_path.relative_to(self.experiment_dir)),
            "type": artifact_type,
            "timestamp": datetime.now().isoformat(),
            "size_bytes": dest_path.stat().st_size
        }
        
        exp_data["artifacts"].append(artifact_entry)
        self._save_registry()
    
    def save_model(self, model, model_name: str = None) -> str:
        """
        Save a trained model.
        
        Args:
            model: Trained model object
            model_name: Optional model name
            
        Returns:
            Path to saved model
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        if model_name is None:
            model_name = f"model_{self.current_experiment}.h5"
        
        model_path = self.experiment_dir / "models" / model_name
        
        # Save model (assuming Keras/TensorFlow model)
        try:
            model.save(model_path)
        except AttributeError:
            # Fallback to pickle for other model types
            with open(model_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f)
            model_path = model_path.with_suffix('.pkl')
        
        exp_data = self.registry["experiments"][self.current_experiment]
        exp_data["model_path"] = str(model_path.relative_to(self.experiment_dir))
        self._save_registry()
        
        self.logger.info(f"Model saved: {model_path}")
        return str(model_path)
    
    def plot_metrics(self, metrics: List[str] = None, save_plot: bool = True) -> None:
        """
        Plot experiment metrics.
        
        Args:
            metrics: List of metrics to plot. If None, plots all metrics.
            save_plot: Whether to save the plot as an artifact
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        exp_data = self.registry["experiments"][self.current_experiment]
        
        if not exp_data["metrics"]:
            self.logger.warning("No metrics to plot")
            return
        
        if metrics is None:
            metrics = list(exp_data["metrics"].keys())
        
        # Create subplots
        n_metrics = len(metrics)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2 if n_metrics > 1 else 1,
            figsize=(12, 4 * ((n_metrics + 1) // 2))
        )
        
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics > 1:
            axes = axes.flatten()
        
        for i, metric_name in enumerate(metrics):
            if metric_name not in exp_data["metrics"]:
                continue
            
            metric_data = exp_data["metrics"][metric_name]
            
            # Extract values and steps
            values = [entry["value"] for entry in metric_data]
            steps = [entry.get("step", i) for i, entry in enumerate(metric_data)]
            
            axes[i].plot(steps, values, marker='o', linewidth=2, markersize=4)
            axes[i].set_title(f"{metric_name.replace('_', ' ').title()}")
            axes[i].set_xlabel("Step/Epoch")
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.experiment_dir / "plots" / f"{self.current_experiment}_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Log as artifact
            exp_data["plots"].append(str(plot_path.relative_to(self.experiment_dir)))
            self._save_registry()
            
            self.logger.info(f"Metrics plot saved: {plot_path}")
        
        plt.show()
    
    def end_experiment(self, status: str = "completed") -> None:
        """
        End the current experiment.
        
        Args:
            status: Final status of the experiment
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment to end.")
        
        exp_data = self.registry["experiments"][self.current_experiment]
        exp_data["end_time"] = datetime.now().isoformat()
        exp_data["status"] = status
        
        # Calculate duration
        start_time = datetime.fromisoformat(exp_data["start_time"])
        end_time = datetime.fromisoformat(exp_data["end_time"])
        duration = (end_time - start_time).total_seconds()
        exp_data["duration_seconds"] = duration
        
        self._save_registry()
        
        self.logger.info(f"Experiment {self.current_experiment} ended with status: {status}")
        self.current_experiment = None
    
    def list_experiments(self, tags: List[str] = None, status: str = None) -> pd.DataFrame:
        """
        List all experiments with optional filtering.
        
        Args:
            tags: Filter by tags
            status: Filter by status
            
        Returns:
            DataFrame with experiment information
        """
        experiments = []
        
        for exp_id, exp_data in self.registry["experiments"].items():
            # Apply filters
            if tags and not any(tag in exp_data.get("tags", []) for tag in tags):
                continue
            
            if status and exp_data.get("status") != status:
                continue
            
            # Get best metrics
            best_metrics = {}
            for metric_name, metric_data in exp_data.get("metrics", {}).items():
                if metric_data:
                    values = [entry["value"] for entry in metric_data]
                    if "loss" in metric_name.lower():
                        best_metrics[f"best_{metric_name}"] = min(values)
                    else:
                        best_metrics[f"best_{metric_name}"] = max(values)
            
            experiment_info = {
                "id": exp_id,
                "name": exp_data.get("name", ""),
                "status": exp_data.get("status", "unknown"),
                "start_time": exp_data.get("start_time", ""),
                "duration_seconds": exp_data.get("duration_seconds", 0),
                "tags": ", ".join(exp_data.get("tags", [])),
                **best_metrics
            }
            
            experiments.append(experiment_info)
        
        return pd.DataFrame(experiments)
    
    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Experiment data dictionary
        """
        if exp_id not in self.registry["experiments"]:
            raise ValueError(f"Experiment {exp_id} not found")
        
        return self.registry["experiments"][exp_id]
    
    def compare_experiments(self, exp_ids: List[str], metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison data
        """
        comparison_data = []
        
        for exp_id in exp_ids:
            if exp_id not in self.registry["experiments"]:
                self.logger.warning(f"Experiment {exp_id} not found, skipping")
                continue
            
            exp_data = self.registry["experiments"][exp_id]
            
            row = {
                "experiment_id": exp_id,
                "name": exp_data.get("name", ""),
                "status": exp_data.get("status", "unknown")
            }
            
            # Add configuration parameters
            for key, value in exp_data.get("config", {}).items():
                row[f"config_{key}"] = value
            
            # Add metrics
            exp_metrics = exp_data.get("metrics", {})
            if metrics is None:
                metrics = list(exp_metrics.keys())
            
            for metric_name in metrics:
                if metric_name in exp_metrics and exp_metrics[metric_name]:
                    values = [entry["value"] for entry in exp_metrics[metric_name]]
                    row[f"final_{metric_name}"] = values[-1]
                    
                    if "loss" in metric_name.lower():
                        row[f"best_{metric_name}"] = min(values)
                    else:
                        row[f"best_{metric_name}"] = max(values)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def delete_experiment(self, exp_id: str, confirm: bool = False) -> None:
        """
        Delete an experiment and all its artifacts.
        
        Args:
            exp_id: Experiment ID to delete
            confirm: Confirmation flag
        """
        if not confirm:
            raise ValueError("Must set confirm=True to delete experiment")
        
        if exp_id not in self.registry["experiments"]:
            raise ValueError(f"Experiment {exp_id} not found")
        
        # Remove experiment directory
        exp_dir = self.experiment_dir / "metadata" / exp_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        
        # Remove model if exists
        exp_data = self.registry["experiments"][exp_id]
        if exp_data.get("model_path"):
            model_path = self.experiment_dir / exp_data["model_path"]
            if model_path.exists():
                model_path.unlink()
        
        # Remove from registry
        del self.registry["experiments"][exp_id]
        self._save_registry()
        
        self.logger.info(f"Experiment {exp_id} deleted")


def main():
    """Example usage of the experiment tracker."""
    tracker = ExperimentTracker()
    
    # Start an experiment
    exp_id = tracker.start_experiment(
        name="Transfer Learning Test",
        description="Testing transfer learning with DenseNet121",
        tags=["transfer_learning", "densenet"],
        config={
            "model_type": "transfer_learning",
            "base_model": "DenseNet121",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    )
    
    # Log some metrics
    for epoch in range(10):
        tracker.log_metrics({
            "train_loss": 0.5 - epoch * 0.03,
            "train_accuracy": 0.6 + epoch * 0.04,
            "val_loss": 0.6 - epoch * 0.025,
            "val_accuracy": 0.55 + epoch * 0.035
        }, step=epoch)
    
    # Plot metrics
    tracker.plot_metrics()
    
    # End experiment
    tracker.end_experiment("completed")
    
    # List experiments
    experiments_df = tracker.list_experiments()
    print(experiments_df)


if __name__ == "__main__":
    main()