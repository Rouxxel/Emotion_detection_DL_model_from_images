#!/usr/bin/env python3
"""
Advanced Configuration Management

This module provides advanced configuration management with support for
multiple configuration sources, validation, and environment-specific settings.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass, field
from enum import Enum


class ConfigSource(Enum):
    """Configuration source types."""
    JSON = "json"
    YAML = "yaml"
    ENV = "environment"
    DICT = "dictionary"


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    path: str
    required: bool = True
    type_check: type = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None


class ConfigurationManager:
    """Advanced configuration manager with validation and multiple sources."""
    
    def __init__(self, base_config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        self.validation_rules = []
        self.config_sources = []
        
        # Load base configuration if provided
        if base_config_path:
            self.load_from_file(base_config_path)
    
    def load_from_file(self, file_path: str, source_priority: int = 0) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file
            source_priority: Priority of this source (higher = more important)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Determine file type
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            source_type = ConfigSource.JSON
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            source_type = ConfigSource.YAML
        else:
            raise ValueError(f"Unsupported configuration file type: {file_path.suffix}")
        
        self._merge_config(config_data, source_type, str(file_path), source_priority)
        self.logger.info(f"Loaded configuration from {file_path}")
    
    def load_from_dict(self, config_dict: Dict[str, Any], source_priority: int = 0) -> None:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            source_priority: Priority of this source
        """
        self._merge_config(config_dict, ConfigSource.DICT, "dictionary", source_priority)
    
    def load_from_environment(self, prefix: str = "EMOTION_DETECTION_", source_priority: int = 100) -> None:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            source_priority: Priority of this source
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Split nested keys
                keys = config_key.split('__')
                
                # Navigate to the correct nested dictionary
                current_dict = env_config
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = {}
                    current_dict = current_dict[k]
                
                # Set the value with type conversion
                final_key = keys[-1]
                current_dict[final_key] = self._convert_env_value(value)
        
        if env_config:
            self._merge_config(env_config, ConfigSource.ENV, "environment", source_priority)
            self.logger.info("Loaded configuration from environment variables")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate Python type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON (for lists, dicts)
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Return as string
        return value
    
    def _merge_config(self, new_config: Dict[str, Any], source_type: ConfigSource, 
                     source_name: str, priority: int) -> None:
        """Merge new configuration with existing configuration."""
        self.config_sources.append({
            "type": source_type,
            "name": source_name,
            "priority": priority,
            "config": new_config.copy()
        })
        
        # Sort sources by priority and merge
        self.config_sources.sort(key=lambda x: x["priority"])
        
        merged_config = {}
        for source in self.config_sources:
            merged_config = self._deep_merge(merged_config, source["config"])
        
        self.config = merged_config
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def add_validation_rule(self, rule: ConfigValidationRule) -> None:
        """Add a validation rule."""
        self.validation_rules.append(rule)
    
    def add_validation_rules(self, rules: List[ConfigValidationRule]) -> None:
        """Add multiple validation rules."""
        self.validation_rules.extend(rules)
    
    def validate(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for rule in self.validation_rules:
            try:
                value = self.get(rule.path)
                
                # Check if required
                if rule.required and value is None:
                    errors.append(f"Required configuration '{rule.path}' is missing")
                    continue
                
                if value is None:
                    continue  # Skip validation for optional missing values
                
                # Type check
                if rule.type_check and not isinstance(value, rule.type_check):
                    errors.append(f"Configuration '{rule.path}' must be of type {rule.type_check.__name__}")
                
                # Range checks
                if rule.min_value is not None and value < rule.min_value:
                    errors.append(f"Configuration '{rule.path}' must be >= {rule.min_value}")
                
                if rule.max_value is not None and value > rule.max_value:
                    errors.append(f"Configuration '{rule.path}' must be <= {rule.max_value}")
                
                # Allowed values check
                if rule.allowed_values and value not in rule.allowed_values:
                    errors.append(f"Configuration '{rule.path}' must be one of {rule.allowed_values}")
                
                # Custom validator
                if rule.custom_validator:
                    try:
                        if not rule.custom_validator(value):
                            errors.append(f"Configuration '{rule.path}' failed custom validation")
                    except Exception as e:
                        errors.append(f"Configuration '{rule.path}' custom validation error: {str(e)}")
                        
            except KeyError:
                if rule.required:
                    errors.append(f"Required configuration '{rule.path}' is missing")
        
        return errors
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            path: Configuration path (e.g., 'dl_model.batch_size')
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        keys = path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            path: Configuration path
            value: Value to set
        """
        keys = path.split('.')
        current = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Section dictionary
        """
        return self.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return self.config.copy()
    
    def save_to_file(self, file_path: str, format_type: str = "json") -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Output file path
            format_type: File format ('json' or 'yaml')
        """
        file_path = Path(file_path)
        
        if format_type.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, default=str)
        elif format_type.lower() in ["yaml", "yml"]:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Configuration saved to {file_path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration sources and validation status."""
        validation_errors = self.validate()
        
        return {
            "sources": [
                {
                    "type": source["type"].value,
                    "name": source["name"],
                    "priority": source["priority"]
                }
                for source in self.config_sources
            ],
            "validation_rules": len(self.validation_rules),
            "validation_errors": validation_errors,
            "is_valid": len(validation_errors) == 0,
            "total_config_keys": len(self._flatten_dict(self.config))
        }
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def create_emotion_detection_config_manager() -> ConfigurationManager:
    """Create a pre-configured configuration manager for emotion detection."""
    config_manager = ConfigurationManager()
    
    # Define validation rules
    validation_rules = [
        # Image configuration
        ConfigValidationRule("images.height", required=True, type_check=int, min_value=32, max_value=512),
        ConfigValidationRule("images.width", required=True, type_check=int, min_value=32, max_value=512),
        
        # Classes configuration
        ConfigValidationRule("classes.labels", required=True, type_check=list),
        ConfigValidationRule("classes.emojis", required=True, type_check=list),
        
        # Dataset configuration
        ConfigValidationRule("datasets.train_directory", required=True, type_check=str),
        ConfigValidationRule("datasets.test_directory", required=True, type_check=str),
        
        # Model configuration
        ConfigValidationRule("dl_model.batch_size", required=True, type_check=int, min_value=1, max_value=512),
        ConfigValidationRule("dl_model.learn_r", required=True, type_check=(int, float), min_value=0.0001, max_value=1.0),
        ConfigValidationRule("dl_model.fixed_seed", required=True, type_check=int, min_value=0),
        
        # Transfer learning specific
        ConfigValidationRule("dl_model.transfer_learning.epoch", required=True, type_check=int, min_value=1),
        ConfigValidationRule("dl_model.transfer_learning.fine_tuning_epochs", required=True, type_check=int, min_value=1),
        ConfigValidationRule("dl_model.transfer_learning.early_stop_crit", required=True, type_check=int, min_value=1),
        
        # No transfer learning specific
        ConfigValidationRule("dl_model.no_transfer_learning.epoch", required=True, type_check=int, min_value=1),
    ]
    
    config_manager.add_validation_rules(validation_rules)
    
    return config_manager


def main():
    """Example usage of the configuration manager."""
    # Create configuration manager
    config_manager = create_emotion_detection_config_manager()
    
    # Load from JSON file
    config_manager.load_from_file("configuration/config_file.json")
    
    # Load from environment variables
    config_manager.load_from_environment()
    
    # Validate configuration
    errors = config_manager.validate()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
    
    # Get configuration summary
    summary = config_manager.get_config_summary()
    print(f"\nConfiguration Summary:")
    print(f"  Sources: {len(summary['sources'])}")
    print(f"  Validation rules: {summary['validation_rules']}")
    print(f"  Is valid: {summary['is_valid']}")
    print(f"  Total config keys: {summary['total_config_keys']}")
    
    # Example usage
    batch_size = config_manager.get("dl_model.batch_size", 32)
    print(f"\nBatch size: {batch_size}")


if __name__ == "__main__":
    main()
