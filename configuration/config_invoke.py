import json
import os
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file with environment variable overrides.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
    
    Returns:
        Dict containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "config_file.json")
        else:
            json_path = config_path
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Override with environment variables if they exist
        config = _apply_env_overrides(config)
        
        return config
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {str(e)}", e.doc, e.pos)
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {str(e)}")

def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.
    
    Environment variables should be prefixed with EMOTION_DETECTION_
    and use double underscores to separate nested keys.
    
    Example: EMOTION_DETECTION_DL_MODEL__BATCH_SIZE=32
    """
    env_prefix = "EMOTION_DETECTION_"
    
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(env_prefix):].lower()
            
            # Split nested keys
            keys = config_key.split('__')
            
            # Navigate to the correct nested dictionary
            current_dict = config
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            
            # Set the value (try to convert to appropriate type)
            final_key = keys[-1]
            current_dict[final_key] = _convert_env_value(value)
    
    return config

def _convert_env_value(value: str) -> Any:
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
    
    # Return as string
    return value

# Load default configuration
json_data = load_config()
