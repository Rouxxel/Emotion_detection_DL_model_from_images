import json
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Returns:
        Dict containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "config_file.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {str(e)}")

json_data = load_config()
