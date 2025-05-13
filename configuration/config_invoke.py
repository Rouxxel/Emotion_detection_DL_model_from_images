import json
import os

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "config_file.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

json_data = load_config()
