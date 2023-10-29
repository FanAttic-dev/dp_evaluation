from pathlib import Path
import yaml


def load_yaml(file_name: str):
    with open(file_name, 'r') as f:
        return yaml.safe_load(f)


def save_to_file(file_path: Path, text: str):
    with open(file_path, 'w') as f:
        f.write(text)
