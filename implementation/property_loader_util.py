from pathlib import Path
import yaml

FILE_BASE = Path(__file__).parent
def get_property_value(key: str, file_name: str) -> str:
    with open(FILE_BASE / file_name, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    if key not in prompts:
        raise KeyError(f"Prompt '{key}' not found")
    return prompts[key].strip()