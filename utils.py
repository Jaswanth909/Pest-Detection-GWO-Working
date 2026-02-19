import yaml
from pathlib import Path

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_int(x):
    return int(round(x))

def clip(x, low, high):
    return max(low, min(high, x))

def make_dirs():
    Path("results").mkdir(exist_ok=True, parents=True)
