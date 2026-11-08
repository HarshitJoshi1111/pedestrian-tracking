"""Read params.yaml, the single place for all settings."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DETECTIONS = ROOT / "data" / "detections"
REPORTS = ROOT / "reports"


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
