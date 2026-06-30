import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "late_delivery_model.joblib"
METADATA_PATH = PROJECT_DIR / "model_metadata.json"
PREDICTIONS_PATH = PROJECT_DIR / "predictions.csv"


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    """Load model metadata saved by train_model.py."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata was not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the trained scikit-learn pipeline once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact was not found: {MODEL_PATH}")

    # The current artifact was trained with a newer sklearn version than the
    # local virtual environment. Keep the service usable while avoiding noisy
    # startup logs; /model-info exposes the training sklearn version.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


def get_model_name() -> str:
    metadata = load_metadata()
    return str(metadata.get("model_type", "late_delivery_model"))


def get_threshold() -> float:
    metadata = load_metadata()
    return float(metadata.get("threshold", 0.5))