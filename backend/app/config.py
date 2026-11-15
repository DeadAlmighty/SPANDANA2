"""SPANDANA backend configuration.

Reads the SAME central config.json used by the training pipeline and adds
environment-variable overrides. No path is hard-coded in the endpoints.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# backend/app/config.py -> parents: [app, backend, PROJECT_ROOT]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    RAW = json.load(f)

API_HOST = os.environ.get("SPANDANA_API_HOST", RAW["api"]["host"])
API_PORT = int(os.environ.get("SPANDANA_API_PORT", RAW["api"]["port"]))

_env_models = os.environ.get("SPANDANA_MODELS_DIR", "")
MODELS_DIR = (Path(_env_models).expanduser().resolve() if _env_models
              else (PROJECT_ROOT / RAW["artifacts"]["models_dir"]).resolve())

METRICS_JSON = MODELS_DIR / "metrics.json"
VERSION = "1.0.0"

_origins = os.environ.get(
    "SPANDANA_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
CORS_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]

TRAINING_DIR = PROJECT_ROOT / "training"

ARTIFACT_FILES = {
    "perceptron": ["perceptron_pipeline.joblib", "perceptron_meta.json"],
    "cnn": ["cnn_best.pt", "cnn_meta.json"],
    "shared": ["preprocessing.json"],
}
