"""Model/artifact availability reporting for /health and /models/status."""
from __future__ import annotations

import json

from app.config import ARTIFACT_FILES, METRICS_JSON, MODELS_DIR, VERSION


def artifact_status() -> dict:
    """Which artefacts exist on disk (nothing is loaded here)."""
    status: dict[str, dict] = {}
    for model, files in ARTIFACT_FILES.items():
        if model == "shared":
            continue
        present = {f: (MODELS_DIR / f).exists() for f in files}
        status[model] = {
            "available": all(present.values()),
            "files": present,
        }
    metrics_ok = METRICS_JSON.exists()
    return {
        "version": VERSION,
        "models_dir": str(MODELS_DIR),
        "models": status,
        "metrics_available": metrics_ok,
        "metrics_version": _metrics_generated(),
    }


def _metrics_generated() -> str | None:
    if not METRICS_JSON.exists():
        return None
    try:
        data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        return data.get("generated")
    except Exception:  # noqa: BLE001
        return None


def available_model_ids() -> list[str]:
    st = artifact_status()
    return [m for m, info in st["models"].items() if info["available"]]
