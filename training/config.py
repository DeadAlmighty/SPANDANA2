"""Central configuration loader for the SPANDANA training pipeline.

All dataset / artifact locations, sampling settings and training
hyper-parameters live in ONE file: ``config.json`` at the project root.
Edit that file (or set environment overrides) — never edit the scripts.

Environment overrides (optional):
    SPANDANA_PROJECT_ROOT   where the project lives (default: auto-detect)
    SPANDANA_RAW_DIR        PTB-XL data directory
    SPANDANA_MODELS_DIR     trained model artifact directory

Usage inside scripts:

    from config import load_config
    cfg = load_config()
    print(cfg.raw_dir)          # resolved absolute Path
    print(cfg.fs)               # 100
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root = parent of the training/ folder (works no matter where the
# script is launched from, including `python training/train_cnn.py`).
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("SPANDANA_PROJECT_ROOT", str(_HERE.parent))
).resolve()

CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass
class Config:
    """Typed view over config.json with every path resolved to absolute."""

    raw: dict[str, Any]

    # ---- dataset ------------------------------------------------------
    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def fs(self) -> int:
        return int(self.dataset["fs"])

    @property
    def signal_seconds(self) -> int:
        return int(self.dataset["signal_seconds"])

    @property
    def signal_len(self) -> int:
        return self.fs * self.signal_seconds

    @property
    def n_leads(self) -> int:
        return int(self.dataset["n_leads"])

    @property
    def lead_order(self) -> list[str]:
        return list(self.dataset["lead_order"])

    @property
    def raw_dir(self) -> Path:
        """PTB-XL root. SPANDANA_RAW_DIR overrides config.json."""
        base = Path(os.environ.get("SPANDANA_RAW_DIR", "") or self.dataset["raw_dir"])
        return self._resolve(base)

    def find_data_dir(self) -> Path | None:
        """Locate the extracted PTB-XL folder (holds ptbxl_database.csv)."""
        candidates = [self.raw_dir] + sorted(
            p for p in self.raw_dir.glob("*") if p.is_dir()
        )
        for c in candidates:
            if (c / "ptbxl_database.csv").exists():
                return c
        return None

    # ---- labels -------------------------------------------------------
    @property
    def lvh_threshold(self) -> int:
        return int(self.raw["labels"]["lvh_likelihood_threshold"])

    @property
    def val_fold(self) -> int:
        return int(self.raw["labels"]["val_fold"])

    @property
    def test_fold(self) -> int:
        return int(self.raw["labels"]["test_fold"])

    # ---- directories / files ------------------------------------------
    @property
    def preprocessed_dir(self) -> Path:
        return self._resolve(self.raw["preprocessing"]["out_dir"])

    @property
    def signals_path(self) -> Path:
        return self.preprocessed_dir / "signals.npy"

    @property
    def index_path(self) -> Path:
        return self.preprocessed_dir / "index.csv"

    @property
    def norm_path(self) -> Path:
        return self._resolve(self.raw["artifacts"]["models_dir"]) / "preprocessing.json"

    @property
    def metadata_dir(self) -> Path:
        return self._resolve(self.raw["artifacts"]["metadata_dir"])

    @property
    def lvh_metadata_path(self) -> Path:
        return self.metadata_dir / "spandana_lvh_metadata.csv"

    @property
    def splits_path(self) -> Path:
        return self.metadata_dir / "spandana_split_assignments.csv"

    @property
    def patient_summary_path(self) -> Path:
        return self.metadata_dir / "spandana_patient_summary.csv"

    @property
    def feature_rows_path(self) -> Path:
        return self._resolve(self.raw["features"]["out_path"])

    @property
    def models_dir(self) -> Path:
        env = os.environ.get("SPANDANA_MODELS_DIR", "")
        if env:
            return Path(env).expanduser().resolve()
        return self._resolve(self.raw["artifacts"]["models_dir"])

    @property
    def metrics_json(self) -> Path:
        return self._resolve(self.raw["artifacts"]["metrics_json"])

    @property
    def reports_dir(self) -> Path:
        return self._resolve(self.raw["artifacts"]["reports_dir"])

    # ---- model hyper-parameters ---------------------------------------
    @property
    def perceptron(self) -> dict[str, Any]:
        return self.raw["perceptron"]

    @property
    def cnn(self) -> dict[str, Any]:
        return self.raw["cnn"]

    # ---- helpers -------------------------------------------------------
    def _resolve(self, p: Path | str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.preprocessed_dir,
            self.metadata_dir,
            self.reports_dir,
            self.models_dir,
            self._resolve(self.cnn["run_dir"]),
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Central configuration not found: {CONFIG_PATH}\n"
            "Run this script from the SPANDANA project (config.json must sit "
            "at the project root) or set SPANDANA_PROJECT_ROOT."
        )
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    cfg = Config(raw)
    cfg.ensure_dirs()
    return cfg
