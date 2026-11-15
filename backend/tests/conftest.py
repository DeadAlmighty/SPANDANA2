"""Shared pytest fixtures for the SPANDANA backend tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]      # project root
sys.path.insert(0, str(ROOT / "backend"))

SAMPLE_CSV = ROOT / "scripts" / "sample-ecg.csv"
SAMPLE_DAT = ROOT / "scripts" / "sample-ecg.dat"
SAMPLE_HEA = ROOT / "scripts" / "sample-ecg.hea"


@pytest.fixture(scope="session")
def sample_leads() -> dict[str, list[float]]:
    """The sample 12-lead recording as a leads dict (frontend-style)."""
    import pandas as pd

    df = pd.read_csv(SAMPLE_CSV, header=None)
    arr = df.select_dtypes(include=[np.number]).values
    assert arr.shape[1] == 12
    leads = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    return {name: [round(float(v), 6) for v in arr[:, j]]
            for j, name in enumerate(leads)}


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c
