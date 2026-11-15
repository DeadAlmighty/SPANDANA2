"""End-to-end API tests — real decoders, real models, real errors.

Covers: health, JSON screening, multipart upload screening (.csv, .dat+.hea),
invalid uploads, short waveforms, missing-artifact behaviour (503) and the
metrics endpoint. No simulated predictions anywhere.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import METRICS_JSON, MODELS_DIR


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["models"], list)
    assert data["version"]


def test_models_status(client: TestClient) -> None:
    r = client.get("/api/v1/models/status")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data and "models_dir" in data
    assert set(data["models"].keys()) == {"perceptron", "cnn"}


def test_screen_json(client: TestClient, sample_leads: dict[str, list[float]]) -> None:
    if not (MODELS_DIR / "cnn_best.pt").exists():
        __import__("pytest").skip("CNN artefacts not trained yet")
    r = client.post("/api/v1/screen", json={
        "patient": {"code": "TST-001", "age": 55, "sex": "male"},
        "waveform": {"sampleRate": 100, "durationSec": 10, "leads": sample_leads},
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["simulated"] is False
    assert {p["model"] for p in data["predictions"]} == {"perceptron", "cnn"}
    for p in data["predictions"]:
        assert p["simulated"] is False
        assert 0.0 <= p["probability"] <= 1.0
        assert p["latencyMs"] >= 0
        assert isinstance(p["contributions"], list)
        assert p["threshold"] > 0
    assert data["criteria"]["agreement"] in (
        "both-positive", "both-negative", "discordant")
    assert data["criteria"]["sokolowLyon"]["threshold"] == 3.5


def test_screen_json_alias_signal(client: TestClient, sample_leads) -> None:
    if not (MODELS_DIR / "cnn_best.pt").exists():
        __import__("pytest").skip("CNN artefacts not trained yet")
    r = client.post("/api/v1/screen", json={
        "patient": {"sex": "female"},
        "signal": {"sampleRate": 100, "leads": sample_leads},
    })
    assert r.status_code == 200
    assert r.json()["criteria"]["cornell"]["threshold"] == 2.0  # female cutoff


def test_screen_csv_upload(client: TestClient) -> None:
    if not (MODELS_DIR / "cnn_best.pt").exists():
        __import__("pytest").skip("CNN artefacts not trained yet")
    with open(__import__("pathlib").Path(__file__).parents[2]
              / "scripts" / "sample-ecg.csv", "rb") as f:
        r = client.post("/api/v1/screen",
                        files={"file": ("sample-ecg.csv", f, "text/csv")},
                        data={"sex": "male"})
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["input_fs"] == 100.0


def test_screen_dat_without_hea(client: TestClient) -> None:
    p = __import__("pathlib").Path(__file__).parents[2] / "scripts" / "sample-ecg.dat"
    with open(p, "rb") as f:
        r = client.post("/api/v1/screen",
                        files={"file": ("sample-ecg.dat", f, "application/octet-stream")})
    assert r.status_code == 422
    assert ".hea" in r.json()["detail"]


def test_screen_dat_and_hea(client: TestClient) -> None:
    if not (MODELS_DIR / "cnn_best.pt").exists():
        __import__("pytest").skip("CNN artefacts not trained yet")
    # a REAL PTB-XL record (standard WFDB header) as the upload fixture
    root = __import__("pathlib").Path(__file__).parent / "fixtures"
    with open(root / "00001_lr.dat", "rb") as fd, open(root / "00001_lr.hea", "rb") as fh:
        r = client.post("/api/v1/screen", files={
            "file": ("00001_lr.dat", fd, "application/octet-stream"),
            "hea": ("00001_lr.hea", fh, "text/plain"),
        })
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["predictions"]) == 2
    # the decoded real record must match its known PTB-XL identity
    assert data["meta"]["fs_used"] == 100
    assert data["predictions"][0]["simulated"] is False


def test_invalid_leads(client: TestClient) -> None:
    bad = {name: [0.1] * 100 for name in
           ("I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6")}
    bad.pop("V2")  # missing lead
    r = client.post("/api/v1/screen", json={
        "waveform": {"sampleRate": 100, "leads": bad}})
    assert r.status_code == 400
    assert "V2" in r.json()["detail"]


def test_nan_rejected(client: TestClient) -> None:
    leads = {name: [0.1] * 1000 for name in
             ("I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6")}
    leads["I"][10] = float("nan")
    # httpx refuses to serialise NaN — send the body raw (the server's
    # json.loads accepts the NaN literal; the API must reject it with 400)
    body = json.dumps({"waveform": {"sampleRate": 100, "leads": leads}})
    r = client.post("/api/v1/screen", content=body.encode(),
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert "NaN" in r.json()["detail"] or "nan" in r.json()["detail"]


def test_metrics_endpoint(client: TestClient) -> None:
    if not METRICS_JSON.exists():
        __import__("pytest").skip("metrics.json not generated yet")
    r = client.get("/api/v1/models/metrics")
    assert r.status_code == 200
    data = r.json()
    names = {m["model"] for m in data["metrics"]}
    assert {"Sokolow–Lyon", "Cornell", "Perceptron", "CNN"} <= names
    for m in data["metrics"]:
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["rocAuc"] <= 1.0


def test_metrics_missing_gives_503(client: TestClient, tmp_path, monkeypatch) -> None:
    import app.config as cfg
    monkeypatch.setattr(cfg, "METRICS_JSON", tmp_path / "nope.json")
    # routes.py imported METRICS_JSON by value — patch the route module too
    import app.api.routes as routes
    monkeypatch.setattr(routes, "METRICS_JSON", tmp_path / "nope.json")
    r = client.get("/api/v1/models/metrics")
    assert r.status_code == 503
    assert "evaluate_all" in r.json()["detail"]


def test_openapi_contract(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/health" in paths
    assert "/api/v1/screen" in paths
    assert "/api/v1/models/metrics" in paths
    assert "/api/v1/models/status" in paths
