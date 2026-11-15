"""Pipeline tests — dataset loading, LVH labels, preprocessing, features,
model loading, inference and missing-artifact behaviour.

These tests run against whatever artifacts exist on disk; anything missing
is skipped with a clear message (fresh machines run the pipeline first).

Run:  pytest training/tests -q       (from the project root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1].parent      # project root
sys.path.insert(0, str(ROOT / "training"))

from config import load_config                          # noqa: E402
from metrics import classification_metrics              # noqa: E402

cfg = load_config()


def _have(path: Path) -> bool:
    return path.exists()


# ---------------------------------------------------------------- metadata --
def test_metadata_tables_present():
    data_dir = cfg.find_data_dir()
    if data_dir is None:
        pytest.skip("PTB-XL not downloaded yet (python training/download_dataset.py)")
    db = pd.read_csv(data_dir / "ptbxl_database.csv")
    assert len(db) == 21799
    assert (data_dir / "scp_statements.csv").exists()


def test_lvh_metadata_and_labels():
    if not _have(cfg.lvh_metadata_path):
        pytest.skip("run python training/extract_lvh.py first")
    meta = pd.read_csv(cfg.lvh_metadata_path)
    assert len(meta) == 21799
    counts = meta["label"].value_counts()
    # the cohort definition is deterministic — exact expected sizes
    assert counts.get("LVH_positive", 0) == 1085
    assert counts.get("LVH_low_certainty", 0) == 1047
    assert counts.get("control", 0) == 19667
    # LVH labels must match the likelihood threshold policy
    pos = meta[meta["label"] == "LVH_positive"]
    assert (pos["lvh_likelihood"] >= cfg.lvh_threshold).all()
    low = meta[meta["label"] == "LVH_low_certainty"]
    assert ((low["lvh_likelihood"] > 0) &
            (low["lvh_likelihood"] < cfg.lvh_threshold)).all()


def test_splits_patient_wise_no_leakage():
    if not _have(cfg.splits_path):
        pytest.skip("run python training/create_splits.py first")
    sp = pd.read_csv(cfg.splits_path)
    assert set(sp["split"].unique()) == {"train", "val", "test"}
    # folds map 1-8 / 9 / 10
    assert (sp.loc[sp["split"] == "val", "strat_fold"] == cfg.val_fold).all()
    assert (sp.loc[sp["split"] == "test", "strat_fold"] == cfg.test_fold).all()
    # PATIENT-LEVEL LEAKAGE CHECK — the core guarantee of the pipeline
    per_patient = sp.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all(), "a patient appears in several splits!"


def test_feature_rows_consistency():
    if not _have(cfg.feature_rows_path):
        pytest.skip("run python training/extract_features.py first")
    fr = pd.read_csv(cfg.feature_rows_path)
    assert len(fr) == 20752
    # Sokolow-Lyon = S_V1 + max(R_V5, R_V6), threshold 3.5 mV
    sl = fr["sv1"] + np.maximum(fr["rv5"], fr["rv6"])
    assert np.allclose(sl, fr["sokolow_lyon_sum"], atol=1e-6)
    assert (fr["sokolow_lyon_positive"] == (sl >= 3.5)).all()
    # Cornell = R_aVL + S_V3 with sex-specific thresholds
    co = fr["ravl"] + fr["sv3"]
    assert np.allclose(co, fr["cornell_sum"], atol=1e-6)
    assert (fr["cornell_positive"] ==
            (co >= fr["cornell_threshold_mV"])).all()
    assert set(fr["split"].unique()) == {"train", "val", "test"}


# ------------------------------------------------------------- preprocessing --
def test_signals_array():
    if not _have(cfg.signals_path):
        pytest.skip("run python training/preprocess_ecg.py first")
    X = np.load(cfg.signals_path, mmap_mode="r")
    assert X.shape == (20752, cfg.signal_len, cfg.n_leads)
    assert X.dtype == np.float32
    fr = pd.read_csv(cfg.feature_rows_path)
    # spot-check: recompute the five amplitudes of row 0 from the array
    rec = np.asarray(X[0])
    base6 = np.median(rec[:, 6])
    assert abs((base6 - rec[:, 6].min()) - fr.iloc[0]["sv1"]) < 1e-3
    rv5 = rec[:, 10].max() - np.median(rec[:, 10])
    assert abs(rv5 - fr.iloc[0]["rv5"]) < 1e-3


def test_normalization_stats_fitted_train_only():
    norm = json_load(cfg.norm_path)
    assert norm["fitted_on"].startswith("train split only")
    assert len(norm["mean"]) == 12 and len(norm["std"]) == 12


def json_load(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ models --
@pytest.fixture(scope="module")
def artifacts():
    try:
        import predict_ecg
        return predict_ecg.load_artifacts()
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_model_artifacts_load(artifacts):
    assert artifacts["pipe"] is not None
    assert artifacts["cnn"] is not None
    assert 0 < artifacts["p_meta"]["threshold"] < 1


def test_single_ecg_inference(sample_leads_fixture):
    import predict_ecg
    try:
        out = predict_ecg.predict_from_leads(sample_leads_fixture, 100.0, sex="male")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    preds = {p["model"]: p for p in out["predictions"]}
    assert set(preds) == {"perceptron", "cnn"}
    for p in preds.values():
        assert 0.0 <= p["probability"] <= 1.0
        assert p["simulated"] is False
        assert p["latencyMs"] >= 0
    # criteria values are physically plausible and internally consistent
    # (API sums are rounded to 4 decimals, hence the tolerance)
    c = out["criteria"]
    sl = c["features"]["sv1"] + max(c["features"]["rv5"], c["features"]["rv6"])
    assert abs(sl - c["sokolowLyon"]["sum"]) < 1e-3
    co = c["features"]["ravl"] + c["features"]["sv3"]
    assert abs(co - c["cornell"]["sum"]) < 1e-3
    assert c["agreement"] in ("both-positive", "both-negative", "discordant")


def test_inference_missing_lead_fails(sample_leads_fixture):
    import predict_ecg
    broken = {k: v for k, v in sample_leads_fixture.items() if k != "V3"}
    with pytest.raises(ValueError):
        predict_ecg.predict_from_leads(broken, 100.0, sex="male")


def test_inference_nan_rejected():
    import predict_ecg
    leads = {name: [0.1] * 1000 for name in
             ("I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6")}
    leads["II"][5] = float("nan")
    with pytest.raises(ValueError):
        predict_ecg.predict_from_leads(leads, 100.0, sex="male")


def test_missing_artifacts_clear_error(tmp_path):
    import importlib
    import predict_ecg
    saved = predict_ecg._ARTIFACTS
    predict_ecg._ARTIFACTS = None                    # clear the cache
    try:
        with pytest.raises(FileNotFoundError) as e:
            predict_ecg.load_artifacts(models_dir=tmp_path)
        assert "missing" in str(e.value)
    finally:
        predict_ecg._ARTIFACTS = saved


def test_metrics_helper():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    score = np.array([0.1, 0.2, 0.6, 0.8, 0.9, 0.3, 0.4, 0.7])
    m = classification_metrics(y, pred, score)
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 2 and m["tn"] == 2
    assert abs(m["sensitivity"] - 3 / 5) < 1e-9
    assert abs(m["specificity"] - 2 / 3) < 1e-9
    assert abs(m["precision"] - 3 / 4) < 1e-9
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1


@pytest.fixture()
def sample_leads_fixture():
    df = pd.read_csv(ROOT / "scripts" / "sample-ecg.csv", header=None)
    arr = df.select_dtypes(include=[np.number]).values
    names = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    return {name: arr[:, j] for j, name in enumerate(names)}
