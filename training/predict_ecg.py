#!/usr/bin/env python3
"""SPANDANA — single-ECG inference with BOTH trained models + criteria.

This module is the ONE place that knows how to turn a decoded 12-lead
waveform into SPANDANA screening outputs:

    Sokolow-Lyon + Cornell voltage criteria   (same convention as the frontend)
    Perceptron probability                    (StandardScaler+Perceptron + Platt)
    CNN probability                           (PyTorch SpandanaCNN, z-scored input)

Everything the backend /api/v1/screen endpoint returns comes from here, so
training and inference share identical preprocessing. All model artefacts
must exist (train them first):

    python training/train_perceptron.py && python training/train_cnn.py

CLI:
    python training/predict_ecg.py data/ptbxl/.../records100/00000/00001_lr
    python training/predict_ecg.py some_upload.csv
    python training/predict_ecg.py recording.dat --hea recording.hea
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# flat imports — works as a script AND when the backend adds this folder to sys.path
try:
    from config import load_config
    from cnn_model import lead_saliency, load_cnn, normalize, to_channels_first
except ImportError:  # package-style import (backend/app)
    from .config import load_config                      # type: ignore
    from .cnn_model import lead_saliency, load_cnn, normalize, to_channels_first  # type: ignore

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
FEAT_NAMES = ["S_V1", "R_V5", "R_V6", "R_aVL", "S_V3"]

_ARTIFACTS: dict[str, Any] | None = None


def load_artifacts(models_dir: Path | None = None) -> dict[str, Any]:
    """Lazy, cached load of every artefact; raises with a CLEAR message
    when something is missing (the backend turns this into HTTP 503)."""
    global _ARTIFACTS
    if _ARTIFACTS is not None and (models_dir is None or _ARTIFACTS["models_dir"] == models_dir):
        return _ARTIFACTS

    cfg = load_config()
    mdir = models_dir or cfg.models_dir

    missing = [n for n in ("preprocessing.json", "perceptron_pipeline.joblib",
                           "perceptron_meta.json", "cnn_best.pt", "cnn_meta.json")
               if not (mdir / n).exists()]
    if missing:
        raise FileNotFoundError(
            "model artefact(s) missing in " + str(mdir) + ": " + ", ".join(missing) +
            " — train the models first (python training/train_perceptron.py, "
            "python training/train_cnn.py, python training/evaluate_cnn.py)")

    import joblib
    import torch

    norm = json.loads((mdir / "preprocessing.json").read_text(encoding="utf-8"))
    p_meta = json.loads((mdir / "perceptron_meta.json").read_text(encoding="utf-8"))
    c_meta = json.loads((mdir / "cnn_meta.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = load_cnn(mdir / "cnn_best.pt", device,
                   channels=tuple(c_meta.get("channels", (32, 64, 96))),
                   dropout=c_meta.get("dropout", 0.3))
    _ARTIFACTS = {
        "models_dir": mdir, "cfg": cfg, "norm": norm,
        "mean": np.array(norm["mean"]), "std": np.array(norm["std"]),
        "pipe": joblib.load(mdir / "perceptron_pipeline.joblib"),
        "p_meta": p_meta, "cnn": cnn, "c_meta": c_meta, "device": device,
    }
    return _ARTIFACTS


def wave_to_array(leads: dict[str, list[float]], fs: float,
                  fs_target: int, seconds: int) -> np.ndarray:
    """leads dict -> (T, 12) float array in LEADS order, resampled if needed."""
    arr = np.zeros((int(round(seconds * fs_target)), len(LEADS)), dtype=np.float64)

    def _norm(n: str) -> str:
        upper = n.strip().upper()
        return _ALIASES.get(upper, upper)

    for j, name in enumerate(LEADS):
        key = next((k for k in leads if _norm(k) == _norm(name)), None)
        if key is None:
            raise ValueError(f"lead '{name}' missing from the waveform")
        v = np.asarray(leads[key], dtype=np.float64)
        if not np.isfinite(v).all():
            raise ValueError(f"lead '{name}' contains NaN/Inf samples")
        if fs != fs_target:  # linear resample (e.g. 500 Hz upload -> 100 Hz)
            t_new = np.linspace(0, 1, arr.shape[0])
            t_old = np.linspace(0, 1, len(v))
            v = np.interp(t_new, t_old, v)
        n = min(len(v), arr.shape[0])
        arr[:n, j] = v[:n]
    if not np.isfinite(arr).all():
        raise ValueError("waveform contains non-finite values")
    return arr


def measure(rec: np.ndarray) -> dict[str, float]:
    """R/S amplitudes in mV — identical convention to extract_features.py."""
    def base(i: int) -> float:
        return float(np.median(rec[:, i]))
    return {
        "sv1": base(6) - float(rec[:, 6].min()),
        "rv5": float(rec[:, 10].max()) - base(10),
        "rv6": float(rec[:, 11].max()) - base(11),
        "ravl": float(rec[:, 4].max()) - base(4),
        "sv3": base(8) - float(rec[:, 8].min()),
    }


def criteria_from(rec: np.ndarray, sex: str | None) -> dict[str, Any]:
    """Sokolow-Lyon / Cornell — same thresholds as src/lib/ecg/criteria.ts."""
    m = measure(rec)
    sl_thr, sl_ref = 3.5, "Sokolow-Lyon: S(V1) + max(R(V5), R(V6)) >= 3.5 mV"
    c_thr = (2.0 if (sex or "male") == "female" else 2.8)
    c_ref = f"Cornell: R(aVL) + S(V3) >= {c_thr} mV ({sex or 'male'})"
    sl = m["sv1"] + max(m["rv5"], m["rv6"])
    co = m["ravl"] + m["sv3"]
    sl_pos, co_pos = sl >= sl_thr, co >= c_thr
    agreement = ("both-positive" if sl_pos and co_pos else
                 "both-negative" if not sl_pos and not co_pos else "discordant")
    return {
        "features": m,
        "sokolowLyon": {"sum": round(sl, 4), "threshold": sl_thr,
                        "positive": bool(sl_pos), "reference": sl_ref},
        "cornell": {"sum": round(co, 4), "threshold": c_thr,
                    "positive": bool(co_pos), "reference": c_ref},
        "agreement": agreement,
    }


def predict_from_leads(leads: dict[str, list[float]], fs: float,
                       sex: str | None = None,
                       models_dir: Path | None = None) -> dict[str, Any]:
    """Full SPANDANA screening for one decoded 12-lead waveform."""
    import torch

    art = load_artifacts(models_dir)
    rec = wave_to_array(leads, fs, art["cfg"].fs, art["cfg"].signal_seconds)

    res: dict[str, Any] = {}
    res["criteria"] = criteria_from(rec, sex)

    # ---- Perceptron ------------------------------------------------------
    t0 = time.perf_counter()
    feats = res["criteria"]["features"]
    x = np.array([[feats["sv1"], feats["rv5"], feats["rv6"],
                   feats["ravl"], feats["sv3"]]], dtype=float)
    margin = float(art["pipe"].decision_function(x)[0])
    a = art["p_meta"]["calibration"]["a"]; b = art["p_meta"]["calibration"]["b"]
    p_perc = float(1.0 / (1.0 + np.exp(-(a * margin + b))))
    thr_p = float(art["p_meta"]["threshold"])
    coefs = art["pipe"].named_steps["clf"].coef_[0]
    zs = art["pipe"].named_steps["scaler"].transform(x)[0]
    perc_contrib = [{"feature": name, "weight": round(float(a * w), 4),
                     "value": round(float(v), 4)}
                    for name, w, v in zip(FEAT_NAMES, coefs, zs)]
    res["perceptron"] = {
        "model": "perceptron",
        "probability": round(p_perc, 4), "positive": bool(p_perc >= thr_p),
        "threshold": thr_p,
        "latencyMs": round((time.perf_counter() - t0) * 1000, 1),
        "simulated": False, "contributions": perc_contrib,
    }

    # ---- CNN -------------------------------------------------------------
    t0 = time.perf_counter()
    rec_z = normalize(rec, art["mean"], art["std"])
    xb = torch.from_numpy(to_channels_first(rec_z).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        p_cnn = float(torch.sigmoid(art["cnn"](xb.to(art["device"]))[0]))
    thr_c = float(art["c_meta"].get("threshold", 0.5))  # evaluate_cnn.py tunes it
    sal = lead_saliency(art["cnn"], rec_z.astype(np.float32), art["device"])
    rms = np.sqrt((rec ** 2).mean(axis=0))
    cnn_contrib = [{"feature": f"Lead {name}", "weight": round(float(s), 4),
                    "value": round(float(r), 4)}
                   for name, s, r in zip(LEADS, sal, rms)]
    res["cnn"] = {
        "model": "cnn",
        "probability": round(p_cnn, 4), "positive": bool(p_cnn >= thr_c),
        "threshold": thr_c,
        "latencyMs": round((time.perf_counter() - t0) * 1000, 1),
        "simulated": False, "contributions": cnn_contrib,
    }

    res["predictions"] = [res["perceptron"], res["cnn"]]
    res["meta"] = {
        "fs_used": art["cfg"].fs, "samples_used": int(rec.shape[0]),
        "input_fs": fs, "unit": "mV", "models_dir": str(art["models_dir"]),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="SPANDANA single-ECG inference")
    ap.add_argument("file", help=".csv/.txt waveform, WFDB record path (no ext), or .dat")
    ap.add_argument("--hea", default=None, help=".hea file when passing a bare .dat")
    ap.add_argument("--sex", default=None, choices=["male", "female"])
    args = ap.parse_args()

    path = Path(args.file)
    if path.suffix.lower() == ".dat" and args.hea:
        import wfdb
        rec_path = str(path.with_suffix(""))
        # wfdb needs the header next to the dat
        rec, _ = wfdb.rdsamp(rec_path)
        leads = {name: rec[:, i] for i, name in enumerate(
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])}
        fs = 100.0
    else:
        try:
            import pandas as pd
            df = pd.read_csv(path, header=None)
            arr = df.select_dtypes(include=[np.number]).values
            if arr.shape[1] == 12:
                leads = {name: arr[:, j] for j, name in enumerate(LEADS)}
            elif arr.shape[0] == 12:  # leads as rows
                leads = {name: arr[j] for j, name in enumerate(LEADS)}
            else:
                raise SystemExit(f"expected 12 numeric columns or rows, got {arr.shape}")
            fs = 100.0
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"could not read {path}: {e}")

    out = predict_from_leads(leads, fs, sex=args.sex)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
