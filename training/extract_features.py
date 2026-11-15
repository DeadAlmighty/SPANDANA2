#!/usr/bin/env python3
"""SPANDANA step 6/6 — extract the Perceptron voltage features.

For every kept recording (row of signals.npy) this measures, exactly as
the frontend does (src/lib/ecg/real-signal.ts):

    baseline  = median of the trace (per lead)
    R         = max - baseline      (positive deflection)
    S         = baseline - min      (negative deflection)

Features (mV): S_V1, R_V5, R_V6, R_aVL, S_V3
Traditional criteria computed from them:
    Sokolow-Lyon : S_V1 + max(R_V5, R_V6) >= 3.5 mV
    Cornell      : R_aVL + S_V3 >= 2.8 mV (male) / 2.0 mV (female)

Outputs:
    data/metadata/feature_rows.csv   (one row per recording)
    data/preprocessed/index.csv      (row_idx -> ecg_id mapping)

Usage:
    python training/extract_features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import load_config

LEAD = {name: i for i, name in enumerate(
    ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])}


def measure(rec: np.ndarray) -> dict[str, float]:
    """rec: (T, 12) mV — returns the five R/S amplitudes in mV."""
    def base(i: int) -> float:
        return float(np.median(rec[:, i]))

    return {
        "sv1": base(LEAD["V1"]) - float(rec[:, LEAD["V1"]].min()),
        "rv5": float(rec[:, LEAD["V5"]].max()) - base(LEAD["V5"]),
        "rv6": float(rec[:, LEAD["V6"]].max()) - base(LEAD["V6"]),
        "ravl": float(rec[:, LEAD["aVL"]].max()) - base(LEAD["aVL"]),
        "sv3": base(LEAD["V3"]) - float(rec[:, LEAD["V3"]].min()),
    }


def main() -> None:
    cfg = load_config()
    X = np.load(cfg.signals_path, mmap_mode="r")
    meta = pd.read_csv(cfg.lvh_metadata_path)
    meta = meta[meta["label"].isin(("LVH_positive", "control"))] \
        .sort_values("ecg_id").reset_index(drop=True)
    assert X.shape[0] == len(meta), (
        f"signals.npy has {X.shape[0]} rows but metadata has {len(meta)} — "
        "run preprocess_ecg.py (build or --check) first."
    )

    print(f"measuring {len(meta)} recordings ({X.shape[1]} samples x {X.shape[2]} leads) ...")
    rows = [measure(np.asarray(X[i])) for i in range(len(meta))]
    f = pd.DataFrame(rows)

    sl_thr = float(cfg.raw["features"]["sokolow_threshold_mV"])
    f["sokolow_lyon_sum"] = f["sv1"] + np.maximum(f["rv5"], f["rv6"])
    f["sokolow_lyon_positive"] = f["sokolow_lyon_sum"] >= sl_thr
    f["cornell_sum"] = f["ravl"] + f["sv3"]
    f["cornell_threshold_mV"] = np.where(meta["sex"] == "male", 2.8, 2.0)
    f["cornell_positive"] = f["cornell_sum"] >= f["cornell_threshold_mV"]

    out = pd.concat([
        meta[["ecg_id", "patient_id", "label", "strat_fold", "age", "sex"]].copy(),
        f,
    ], axis=1)
    out.insert(1, "row_idx", np.arange(len(out)))
    fold_map = {**{fl: "train" for fl in range(1, cfg.val_fold)},
                cfg.val_fold: "val", cfg.test_fold: "test"}
    out.insert(4, "split", out["strat_fold"].map(fold_map))
    out = out[["ecg_id", "row_idx", "patient_id", "label", "split", "strat_fold",
               "age", "sex", "sv1", "rv5", "rv6", "ravl", "sv3",
               "sokolow_lyon_sum", "sokolow_lyon_positive",
               "cornell_sum", "cornell_threshold_mV", "cornell_positive"]]

    out.to_csv(cfg.feature_rows_path, index=False)
    idx = out[["row_idx", "ecg_id", "patient_id", "label", "strat_fold", "split"]]
    idx.to_csv(cfg.index_path, index=False)
    print(f"saved: {cfg.feature_rows_path}  ({len(out)} rows)")
    print(f"saved: {cfg.index_path}")

    pos = out[out["label"] == "LVH_positive"]
    print(f"\nSokolow-Lyon positive: {int(out['sokolow_lyon_positive'].sum())} recordings "
          f"({int(pos['sokolow_lyon_positive'].sum())} of {len(pos)} LVH-positive)")
    print(f"Cornell positive      : {int(out['cornell_positive'].sum())} recordings "
          f"({int(pos['cornell_positive'].sum())} of {len(pos)} LVH-positive)")


if __name__ == "__main__":
    main()
