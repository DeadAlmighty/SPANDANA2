#!/usr/bin/env python3
"""SPANDANA step 5/6 — patient-wise train / validation / test splits.

Uses the OFFICIAL PTB-XL strat_fold column (already patient-wise):
    folds 1-8 -> train, fold 9 -> validation, fold 10 -> test
(config.json → labels.val_fold / labels.test_fold).

The script PROVES there is no patient-level leakage: every patient must
appear in exactly one split. Outputs:

    data/metadata/spandana_split_assignments.csv  (per recording)
    data/metadata/spandana_patient_summary.csv    (per patient)

Usage:
    python training/create_splits.py
"""
from __future__ import annotations

import pandas as pd

from config import load_config


def main() -> None:
    cfg = load_config()
    meta = pd.read_csv(cfg.lvh_metadata_path)
    val_f, test_f = cfg.val_fold, cfg.test_fold

    fold_map = {**{f: "train" for f in range(1, val_f)},
                val_f: "val", test_f: "test"}
    meta["split"] = meta["strat_fold"].map(fold_map)
    assert meta["split"].notna().all(), "unexpected strat_fold values"

    # ---- leakage proof ---------------------------------------------------
    patients_per_split = meta.groupby("patient_id")["split"].nunique()
    leaked = int((patients_per_split > 1).sum())
    print(f"patients appearing in more than one split: {leaked}")
    assert leaked == 0, "PATIENT-LEVEL LEAKAGE DETECTED — aborting."

    sizes = meta["split"].value_counts()
    pos = meta[meta["label"] == "LVH_positive"]
    print("\nsplit sizes (recordings):")
    for s in ("train", "val", "test"):
        p = int((pos["split"] == s).sum())
        n = int(sizes.get(s, 0))
        print(f"  {s:5s}: {n:6d} recordings, {p:4d} LVH_positive "
              f"({p / max(n, 1) * 100:.2f} %)")

    # patient-level integrity: is each patient's every recording in one fold?
    per_patient_folds = meta.groupby("patient_id")["strat_fold"].nunique()
    print(f"\npatients spanning several folds (within one split, allowed): "
          f"{int((per_patient_folds > 1).sum())}")

    # ---- persist ----------------------------------------------------------
    out = meta[["ecg_id", "patient_id", "strat_fold", "split", "label",
                "lvh_likelihood"]].copy()
    out.to_csv(cfg.splits_path, index=False)
    print(f"\nsaved: {cfg.splits_path}")

    # one row per patient (basis for patient-wise reporting)
    pat = (meta.groupby("patient_id")
           .agg(n_records=("ecg_id", "size"),
                n_lvh=("label", lambda s: int((s == "LVH_positive").sum())),
                split=("split", "first"),
                strat_fold=("strat_fold", "first"),
                age=("age", "first"),
                sex=("sex", "first"))
           .reset_index())
    pat.to_csv(cfg.patient_summary_path, index=False)
    print(f"saved: {cfg.patient_summary_path}  ({len(pat)} patients)")


if __name__ == "__main__":
    main()
