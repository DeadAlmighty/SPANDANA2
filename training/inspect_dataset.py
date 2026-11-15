#!/usr/bin/env python3
"""SPANDANA step 2/6 — inspect the PTB-XL dataset structure.

Loads the two official metadata tables, prints the structure (rows,
columns, missing values, folds, sites, sex/age) and verifies the 100 Hz
recording conventions. Results are also written to
``data/reports/dataset_inspection.csv``.

Usage:
    python training/inspect_dataset.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from config import load_config


def main() -> None:
    cfg = load_config()
    data_dir = cfg.find_data_dir()
    assert data_dir is not None, (
        "PTB-XL not found — run `python training/download_dataset.py` first."
    )
    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)

    print("=" * 70)
    print("PTB-XL v" + str(cfg.dataset["version"]) + " — dataset inspection")
    print("=" * 70)
    print(f"data directory      : {data_dir}")
    print(f"recording table     : {db.shape[0]} rows x {db.shape[1]} cols")
    print(f"statement dictionary: {scp.shape[0]} rows x {scp.shape[1]} cols")

    # ---- 100 Hz conventions ------------------------------------------
    lr = db["filename_lr"]
    assert lr.str.contains("records100").all(), "filename_lr must point at records100/"
    assert lr.str.endswith("_lr").all(), "100 Hz filenames end in _lr"
    n_dat = len(list((data_dir / "records100").glob("*/*.dat")))
    print(f"\n100 Hz path convention : OK (records100/<folder>/<id>_lr)")
    print(f"signal files on disk   : {n_dat} / {db.shape[0]} records")
    print(f"sampling rate (config) : {cfg.fs} Hz x {cfg.signal_seconds} s = "
          f"{cfg.signal_len} samples/lead x {cfg.n_leads} leads")

    # ---- folds / demographics ----------------------------------------
    print("\nstrat_fold distribution (official patient-wise split):")
    print(db["strat_fold"].value_counts().sort_index().to_string())
    print(f"\nunique patients        : {db['patient_id'].nunique()}")
    print(f"records per patient    : {db.groupby('patient_id').size().mean():.2f} average")
    print(f"sex                    : {db['sex'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"age (years)            : mean {db['age'].mean():.1f}, "
          f"median {db['age'].median():.0f}")

    # ---- missing values ----------------------------------------------
    miss = db.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print("\nmissing values per column (only columns with gaps):")
    print(miss.to_string())

    # ---- SCP statement dictionary ------------------------------------
    diag = scp[scp["diagnostic"] == 1.0]
    print(f"\nSCP statements total {len(scp)} | diagnostic {len(diag)} "
          f"| form {int(scp['form'].sum())} | rhythm {int(scp['rhythm'].sum())}")
    print("diagnostic classes:", sorted(diag["diagnostic_class"].dropna().unique()))

    # ---- sample scp_codes row ----------------------------------------
    sample = db.iloc[0]["scp_codes"]
    print(f"\nexample scp_codes cell: {sample}")
    print(f"  -> parsed: {ast.literal_eval(sample)}")

    # ---- persist ------------------------------------------------------
    out = cfg.reports_dir / "dataset_inspection.csv"
    miss.rename("n_missing").to_frame().to_csv(out)
    print(f"\nmissing-value table saved to: {out}")


if __name__ == "__main__":
    main()
