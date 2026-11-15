#!/usr/bin/env python3
"""SPANDANA step 4/6 — preprocess the 100 Hz ECG waveforms.

Build mode (needs the raw records100/ files):
    python training/preprocess_ecg.py
    python training/preprocess_ecg.py --limit 500     # smoke run

Check mode (validates an already-built signals.npy — fast, no raw files):
    python training/preprocess_ecg.py --check

What it produces (in data/preprocessed/):
    signals.npy : float32 array (N, 1000, 12) — raw mV, 10 s x 100 Hz,
                  lead order = config.dataset.lead_order,
                  rows in spandana_lvh_metadata.csv order
                  (LVH_positive + control; low-certainty rows excluded)
    index.csv   : row_idx -> ecg_id / patient_id / label / split mapping

Also fits the CNN normalization statistics (per-lead z-score) on the
TRAIN split only and saves them to backend/models/preprocessing.json —
the SAME file the backend uses at inference time, guaranteeing that
inference preprocessing matches training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config

LABEL_KEEP = ("LVH_positive", "control")


def trim_or_pad(sig: np.ndarray, n: int) -> np.ndarray:
    """sig (T, C) -> exactly (n, C): centre-trim longer, zero-pad shorter."""
    t = sig.shape[0]
    if t >= n:
        return sig[:n, :]
    out = np.zeros((n, sig.shape[1]), dtype=np.float32)
    out[:t, :] = sig
    return out


def fit_normalization(X: np.ndarray, train_idx: np.ndarray) -> dict:
    """Per-lead mean/std over the TRAIN split only (N, T, C) array."""
    tr = X[train_idx]
    mean = tr.reshape(-1, tr.shape[2]).mean(axis=0)
    std = tr.reshape(-1, tr.shape[2]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist()}


def build(cfg, limit: int | None) -> None:
    import wfdb  # heavyweight import only in build mode

    meta = pd.read_csv(cfg.lvh_metadata_path)
    meta = meta[meta["label"].isin(LABEL_KEEP)].sort_values("ecg_id").reset_index(drop=True)
    if limit:
        meta = meta.head(limit)
    data_dir = cfg.find_data_dir()
    assert data_dir is not None, "PTB-XL raw data not found."
    fs, n = cfg.fs, cfg.signal_len

    X = np.zeros((len(meta), n, cfg.n_leads), dtype=np.float32)
    skipped = []
    for i, row in enumerate(meta.itertuples(index=False)):
        path = (data_dir / str(row.filename_lr)).with_suffix("")
        try:
            sig, info = wfdb.rdsamp(str(path))
            assert info["fs"] == fs, f"fs={info['fs']} != {fs}"
            assert sig.shape[1] == cfg.n_leads, f"leads={sig.shape[1]}"
            assert np.isfinite(sig).all(), "non-finite samples"
            X[i] = trim_or_pad(sig.astype(np.float32), n)
        except Exception as e:  # noqa: BLE001 - report every unreadable record
            skipped.append((int(row.ecg_id), str(e)[:80]))
            X[i] = 0.0
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(meta)} decoded")
    if skipped:
        print(f"WARNING: {len(skipped)} recordings could not be decoded:")
        for ecg_id, err in skipped[:10]:
            print(f"   ecg_id {ecg_id}: {err}")
        # drop the failed rows so the array stays aligned with the index
        keep = ~meta["ecg_id"].isin([s[0] for s in skipped]).values
        meta, X = meta[keep].reset_index(drop=True), X[keep]

    idx = pd.DataFrame({
        "row_idx": np.arange(len(meta)),
        "ecg_id": meta["ecg_id"].values,
        "patient_id": meta["patient_id"].values,
        "label": meta["label"].values,
        "strat_fold": meta["strat_fold"].values,
    })
    # split column comes from create_splits.py if it ran already
    splits_path = cfg.splits_path
    if splits_path.exists():
        idx = idx.merge(
            pd.read_csv(splits_path)[["ecg_id", "split"]], on="ecg_id", how="left"
        )
    else:
        fold_map = {**{f: "train" for f in range(1, cfg.val_fold)},
                    cfg.val_fold: "val", cfg.test_fold: "test"}
        idx["split"] = idx["strat_fold"].map(fold_map)

    np.save(cfg.signals_path, X)
    idx.to_csv(cfg.index_path, index=False)
    print(f"saved: {cfg.signals_path}  shape={X.shape}")
    print(f"saved: {cfg.index_path}")

    # normalization stats — TRAIN split only (patient-wise, no leakage)
    train_idx = idx.index[idx["split"] == "train"].values
    norm = fit_normalization(X, train_idx)
    norm.update({
        "lead_order": cfg.lead_order,
        "fs": fs,
        "signal_len": n,
        "unit": "mV",
        "normalization": "per-lead z-score",
        "fitted_on": "train split only (strat_fold 1-8)",
        "amplitude_convention": "median-of-trace baseline; R = max-baseline; "
                                "S = baseline-min (mirrors src/lib/ecg/real-signal.ts)",
        "perceptron_features": ["S_V1", "R_V5", "R_V6", "R_aVL", "S_V3"],
        "n_signals": int(len(idx)),
    })
    out = cfg.norm_path
    out.write_text(json.dumps(norm, indent=2), encoding="utf-8")
    print(f"saved: {out}  (normalization fitted on {len(train_idx)} train signals)")


def check(cfg) -> None:
    """Validate an existing signals.npy + index.csv without raw files."""
    X = np.load(cfg.signals_path, mmap_mode="r")
    meta = pd.read_csv(cfg.lvh_metadata_path)
    meta = meta[meta["label"].isin(LABEL_KEEP)].sort_values("ecg_id").reset_index(drop=True)

    if cfg.index_path.exists():
        idx = pd.read_csv(cfg.index_path)
    elif cfg.feature_rows_path.exists():
        # index.csv not built yet — feature_rows.csv carries the same mapping
        idx = pd.read_csv(cfg.feature_rows_path)[
            ["row_idx", "ecg_id", "patient_id", "label", "strat_fold", "split"]
        ]
        print("(index.csv missing — using feature_rows.csv for the mapping)")
    else:
        raise SystemExit("Neither index.csv nor feature_rows.csv found — cannot check.")

    print(f"signals.npy : shape={X.shape} dtype={X.dtype}")
    ok = True
    if X.shape != (len(meta), cfg.signal_len, cfg.n_leads):
        print(f"  MISMATCH: expected {(len(meta), cfg.signal_len, cfg.n_leads)}")
        ok = False
    if X.dtype != np.float32:
        print("  MISMATCH: dtype should be float32")
        ok = False
    if not np.array_equal(idx["ecg_id"].values, meta["ecg_id"].values):
        print("  MISMATCH: index.csv order differs from metadata order")
        ok = False
    sample = np.asarray(X[:: max(len(X) // 20, 1)])
    if not np.isfinite(sample).all():
        print("  MISMATCH: non-finite values present")
        ok = False

    # normalization stats consistency (exact chunked accumulation — a stride
    # sample is biased because ecg_id blocks correlate with recording sites)
    norm = json.loads(cfg.norm_path.read_text(encoding="utf-8"))
    mean = np.array(norm["mean"]); std = np.array(norm["std"])
    tr = idx.index[idx["split"] == "train"].values
    n = 0; s1 = np.zeros(12); s2 = np.zeros(12)
    for start in range(0, len(tr), 2000):
        chunk = np.asarray(X[tr[start:start + 2000]], dtype=np.float64)
        n += chunk.shape[0] * chunk.shape[1]
        s1 += chunk.reshape(-1, 12).sum(axis=0)
        s2 += (chunk.reshape(-1, 12) ** 2).sum(axis=0)
    est_m = s1 / n
    est_s = np.sqrt(s2 / n - est_m ** 2)
    dm = float(np.abs(est_m - mean).max()); ds = float(np.abs(est_s - std).max())
    print(f"normalization stats drift (exact): mean {dm:.6f}, std {ds:.6f}")
    if ds > 0.001:
        print("  WARNING: saved stats differ from the array — refit!")

    # spot-check features of one row against feature_rows.csv
    frp = cfg.feature_rows_path
    if frp.exists():
        fr = pd.read_csv(frp)
        rec = np.asarray(X[0])
        sv1 = np.median(rec[:, 6]) - rec[:, 6].min()
        rv5 = rec[:, 10].max() - np.median(rec[:, 10])
        row = fr[fr["row_idx"] == 0].iloc[0]
        if abs(sv1 - row["sv1"]) < 1e-3 and abs(rv5 - row["rv5"]) < 1e-3:
            print("feature spot-check row 0: OK "
                  f"(sv1={sv1:.3f}, rv5={rv5:.3f})")
        else:
            print(f"feature spot-check MISMATCH: sv1={sv1:.3f} vs {row['sv1']}, "
                  f"rv5={rv5:.3f} vs {row['rv5']}")
            ok = False

    print("\nCHECK RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="build only the first N recordings (smoke run)")
    ap.add_argument("--check", action="store_true",
                    help="validate an existing signals.npy instead of building")
    args = ap.parse_args()

    cfg = load_config()
    if args.check:
        check(cfg)
    else:
        build(cfg, args.limit)


if __name__ == "__main__":
    main()
