#!/usr/bin/env python3
"""SPANDANA — evaluate the trained CNN on validation and test sets.

Loads backend/models/cnn_best.pt, scores strat_fold 9 (validation) and 10
(test), tunes the decision threshold on VALIDATION ONLY (max F1, frozen
afterwards) and writes:

    data/reports/cnn_test_metrics.json
    backend/models/cnn_meta.json  (updated with the tuned threshold)

Usage:
    python training/evaluate_cnn.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from cnn_model import load_cnn, normalize, to_channels_first
from config import load_config
from metrics import classification_metrics, roc_pr_points


@torch.no_grad()
def score_split(model, X, rows, mean, std, device, batch_size=128) -> np.ndarray:
    out = []
    for s in range(0, len(rows), batch_size):
        rec = np.asarray(X[rows[s:s + batch_size]])
        rec = normalize(rec, mean, std)
        xb = torch.from_numpy(to_channels_first(rec)).float().to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def best_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    cand = np.unique(np.round(prob, 4))
    best_f1, best_t = -1.0, 0.5
    for t in cand:
        f1 = f1_score(y, prob >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main() -> None:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = cfg.cnn
    model = load_cnn(cfg.models_dir / "cnn_best.pt", device,
                     channels=tuple(c["channels"]), dropout=c["dropout"])
    norm = json.loads(cfg.norm_path.read_text(encoding="utf-8"))
    mean, std = np.array(norm["mean"]), np.array(norm["std"])
    X = np.load(cfg.signals_path, mmap_mode="r")
    idx = pd.read_csv(cfg.index_path)

    val = idx[idx["split"] == "val"]
    y_va = (val["label"] == "LVH_positive").astype(int).values
    p_va = score_split(model, X, val["row_idx"].values, mean, std, device)
    thr, f1_va = best_threshold(y_va, p_va)
    print(f"validation: {len(val)} recordings | tuned threshold {thr:.4f} "
          f"(F1 {f1_va:.3f}, AP {average_precision_score(y_va, p_va):.4f}, "
          f"AUROC {roc_auc_score(y_va, p_va):.4f})")

    test = idx[idx["split"] == "test"]
    y_te = (test["label"] == "LVH_positive").astype(int).values
    p_te = score_split(model, X, test["row_idx"].values, mean, std, device)
    pred = (p_te >= thr).astype(int)
    m = classification_metrics(y_te, pred, p_te)
    print(f"test: {m['n']} recordings ({m['positives']} LVH-positive)")
    for k in ("accuracy", "precision", "sensitivity", "specificity", "f1",
              "roc_auc", "pr_auc"):
        print(f"  {k:12s}: {m[k]:.4f}")
    print(f"  confusion   : tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")

    out = {"model": "CNN", "split": "test (strat_fold 10)",
           "threshold": thr, **m, **roc_pr_points(y_te, p_te)}
    path = cfg.reports_dir / "cnn_test_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {path}")

    meta_path = cfg.models_dir / "cnn_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({
        "threshold": thr,
        "threshold_tuned_on": "validation fold 9 (max F1)",
        "val_auprc": float(average_precision_score(y_va, p_va)),
        "val_auroc": float(roc_auc_score(y_va, p_va)),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"updated: {meta_path}")


if __name__ == "__main__":
    main()
