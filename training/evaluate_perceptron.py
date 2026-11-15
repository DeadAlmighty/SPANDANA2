#!/usr/bin/env python3
"""SPANDANA — evaluate the trained Perceptron on the held-out test set.

Loads backend/models/perceptron_pipeline.joblib + perceptron_meta.json
(both produced by train_perceptron.py), scores strat_fold 10 and writes
data/reports/perceptron_test_metrics.json.

The threshold and Platt calibration were frozen on validation — this
script never re-tunes anything.

Usage:
    python training/evaluate_perceptron.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from config import load_config
from metrics import classification_metrics, roc_pr_points


def perceptron_scores(pipe, features: list[str], df: pd.DataFrame,
                      calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated probabilities + margin scores for the feature frame."""
    margin = pipe.decision_function(df[features].values)
    a, b = calibration["a"], calibration["b"]
    prob = 1.0 / (1.0 + np.exp(-(a * margin + b)))
    return prob, margin


def main() -> None:
    cfg = load_config()
    pipe = joblib.load(cfg.models_dir / "perceptron_pipeline.joblib")
    meta = json.loads((cfg.models_dir / "perceptron_meta.json").read_text(encoding="utf-8"))
    fr = pd.read_csv(cfg.feature_rows_path)
    test = fr[fr["split"] == "test"]
    y = (test["label"] == "LVH_positive").astype(int).values

    prob, _ = perceptron_scores(pipe, meta["features"], test, meta["calibration"])
    thr = float(meta["threshold"])
    pred = (prob >= thr).astype(int)

    m = classification_metrics(y, pred, prob)
    print(f"test set: {m['n']} recordings ({m['positives']} LVH-positive)")
    print(f"threshold: {thr:.4f} (frozen from validation)")
    for k in ("accuracy", "precision", "sensitivity", "specificity", "f1",
              "roc_auc", "pr_auc"):
        print(f"  {k:12s}: {m[k]:.4f}")
    print(f"  confusion   : tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")

    out = {"model": "Perceptron", "split": "test (strat_fold 10)",
           "threshold": thr, **m, **roc_pr_points(y, prob)}
    path = cfg.reports_dir / "perceptron_test_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
