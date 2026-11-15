#!/usr/bin/env python3
"""SPANDANA — train the Perceptron on PTB-XL voltage features (Scikit-learn).

Input : data/metadata/feature_rows.csv  (from extract_features.py)
Output: backend/models/perceptron_pipeline.joblib   (StandardScaler + Perceptron)
        backend/models/perceptron_meta.json         (calibration, threshold, tuning log)

Method
------
1. Features (mV): S_V1, R_V5, R_V6, R_aVL, S_V3.
2. StandardScaler fitted on the TRAIN split only.
3. Hyper-parameter grid (penalty x alpha) — every candidate is trained on
   folds 1-8 and scored on fold 9 (validation) by average precision.
   The official patient-wise folds are used; no CV shuffling, no test touch.
4. class_weight='balanced' counters the 1:18 class imbalance.
5. The perceptron outputs a margin, not a probability — a Platt sigmoid is
   fitted on the VALIDATION margins to map score -> probability.
6. The decision threshold is tuned on validation (max F1), then frozen for
   the test set and for backend inference.

Usage:
    python training/train_perceptron.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import load_config

FEATURES = ["sv1", "rv5", "rv6", "ravl", "sv3"]


def best_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the given (validation) data."""
    cand = np.unique(np.round(prob, 4))
    best_f1, best_t = -1.0, 0.5
    for t in cand:
        f1 = f1_score(y, prob >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main() -> None:
    cfg = load_config()
    fr = pd.read_csv(cfg.feature_rows_path)
    x = fr[FEATURES].values
    y = (fr["label"] == "LVH_positive").astype(int).values
    is_tr = (fr["split"] == "train").values
    is_va = (fr["split"] == "val").values

    print(f"train {is_tr.sum()} | val {is_va.sum()} | "
          f"LVH prevalence train {y[is_tr].mean() * 100:.2f}%")

    # ---- hyper-parameter tuning on the patient-wise validation fold ----
    log: list[dict] = []
    best = (-1.0, None)
    for penalty in cfg.perceptron["penalties"]:
        for alpha in cfg.perceptron["alphas"]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", Perceptron(
                    penalty=penalty, alpha=alpha,
                    class_weight="balanced",
                    max_iter=cfg.perceptron["max_iter"],
                    random_state=cfg.perceptron["random_state"],
                    tol=1e-3, shuffle=True,
                )),
            ])
            pipe.fit(x[is_tr], y[is_tr])
            ap = average_precision_score(y[is_va], pipe.decision_function(x[is_va]))
            log.append({"penalty": str(penalty), "alpha": alpha, "val_ap": round(float(ap), 5)})
            if ap > best[0]:
                best = (float(ap), pipe)
            print(f"  penalty={str(penalty):10s} alpha={alpha:<7g} val AP={ap:.4f}")
    pipe = best[1]
    print(f"\nbest: val AP={best[0]:.4f} "
          f"({pipe.named_steps['clf'].penalty}, "
          f"alpha={pipe.named_steps['clf'].alpha})")

    # ---- Platt calibration on validation margins -----------------------
    margin_val = pipe.decision_function(x[is_va])
    platt = LogisticRegression(max_iter=1000)
    platt.fit(margin_val.reshape(-1, 1), y[is_va])
    a = float(platt.coef_[0][0]); b = float(platt.intercept_[0])

    def probability(margin: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(a * margin + b)))

    prob_val = probability(margin_val)
    thr, f1_val = best_threshold(y[is_va], prob_val)
    auc_val = roc_auc_score(y[is_va], prob_val)
    print(f"Platt: prob = sigmoid({a:.4f} * margin + {b:.4f})")
    print(f"validation: AUROC={auc_val:.4f}  AP={best[0]:.4f}  "
          f"tuned threshold={thr:.4f} (F1={f1_val:.3f})")

    # ---- persist --------------------------------------------------------
    joblib.dump(pipe, cfg.models_dir / "perceptron_pipeline.joblib")
    meta = {
        "features": FEATURES,
        "best_params": {"penalty": str(pipe.named_steps["clf"].penalty),
                        "alpha": pipe.named_steps["clf"].alpha},
        "val_average_precision": best[0],
        "val_auroc": float(auc_val),
        "calibration": {"type": "platt", "a": a, "b": b},
        "threshold": thr,
        "threshold_tuned_on": "validation fold 9 (max F1)",
        "tuning_log": log,
        "label_semantics": "1 = LVH_positive (SCP 'LVH' likelihood >= 100), "
                           "0 = control (no LVH statement; NOT necessarily healthy)",
    }
    (cfg.models_dir / "perceptron_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved: {cfg.models_dir / 'perceptron_pipeline.joblib'}")
    print(f"saved: {cfg.models_dir / 'perceptron_meta.json'}")


if __name__ == "__main__":
    main()
