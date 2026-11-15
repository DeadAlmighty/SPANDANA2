"""Shared metric helpers for SPANDANA evaluation scripts.

Single definition of every reported metric so the Perceptron, CNN and the
4-method comparison can never disagree on formulas.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_score: np.ndarray) -> dict[str, Any]:
    """All binary-classification metrics for one method.

    y_pred : hard predictions at the method's operating threshold
    y_score: continuous scores for ROC-AUC / PR-AUC
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positives": int(tp + fn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_pr_points(y_true: np.ndarray, y_score: np.ndarray,
                  n_points: int = 200) -> dict[str, list]:
    """Down-sampled curve points for plotting the ROC / PR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pre, rec, _ = precision_recall_curve(y_true, y_score)

    def sample(x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
        if len(x) > n_points:
            idx = np.unique(np.linspace(0, len(x) - 1, n_points).astype(int))
            x, y = x[idx], y[idx]
        return [round(float(v), 4) for v in x], [round(float(v), 4) for v in y]

    fx, fy = sample(fpr, tpr)
    px, py = sample(rec, pre)
    return {"roc": {"fpr": fx, "tpr": fy}, "pr": {"recall": px, "precision": py}}
