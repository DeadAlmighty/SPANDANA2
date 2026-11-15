"""1D CNN architecture + checkpoint helpers for SPANDANA (PyTorch).

Input  : (B, 12, 1000) — 12 leads x 10 s x 100 Hz, per-lead z-scored
         (stats from backend/models/preprocessing.json, fitted on train only)
Output : single logit (B,) — apply sigmoid for an LVH probability

Layout note: data/preprocessed/signals.npy stores rows as (T, C); the
DataLoader / predict helpers transpose to (C, T) here so the convention is
defined in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SpandanaCNN(nn.Module):
    """Three conv blocks (Conv1d -> BatchNorm -> ReLU -> MaxPool) + dense head."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 96),
                 dropout: float = 0.3, in_leads: int = 12):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv1d(in_leads, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                   # 1000 -> 500
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                   # 500 -> 250
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),                                   # 250 -> 62
            nn.AdaptiveMaxPool1d(8),                           # -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def load_cnn(path: Path, device: torch.device,
             channels: tuple[int, ...] = (32, 64, 96),
             dropout: float = 0.3) -> SpandanaCNN:
    """Build the model and load a trained state dict (always eval-mode)."""
    model = SpandanaCNN(channels=channels, dropout=dropout)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def to_channels_first(x: np.ndarray) -> np.ndarray:
    """(N, T, C) or (T, C) -> (N, C, T) / (C, T) for Conv1d."""
    if x.ndim == 2:
        return np.ascontiguousarray(x.T)
    return np.ascontiguousarray(x.transpose(0, 2, 1))


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-lead z-score with PRE-FITTED stats only (never refit here)."""
    return (x - mean) / std


def lead_saliency(model: SpandanaCNN, x: np.ndarray,
                  device: torch.device) -> np.ndarray:
    """Real gradient saliency: |d logit / d input| summed per lead.

    x: (T, 12) or (12, T) z-scored single recording. Returns (12,)
    non-negative values, normalised so the strongest lead = 1.0.
    """
    in_ch = model.features[0].in_channels
    if x.ndim == 2 and x.shape[0] != in_ch:      # (T, C) -> (C, T)
        x = np.ascontiguousarray(x.T)
    t = torch.tensor(np.asarray(x, dtype=np.float32), device=device,
                     requires_grad=True)
    logit = model(t.unsqueeze(0))[0]
    logit.backward()
    grad = t.grad.detach().abs().sum(dim=1).cpu().numpy()   # (12,)
    g = grad / max(float(grad.max()), 1e-12)
    return g
