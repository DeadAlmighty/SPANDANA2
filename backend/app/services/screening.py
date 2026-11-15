"""Screening service — thin bridge to the training/inference code.

The single source of truth for real inference is
``training/predict_ecg.py`` (shared with the CLI). This service imports it
and translates failures into clean API errors:

  FileNotFoundError  -> 503 (model artefacts missing — train the models)
  ValueError         -> 400 (bad waveform data)
"""
from __future__ import annotations

import sys
import traceback

from app.config import TRAINING_DIR

# make training/*.py importable (flat modules: config, cnn_model, predict_ecg)
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import predict_ecg  # noqa: E402  (from training/)


class ModelsUnavailableError(RuntimeError):
    """Model artefacts are missing — the API answers 503."""


class InvalidWaveformError(ValueError):
    """The uploaded waveform cannot be screened — the API answers 400."""


def screen(leads: dict[str, list[float]] | None,
           fs: float, sex: str | None,
           arrays: dict | None = None) -> dict:
    """Run criteria + Perceptron + CNN on one decoded waveform.

    Exactly one of (leads, arrays) is given; arrays is the numpy path used
    internally. Returns the raw dict from training/predict_ecg.py.
    """
    try:
        if leads is not None:
            return predict_ecg.predict_from_leads(leads, fs, sex=sex)
        raise InvalidWaveformError("no waveform provided")
    except FileNotFoundError as e:
        raise ModelsUnavailableError(str(e)) from e
    except ValueError as e:
        raise InvalidWaveformError(str(e)) from e
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise InvalidWaveformError(f"inference failed: {e}") from e


def screen_arrays(rec: "np.ndarray", fs: float, sex: str | None) -> dict:  # noqa: F821
    import predict_ecg  # already on sys.path

    try:
        # predict_from_leads expects leads; build them from the array
        leads = {name: rec[:, j] for j, name in enumerate(predict_ecg.LEADS)}
        return predict_ecg.predict_from_leads(leads, fs, sex=sex)
    except FileNotFoundError as e:
        raise ModelsUnavailableError(str(e)) from e
    except ValueError as e:
        raise InvalidWaveformError(str(e)) from e
