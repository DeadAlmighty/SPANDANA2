"""Lead-name helpers shared by the ECG decoders."""
from __future__ import annotations

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def normalize_lead(name: str) -> str:
    """'avR' / 'AVR' -> 'aVR'; otherwise the upper-cased name."""
    upper = name.strip().upper()
    return _ALIASES.get(upper, upper)
