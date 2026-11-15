"""Real ECG file decoding for uploaded recordings (no simulation).

Mirrors the in-browser decoders of the frontend (src/lib/ecg/parsers.ts):

  CSV / TXT : one row per sample, 12 numeric columns in standard lead order
              (I, II, III, aVR, aVL, aVF, V1..V6); delimiter auto-detected
              (comma / semicolon / tab / whitespace, with European decimal fix)
  WFDB      : .dat + .hea pair decoded with the wfdb library (signals in mV)

Every failure raises EcgDecodeError with an actionable message that the API
returns as HTTP 400/422.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from app.utils.leads import STANDARD_LEADS, normalize_lead

MIN_ROWS = 50          # mirrors the frontend guard (>= 0.5 s at 100 Hz)
MAX_BYTES = 64 * 1024 * 1024


class EcgDecodeError(ValueError):
    """Raised when an upload cannot be decoded as a 12-lead ECG."""


def _split_rows(text: str) -> list[list[str]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < MIN_ROWS:
        raise EcgDecodeError(
            f"only {len(lines)} usable rows — a multi-second 12-lead recording "
            f"needs thousands of samples (minimum {MIN_ROWS})")
    sample = lines[: min(60, len(lines))]
    counts = {d: sum(line.count(c) for line in sample) for d, c in
              (("comma", ","), ("semicolon", ";"), ("tab", "\t"), ("space", " "))}
    delimiter = max(counts, key=counts.get)
    if delimiter == "comma":
        rx, fix_dec = ",", False
    elif delimiter == "semicolon":
        rx, fix_dec = ";", True
    elif delimiter == "tab":
        rx, fix_dec = "\t", False
    else:
        rx, fix_dec = None, False

    rows: list[list[str]] = []
    for line in lines:
        parts = line.split() if rx is None else line.split(rx)
        parts = [p.strip() for p in parts if p.strip()]
        if fix_dec:
            parts = [p.replace(",", ".") for p in parts]
        rows.append(parts)
    return rows


def decode_csv_bytes(data: bytes, filename: str) -> tuple[dict[str, np.ndarray], float]:
    """CSV/TXT bytes -> (leads dict, fs). fs is unknown for plain CSV —
    the caller passes 100 Hz (PTB-XL convention)."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise EcgDecodeError(f"{filename}: not valid UTF-8 text ({e})") from e
    rows = _split_rows(text)

    width = max(len(r) for r in rows)
    # skip a header row when the first row is not numeric
    def numeric(cells: list[str]) -> bool:
        try:
            [float(c) for c in cells[:width]]
            return True
        except ValueError:
            return False
    if not numeric(rows[0]):
        rows = rows[1:]
    if width != 12:
        raise EcgDecodeError(
            f"{filename}: expected 12 columns (I..V6), found {width}. "
            "Export one row per sample, 12 leads, standard lead order.")
    try:
        arr = np.array([[float(c) for c in r[:12]] for r in rows], dtype=np.float64)
    except ValueError as e:
        raise EcgDecodeError(f"{filename}: non-numeric cell ({e})") from e
    if not np.isfinite(arr).all():
        raise EcgDecodeError(f"{filename}: NaN/Inf values are not allowed")
    return {name: arr[:, j] for j, name in enumerate(STANDARD_LEADS)}, 100.0


def decode_wfdb_bytes(dat: bytes, hea: bytes, stem: str = "upload") -> \
        tuple[dict[str, np.ndarray], float]:
    """DAT+HEA bytes -> (leads dict, fs) via a temp folder and wfdb."""
    if not hea.strip():
        raise EcgDecodeError("the .hea header file is empty")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        rec_stem = Path(stem).stem or "upload"
        (base / f"{rec_stem}.dat").write_bytes(dat)
        (base / f"{rec_stem}.hea").write_bytes(hea)
        try:
            import wfdb
            sig, info = wfdb.rdsamp(str(base / rec_stem))
        except Exception as e:  # noqa: BLE001 - surfaced to the client
            raise EcgDecodeError(f"WFDB decode failed: {e}") from e
        names = [normalize_lead(n) for n in info["sig_name"]]
        missing = [l for l in STANDARD_LEADS if l not in names]
        if missing:
            raise EcgDecodeError(
                f"missing leads in header: {', '.join(missing)}")
        idx = [names.index(l) for l in STANDARD_LEADS]
        sig = np.asarray(sig[:, idx], dtype=np.float64)
        if not np.isfinite(sig).all():
            raise EcgDecodeError("recording contains non-finite samples")
        return ({name: sig[:, j] for j, name in enumerate(STANDARD_LEADS)},
                float(info["fs"]))


def decode_upload(data: bytes, filename: str,
                  hea_data: bytes | None = None) -> tuple[dict[str, np.ndarray], float]:
    """Dispatch on the file extension. Returns (leads, fs)."""
    if len(data) > MAX_BYTES:
        raise EcgDecodeError("file too large (limit 64 MB)")
    name = filename.lower()
    if name.endswith((".csv", ".txt")):
        return decode_csv_bytes(data, filename)
    if name.endswith(".dat"):
        if hea_data is None:
            raise EcgDecodeError(
                "a .dat recording needs its matching .hea header — attach both files")
        return decode_wfdb_bytes(data, hea_data, filename)
    raise EcgDecodeError(
        f"unsupported file type '{filename}' — upload .csv, .txt or .dat (+.hea)")
