#!/usr/bin/env python3
"""SPANDANA step 1/6 — obtain the official PTB-XL v1.0.3 dataset.

Two modes:
  default        : full archive (metadata + records100/, ~1.8 GB download)
  --metadata-only: just the three small CSVs + RECORDS (~10 MB)

Download resume: if a previous partial download exists, it is continued
(HTTP Range) instead of restarted, and finished archives are never
downloaded twice (size check against the expected 1.8 GB).

Usage (VS Code terminal, project root):
    python training/download_dataset.py
    python training/download_dataset.py --metadata-only
    python training/download_dataset.py --zip-file D:/downloads/ptb-xl-1.0.3.zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from config import load_config

EXPECTED_ZIP_BYTES = 1_846_000_000  # ~1.8 GB, used only for sanity checks


def _python_download(url: str, dest: Path) -> None:
    """Resumable downloader (works everywhere incl. plain Windows Python)."""
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        if done and r.status != 206:  # server ignored Range — start over
            done = 0
        total = done + int(r.headers.get("Content-Length") or 0)
        mode = "ab" if done else "wb"
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f} GB "
                      f"({done * 100 // max(total, 1)}%)", end="", flush=True)
    print()


def download_zip(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1e9:
        print(f"Archive already downloaded ({dest.stat().st_size / 1e9:.2f} GB) — skipping.")
        return
    if shutil.which("wget"):  # preferred where available: resumable, quiet-ish
        print(f"Downloading {url} via wget ...")
        rc = os_system(f'wget -c --progress=dot:giga -O "{dest}" "{url}"')
        if rc != 0:
            print("wget failed — falling back to the Python downloader.")
            _python_download(url, dest)
    else:
        print(f"Downloading {url} (resumable) ...")
        _python_download(url, dest)
    assert dest.stat().st_size > 1e9, "Download failed — re-run to resume, or use --zip-file."


def os_system(cmd: str) -> int:
    import subprocess
    return subprocess.call(cmd, shell=True)


def fetch_single_file(base: str, name: str, dest_dir: Path) -> Path:
    dest = dest_dir / name
    if dest.exists():
        print(f"  already present: {name}")
        return dest
    url = f"{base}/{name}"
    _python_download(url, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-only", action="store_true",
                    help="download only the small metadata files (~10 MB)")
    ap.add_argument("--zip-file", type=str, default=None,
                    help="use a local PTB-XL zip instead of downloading")
    ap.add_argument("--keep-zip", action="store_true",
                    help="do not delete the archive after extraction")
    args = ap.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()
    extract_root = cfg.raw_dir
    zip_path = extract_root / "ptb-xl.zip"

    data_dir = cfg.find_data_dir()
    need_signals = not args.metadata_only
    if data_dir is not None:
        has_signals = any((data_dir / "records100").glob("*/*.dat")) if need_signals else True
        scp_ok = (data_dir / "scp_statements.csv").exists()
        if has_signals and scp_ok:
            print(f"PTB-XL already available at: {data_dir}")
            return
        if not scp_ok or not need_signals:
            # fetch the small official CSVs (incl. the statement dictionary)
            base = cfg.dataset["physionet_file_base"]
            print(f"Fetching missing metadata files from {base} ...")
            for name in ("ptbxl_database.csv", "scp_statements.csv", "RECORDS"):
                fetch_single_file(base, name, data_dir)
            if scp_ok or not need_signals:
                return
        print(f"Metadata found at {data_dir} but 100 Hz signals are missing — "
              "fetching the full archive.")
    else:
        print("No extracted PTB-XL found — fetching metadata.")

    if args.zip_file:
        src = Path(args.zip_file)
        assert src.exists(), f"zip file not found: {src}"
        shutil.copyfile(src, zip_path)
    elif need_signals or data_dir is None:
        download_zip(cfg.dataset["zip_url"], zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            wanted = [n for n in zf.namelist()
                      if n.split("/")[-1] in ("ptbxl_database.csv", "scp_statements.csv")
                      or ("/records100/" in n and n.endswith((".dat", ".hea")))
                      or n.endswith("RECORDS")]
            print(f"Extracting metadata + records100/ ({len(wanted)} files) ...")
            zf.extractall(path=extract_root, members=wanted)
        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)  # free ~1.8 GB
    else:
        # metadata-only for an existing partial install
        base = cfg.dataset["physionet_file_base"]
        print(f"Fetching metadata files from {base} ...")
        for name in ("ptbxl_database.csv", "scp_statements.csv", "RECORDS"):
            fetch_single_file(base, name, data_dir)

    data_dir = cfg.find_data_dir()
    assert data_dir is not None, "PTB-XL extraction failed."
    assert (data_dir / "scp_statements.csv").exists(), \
        "scp_statements.csv missing — re-run with --metadata-only."
    n_dat = len(list((data_dir / "records100").glob("*/*.dat"))) \
        if (data_dir / "records100").exists() else 0
    print(f"\nDone. PTB-XL data directory: {data_dir}")
    print(f"  records100 signal files present: {n_dat} (full dataset = 21,799)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted — the download will resume on the next run.")
