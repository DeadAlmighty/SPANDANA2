"""Smoke-run every code cell of SPANDANA_01 against the REAL official metadata.

- Replaces the download cell with a stub pointing at the pre-staged /tmp/ptbxl copy.
- Patches wfdb.rdheader / wfdb.rdrecord to fetch missing sample files from PhysioNet on demand
  (a few KB for headers, ~25 kB per recording) so plotting cells run on real signals too.
- Redirects /content/... paths to a temp dir.
Every other code cell is executed verbatim, in order, in a shared namespace.
"""
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

NB_PATH = Path("/home/z/my-project/download/SPANDANA_01_Dataset_Exploration.ipynb")
STAGE = Path("/tmp/ptbxl")

nb = json.loads(NB_PATH.read_text())
code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

PREAMBLE = '''
import subprocess
import matplotlib
matplotlib.use("Agg")
print("[smoke] Agg backend active")
'''

STUB_DOWNLOAD = '''
DATA_DIR = Path("/tmp/ptbxl")
print("[smoke] download cell stubbed: DATA_DIR =", DATA_DIR)
'''

PATCH_WFDB = '''
import subprocess as _sp
_orig_rdheader = wfdb.rdheader
def _rdheader_dl(path, **kw):
    p = Path(str(path))
    if not p.with_suffix(".hea").exists():
        rel = p.relative_to(DATA_DIR).as_posix()
        url = f"https://physionet.org/files/ptb-xl/1.0.3/{rel}.hea"
        p.parent.mkdir(parents=True, exist_ok=True)
        _sp.run(["curl", "-sSL", "-o", str(p.with_suffix(".hea")), url], check=True)
    return _orig_rdheader(path, **kw)
wfdb.rdheader = _rdheader_dl

_orig_rdrecord = wfdb.rdrecord
def _rdrecord_dl(path, **kw):
    p = Path(str(path))
    for ext in (".hea", ".dat"):
        if not p.with_suffix(ext).exists():
            rel = p.relative_to(DATA_DIR).as_posix()
            url = f"https://physionet.org/files/ptb-xl/1.0.3/{rel}{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            _sp.run(["curl", "-sSL", "-o", str(p.with_suffix(ext)), url], check=True)
    return _orig_rdrecord(path, **kw)
wfdb.rdrecord = _rdrecord_dl
print("[smoke] wfdb fetch-on-missing patches active")
'''

ns = {}


def src_of(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def transform(src):
    src = src.replace("/content/", "/tmp/smoke_content/")
    lines = [ln for ln in src.splitlines()
             if not ln.lstrip().startswith(("%", "!"))]
    return "\n".join(lines)


# Locate the download cell (contains wget) to stub it
exec(PREAMBLE, ns)
patched_done = False

failures = []
for n, (idx, cell) in enumerate(code_cells):
    src = src_of(cell)
    if "wget" in src:
        src = STUB_DOWNLOAD + PATCH_WFDB
        patched_done = True
    src = transform(src)
    try:
        exec(compile(src, f"cell_{idx:02d}", "exec"), ns)
        print(f"[smoke] cell {idx:02d} OK")
    except Exception:
        failures.append(idx)
        print(f"[smoke] cell {idx:02d} FAILED:")
        traceback.print_exc()
        break

assert patched_done, "download cell not found"

if failures:
    sys.exit(1)

# ---- post-run verification against known REAL facts -------------------------
db = ns["db"]
n = len(db)
pos = int((db.label == "LVH_positive").sum())
unc = int((db.label == "LVH_low_certainty").sum())
ctrl = int((db.label == "control").sum())
patients = db.patient_id.nunique()
pos_pat = db.loc[db.label == "LVH_positive", "patient_id"].nunique()

print("\n[smoke] cohort results:", {
    "recordings": n, "patients": patients, "positive": pos, "low_certainty": unc, "control": ctrl,
    "positive_patients": pos_pat,
})
assert n == 21799 and patients == 18869, "unexpected dataset size"
assert pos == 1085 and pos_pat == 950, f"unexpected LVH counts: {pos}/{pos_pat}"
assert pos + unc + ctrl == n, "labels not exhaustive"

out_dir = Path("/tmp/smoke_content/spandana_outputs")
csvs = sorted(p.name for p in out_dir.glob("*.csv"))
print("[smoke] artefacts:", csvs)
assert csvs == ["spandana_lvh_metadata.csv", "spandana_patient_summary.csv",
                "spandana_split_assignments.csv"], csvs

import pandas as pd
chk = pd.read_csv(out_dir / "spandana_split_assignments.csv")
print("[smoke] split_assignments rows:", len(chk),
      "| split counts:", chk.split.value_counts().to_dict())
assert len(chk) == n
assert chk.split.value_counts().to_dict() == {"train": 17418, "val": 2183, "test": 2198}
assert (chk.groupby("patient_id").split.nunique() == 1).all(), "patient leakage in exported split!"

print("\n[smoke] ALL CODE CELLS EXECUTED OK AGAINST REAL PTB-XL DATA — VERIFICATION PASSED")
