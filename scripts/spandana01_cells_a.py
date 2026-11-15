"""SPANDANA Notebook 01 - cell definitions, part A (title through section 6).

Each cell is a tuple: ("md" | "code", source_string).
All sources are raw triple-single-quoted strings; docstrings inside code use double quotes.
"""

CELLS_A = [
    ("md", r'''# SPANDANA — AI Cardiac Intelligence
## Notebook 01 · PTB-XL Dataset Exploration & LVH Label Verification

**See Beyond the Heartbeat.**

This notebook starts the SPANDANA AI research phase. It verifies the **PTB-XL** corpus (PhysioNet,
CC BY 4.0) and builds a fully traceable, **patient-wise** metadata set for the
**left-ventricular-hypertrophy (LVH) screening task** that the SPANDANA web platform will serve.

### What this notebook does
1. Loads the two **official metadata tables** — `ptbxl_database.csv` and `scp_statements.csv`.
2. Explains and verifies the dataset structure and the 100 Hz sampling conventions.
3. Identifies **LVH-related diagnostic codes** from the official SCP statement dictionary.
4. Extracts recordings annotated with LVH and defines a **careful control group** (non-LVH ≠ healthy).
5. Counts positives / negatives, visualises **8 real 12-lead recordings**, and analyses class balance and missing values.
6. Checks patient IDs, verifies the official patient-wise `strat_fold` split, and exports split-assignment CSVs.

### What this notebook deliberately does NOT do
- No model training, no feature engineering, no predictions — the Perceptron / CNN models come
  later, only after the dataset and its labels are verified.
- Nothing is invented: every number and plot below is computed from the official PhysioNet files.

> **Medical disclaimer.** SPANDANA is an academic research prototype. PTB-XL labels are
> cardiologist ECG-statement annotations, **not** imaging-confirmed diagnoses. Nothing here is a
> medical device or a diagnostic aid.

**Artefacts produced** (in `/content/spandana_outputs/`): `spandana_lvh_metadata.csv`,
`spandana_split_assignments.csv`, `spandana_patient_summary.csv`.

**Colab runtime notes.** A plain CPU runtime is sufficient. The first run downloads the official
1.8 GB archive from PhysioNet and extracts the metadata plus the 100 Hz signals (~0.6 GB);
later runs skip the download. Run the notebook top-to-bottom ("Runtime → Run all").'''),

    ("md", r'''## 1 · Setup & configuration

We install **WFDB** (the PhysioNet waveform I/O library) and collect every research decision in one
configuration block — the LVH likelihood threshold, the official fold-to-split mapping, the demo
sizes. Nothing that affects the cohort definition is ever hidden inside code. If you already host
PTB-XL on Google Drive, mount it, set `PTBXL_DIR` to that folder, and the download step is skipped.'''),

    ("code", r'''# 1.1 Install the waveform library (pandas / numpy / matplotlib are preinstalled in Colab)
%pip install -q wfdb

# 1.2 Imports
import ast, sys, textwrap, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)

print("Python :", sys.version.split()[0])
print("Pandas :", pd.__version__, "| NumPy :", np.__version__, "| WFDB :", wfdb.__version__)

# 1.3 Configuration — every research decision lives here, never buried in code
PTBXL_VERSION = "1.0.3"
ZIP_URL     = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"  # official PhysioNet get-zip target
ZIP_PATH    = Path("/content/ptbxl.zip")
EXTRACT_ROOT = Path("/content/ptbxl")
PTBXL_DIR   = None              # e.g. Path("/content/drive/MyDrive/ptb-xl-1.0.3") to skip the download
OUTPUT_DIR  = Path("/content/spandana_outputs")

FS_EXPECTED = 100               # official sampling rate of the records100/*_lr files
LVH_LIKELIHOOD_THRESHOLD = 100  # SCP likelihood required to call a recording LVH-positive (policy, section 5)
VAL_FOLD, TEST_FOLD = 9, 10     # official recommendation: strat_fold 9 = validation, 10 = test
N_DEMO_POS, N_DEMO_NEG = 4, 4   # real ECGs to visualise per group (8 total, within the 5-10 requested)
KEEP_ZIP = False                # delete the 1.8 GB archive after extraction to save Colab disk
RANDOM_STATE = 42

# SPANDANA palette (matches the web platform design)
SPANDANA_NAVY = "#0B1728"
SPANDANA_BLUE = "#1F6FB2"
SPANDANA_CYAN = "#00A8B5"
SPANDANA_RED  = "#C0392B"
SPANDANA_GREY = "#8A94A6"

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Configuration ready. Output directory:", OUTPUT_DIR)'''),

    ("code", r'''# 1.4 Locate or download PTB-XL v1.0.3
# This notebook only needs two parts of the archive:
#   - the two official metadata CSVs at the archive root,
#   - records100/  (the 100 Hz signal files, ~0.6 GB uncompressed).
# Extraction is layout-independent: the archive top folder is discovered from the zip itself.

def find_data_dir():
    candidates = []
    if PTBXL_DIR is not None:
        candidates.append(Path(PTBXL_DIR))
    if EXTRACT_ROOT.exists():
        candidates.append(EXTRACT_ROOT)
        candidates += [p for p in EXTRACT_ROOT.iterdir() if p.is_dir()]
    for c in candidates:
        if (c / "ptbxl_database.csv").exists():
            return c
    return None

DATA_DIR = find_data_dir()

if DATA_DIR is None:
    if not ZIP_PATH.exists():
        print(f"Downloading PTB-XL v{PTBXL_VERSION} (1.8 GB) from the official PhysioNet mirror ...")
        !wget -c --progress=dot:giga -O {ZIP_PATH} {ZIP_URL}
        assert ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1e9, "Download failed - retry, or fetch the zip manually from physionet.org."
    print("Extracting metadata + records100/ (100 Hz signals) ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        wanted = [n for n in zf.namelist()
                  if n.split("/")[-1] in ("ptbxl_database.csv", "scp_statements.csv")
                  or ("/records100/" in n and n.endswith((".dat", ".hea")))]
        zf.extractall(path=EXTRACT_ROOT, members=wanted)
    print(f"Extracted {len(wanted)} files.")
    if not KEEP_ZIP:
        ZIP_PATH.unlink(missing_ok=True)   # free the Colab disk
    DATA_DIR = find_data_dir()

assert DATA_DIR is not None and (DATA_DIR / "ptbxl_database.csv").exists(), \
    "PTB-XL not found - check the download or point PTBXL_DIR at your local copy."
print("PTB-XL data directory:", DATA_DIR)'''),

    ("md", r'''## 2 · Loading the official metadata

PTB-XL ships exactly two metadata tables:

| File | Grain | Key content |
|---|---|---|
| **`ptbxl_database.csv`** | one row per recording | `patient_id`, demographics, `scp_codes`, `filename_lr` (100 Hz path), `strat_fold` |
| **`scp_statements.csv`** | one row per SCP-ECG statement code | `description`, `diagnostic` flag, `diagnostic_class` (NORM / MI / STTC / CD / HYP) |

The `scp_codes` cell inside the database is a **stringified dictionary** `{'CODE': likelihood}` and
must be parsed before any labeling — done in section 5. We also snapshot the raw column list here
(`ORIGINAL_COLS`), so the missing-value analysis in section 7 inspects the *official* schema, not
the derived columns we add along the way.'''),

    ("code", r'''# 2.1 Load the two official tables
db  = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
ORIGINAL_COLS = list(db.columns)          # raw schema snapshot before we add derived columns

from IPython.display import display

print(f"ptbxl_database.csv : {db.shape[0]:,} recordings x {db.shape[1]} columns")
print(f"scp_statements.csv : {scp.shape[0]} statement codes x {scp.shape[1]} columns")

print("\nptbxl_database.csv — first rows:")
display(db.head(3))
print("scp_statements.csv — first rows:")
display(scp.head(8))'''),

    ("code", r'''# 2.2 Column-level overview of the recording table
print("=== ptbxl_database.csv — dtypes & non-null counts ===")
db.info()

demo_cols = [c for c in ["age", "sex", "height", "weight", "strat_fold"] if c in db.columns]
print("\n=== Key numeric metadata — descriptive statistics ===")
display(db[demo_cols].describe().round(1))

print("\nsex value counts (coding interpreted in section 7):")
print(db.sex.value_counts(dropna=False).to_string())'''),

    ("md", r'''## 3 · Dataset structure & sampling conventions

Per the official documentation, PTB-XL v1.0.3 provides **21,799 clinical 10-second 12-lead ECG
recordings** from **18,869 patients**, collected at the Physikalisch-Technische Bundesanstalt (PTB)
and partner sites between October 1989 and June 1996 with a wide range of commercial recorders.
We verify these documented facts against the actual metadata instead of trusting them blindly.

**Two signal speeds are distributed for every recording:**

| Folder | Metadata column | Sampling rate | Role in SPANDANA |
|---|---|---|---|
| `records100/` | `filename_lr` (`*_lr`) | **100 Hz** | used throughout this notebook — sufficient for QRS-morphology screening (Sokolow-Lyon / Cornell / ML) and much lighter |
| `records500/` | `filename_hr` (`*_hr`) | 500 Hz | reserved for later high-resolution experiments |

Each signal file stores all 12 standard leads in the fixed order
**I, II, III, aVR, aVL, aVF, V1 … V6** (verified per-file below — never assumed). Recordings carry
one or more SCP-ECG statements plus the recommended **`strat_fold`** column (1–10) for patient-wise
train/validation/test splitting.'''),

    ("code", r'''# 3.1 Verify the documented structure directly against the metadata
n_records   = len(db)
n_patients  = db.patient_id.nunique()
per_patient = db.groupby("patient_id").size()

print(f"Recordings             : {n_records:,}")
print(f"Unique patients        : {n_patients:,}")
print(f"Recordings per patient : min={per_patient.min()}, max={per_patient.max()}, mean={per_patient.mean():.2f}")
print("Patients by number of recordings:")
print(per_patient.value_counts().sort_index().to_string())

print("\nstrat_fold distribution (official, patient-wise split recommendation):")
print(db.strat_fold.value_counts().sort_index().to_string())

# 3.2 Spot-check the 100 Hz signal conventions by reading only the .hea headers (no signal data needed)
fss, units, lead_sets = set(), set(), set()
for ecg_id in db.sample(5, random_state=RANDOM_STATE).index:
    hdr = wfdb.rdheader(str(DATA_DIR / db.at[ecg_id, "filename_lr"]))
    fss.add(hdr.fs)
    units.update(hdr.units)
    lead_sets.add(tuple(s.upper() for s in hdr.sig_name))

DISPLAY_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # clinical display names

print("\nSampling rates seen in sampled *_lr headers :", fss)
print("Signal units seen                           :", units)
print("Distinct (case-normalised) lead orders seen :", len(lead_sets))
assert fss == {FS_EXPECTED}, f"expected {FS_EXPECTED} Hz in all *_lr files, saw {fss}"
assert lead_sets == {tuple(s.upper() for s in DISPLAY_LEADS)}, f"unexpected lead order: {lead_sets}"
print("OK: 100 Hz, 12 standard leads in fixed order, physical units — verified on sampled headers.")'''),

    ("md", r'''## 4 · Identifying LVH-related diagnostic codes

`scp_statements.csv` is the official code dictionary. Diagnostic statements carry `diagnostic == 1`
and are grouped into five superclasses (`diagnostic_class`):

| Class | Meaning |
|---|---|
| `NORM` | normal ECG |
| `MI` | myocardial infarction |
| `STTC` | ST/T-change |
| `CD` | conduction disturbance |
| `HYP` | hypertrophy |

The hypertrophy superclass contains **left ventricular hypertrophy (LVH)** plus related codes —
right ventricular hypertrophy, atrial overload/enlargement, septal hypertrophy — and there is also
a *form* statement `VCLVH` ("voltage criteria (QRS) for left ventricular hypertrophy"), which is an
ECG morphological observation, **not** a diagnostic class. For the SPANDANA screening task the
positive label is **only the diagnostic `LVH` code**; every related code is tracked transparently,
because it shapes the control group in section 6.'''),

    ("code", r'''# 4.1 The official statement dictionary
diag = scp[scp.diagnostic == 1]
print(f"{len(scp)} codes total, of which {len(diag)} are diagnostic statements (diagnostic == 1).")
print("Non-diagnostic codes are 'form' (morphology observations) or 'rhythm' statements — they are NOT diagnoses.")
print("\nDiagnostic statements per superclass (diagnostic_class):")
print(diag.diagnostic_class.value_counts(dropna=False).to_string())

print("\nAll codes in the HYP (hypertrophy) superclass:")
display(scp[scp.diagnostic_class == "HYP"][["description", "diagnostic", "diagnostic_subclass"]])'''),

    ("code", r'''# 4.2 Identify LVH-related codes FROM the dictionary file, not from memory
# "LVH-related" = any statement whose official description mentions hypertrophy / enlargement / overload.
hyp_related = scp[scp.description.str.contains("hypertrophy|enlargement|overload", case=False, na=False)]

# The positive label is the DIAGNOSTIC 'LVH' code only.
LVH_CODES = set(hyp_related[hyp_related.description.str.contains("left ventricular", case=False, na=False)].index)
OTHER_HYP_CODES = set(scp[scp.diagnostic_class == "HYP"].index) - set(LVH_CODES)
FORM_LVH_CODES = set(hyp_related.index) - set(scp[scp.diagnostic_class == "HYP"].index)

print("Diagnostic LVH label codes     :", sorted(LVH_CODES))
print("Other hypertrophy-class codes  :", sorted(OTHER_HYP_CODES))
print("LVH-related FORM statements    :", sorted(FORM_LVH_CODES))

assert "LVH" in LVH_CODES, "LVH code not found in scp_statements.csv - do not proceed blindly."

if "VCLVH" in scp.index:
    print("\nImportant distinction — the 'voltage criteria' statement:")
    print(scp.loc[["LVH", "VCLVH"], ["description", "diagnostic", "form"]].to_string())
    print("-> VCLVH is a FORM statement (an ECG morphological observation), NOT a diagnostic class.")
    print("-> SPANDANA labels use the diagnostic LVH code only; VCLVH counts are reported in section 5.")'''),

    ("md", r'''## 5 · Parsing `scp_codes` and extracting the LVH cohort

Each `scp_codes` cell is a stringified dictionary `{'CODE': likelihood}`. The official likelihood
semantics (Wagner et al. 2020, section 2.6):

> *"… for all diagnostic statements, a likelihood information was extracted based on certain
> keywords in the ECG report … The likelihood ranges from 0 to 100 conveying the certainty the
> cardiologist … For form and rhythm statements or in cases where no likelihood information was
> available, the corresponding likelihood was set to zero."*

So **100 means the diagnosing cardiologist was very certain**; intermediate values reflect the
keyword-derived certainty; 0 means no likelihood information. Likelihood is **not** a
validated clinical truth — a limitation documented again in section 11.

**Labeling policy (configurable in section 1).** A recording is **LVH-positive** iff the diagnostic
`LVH` statement has likelihood ≥ `LVH_LIKELIHOOD_THRESHOLD` (default **100**). Recordings where LVH
is annotated with a *lower* likelihood are kept as a separate, clearly flagged `LVH_low_certainty`
group and **excluded from the modelling cohort** — they are neither confident positives nor
controls. Recordings without any LVH statement become the control pool (section 6).'''),

    ("code", r'''# 5.1 Parse the stringified scp_codes dictionaries
# Each cell looks like {'NORM': 100, 'IMI': 0, ...} -> {statement: likelihood}
db["scp_dict"] = db.scp_codes.apply(ast.literal_eval)

# Aggregate each recording's diagnostic statements into the official superclasses
def superclass_set(scp_dict):
    return {diag.at[code, "diagnostic_class"]
            for code in scp_dict
            if code in diag.index and pd.notna(diag.at[code, "diagnostic_class"])}

db["superclasses"] = db.scp_dict.apply(superclass_set)

# Per-recording LVH bookkeeping (diagnostic 'LVH' code only — VCLVH is tracked separately)
db["lvh_likelihood"] = db.scp_dict.apply(lambda d: d.get("LVH", np.nan))
db["has_vclvh_form"] = db.scp_dict.apply(lambda d: "VCLVH" in d)

print("Recordings carrying the diagnostic LVH statement :", int(db.lvh_likelihood.notna().sum()))
print("Recordings carrying the VCLVH form statement     :", int(db.has_vclvh_form.sum()),
      f"(of which {int((db.has_vclvh_form & db.lvh_likelihood.notna()).sum())} also carry LVH)")

print("\nLikelihood values observed for the diagnostic LVH statement (official range 0-100):")
print(db.lvh_likelihood.value_counts(dropna=False).sort_index(ascending=False).to_string())'''),

    ("code", r'''# 5.2 Apply the labeling policy from the configuration (default threshold = 100)
TH = LVH_LIKELIHOOD_THRESHOLD

is_pos = db.lvh_likelihood >= TH
is_unc = db.lvh_likelihood.notna() & (db.lvh_likelihood < TH)   # LVH annotated with lower certainty
db["label"] = np.select([is_pos, is_unc], ["LVH_positive", "LVH_low_certainty"], default="control")
db["is_control"] = db.label == "control"

summary = pd.DataFrame({
    "recordings": db.label.value_counts(),
    "patients"  : db.groupby("label").patient_id.nunique(),
}).reindex(["LVH_positive", "LVH_low_certainty", "control"]).fillna(0).astype(int)
summary["% of recordings"] = (100 * summary.recordings / len(db)).round(1)

print(f"Label summary (policy: LVH-positive iff LVH likelihood >= {TH}):")
print(summary.to_string())

mixed_labels = db.groupby("patient_id").label.nunique()
print(f"\nPatients holding recordings with different labels (repeat visits): {int((mixed_labels > 1).sum())}"
      " — handled patient-wise in section 9.")'''),

    ("md", r'''## 6 · Defining the control group — carefully

**"Control" means *no LVH statement*, never *healthy*.** The pool is deliberately heterogeneous and
stays that way, because a real screening deployment receives mixed referrals. We track three nested
control subsets:

| Subset | Definition | Role |
|---|---|---|
| `control` (primary) | no diagnostic `LVH` statement at all | the negative class for later SPANDANA notebooks |
| `clean control` | control **and** no hypertrophy-class code at all (no RVH / atrial overload / septal hypertrophy either) | stricter negatives, useful for threshold studies |
| `NORM-only` | control whose diagnostic statements are exactly `NORM` | cleanest, but small and biased towards easy cases |

Recordings with `LVH` at likelihood < 100 (the `LVH_low_certainty` group from section 5) belong to
**neither** class and are excluded from the cohort.

> **Documented label limitation #1.** SCP statements describe what the ECG looked like to the
> annotating cardiologists. They are **not** validated against echocardiography — PTB-XL contains no
> LV-mass measurements. ECG-based LVH criteria have limited sensitivity, so the control pool will
> inevitably contain patients with anatomical LVH whose ECG did not show it. Any performance metric
> computed on this cohort later must report this dilution as an evaluation-context limitation.'''),

    ("code", r'''# 6.1 Build the nested control subsets and prove that control != healthy
db["clean_control"] = db.is_control & ~db.superclasses.apply(lambda s: "HYP" in s)
db["norm_only"]     = db.is_control & db.superclasses.apply(lambda s: s == {"NORM"})

composition = pd.DataFrame({
    "recordings": [int((db.label == "LVH_positive").sum()),
                   int(db.is_control.sum()),
                   int(db.clean_control.sum()),
                   int(db.norm_only.sum()),
                   int((db.label == "LVH_low_certainty").sum())],
    "patients"  : [db.loc[db.label == "LVH_positive", "patient_id"].nunique(),
                   db.loc[db.is_control, "patient_id"].nunique(),
                   db.loc[db.clean_control, "patient_id"].nunique(),
                   db.loc[db.norm_only, "patient_id"].nunique(),
                   db.loc[db.label == "LVH_low_certainty", "patient_id"].nunique()],
}, index=["LVH_positive (label)",
          "control — no LVH statement (primary)",
          "clean control — no HYP-class code at all",
          "NORM-only control (cleanest)",
          "LVH_low_certainty (excluded from cohort)"])
print("Cohort composition:")
print(composition.to_string())

# What is actually inside the control pool? A recording can carry several classes at once.
ctrl_classes = pd.Series([c for s in db.loc[db.is_control, "superclasses"] for c in s])
print("\nDiagnostic superclass occurrences INSIDE the control pool:")
print(ctrl_classes.value_counts().to_string())
print(textwrap.fill(
    "Read this table carefully: the control pool contains thousands of recordings with MI, "
    "conduction disturbances, ST/T changes, and even other hypertrophy codes (right ventricular "
    "hypertrophy, atrial overload, septal hypertrophy). 'Control' means 'no LVH statement' — "
    "it never means 'healthy'.", 110))'''),
]
