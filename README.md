# SPANDANA — AI Cardiac Intelligence

**See Beyond the Heartbeat.**

SPANDANA is an academic research platform for screening **left-ventricular
hypertrophy (LVH)** from standard 10-second 12-lead ECGs. It compares the
classic **Sokolow–Lyon** and **Cornell** voltage criteria against two
genuinely trained machine-learning models — a **Perceptron** on QRS voltage
features (Scikit-learn) and a **1D CNN** on the raw 100 Hz waveforms
(PyTorch) — on the same patient-wise held-out test set.

> **Medical disclaimer.** SPANDANA is a research prototype, **not** a
> clinically validated diagnostic device. It must not replace medical
> professionals or echocardiography. PTB-XL labels are cardiologist
> ECG-statement annotations, **not** imaging-confirmed diagnoses.

---

## 1 · What runs where

```
Browser (existing Next.js frontend, unchanged design)
   │  decodes CSV/TXT/DAT+HEA in-browser, measures SL/Cornell locally
   │  POST /api/v1/screen  (JSON: decoded 12-lead waveform)
   ▼
FastAPI backend (backend/app, port 8000)
   │  decodes uploaded files (.csv/.txt/.dat+.hea) via WFDB
   │  └─ training/predict_ecg.py  → Sokolow-Lyon + Cornell + Perceptron + CNN
   ▼
Model artefacts (backend/models/)
   perceptron_pipeline.joblib + perceptron_meta.json
   cnn_best.pt + cnn_meta.json
   preprocessing.json (train-fitted z-score stats — shared by train & inference)
   metrics.json (real evaluation payload for GET /api/v1/models/metrics)
```

**DEMO vs REAL:** until `NEXT_PUBLIC_FASTAPI_URL` is set **and** `/health`
answers, every AI number in the UI comes from the clearly-labelled demo
simulator. Once the backend is reachable, uploaded analyses show REAL
model probabilities (badge `REAL` / "Backend inference"), and the Model
Comparison module serves the real PTB-XL evaluation metrics.

## 2 · Folder structure

```
config.json               ← CENTRAL configuration (paths, fs, training params)
requirements-vscode.txt   ← Python dependencies (Windows-friendly)
setup.ps1                 ← one-time environment setup (PowerShell)
run_backend.ps1           ← python -m uvicorn app.main:app --reload --port 8000
run_frontend.ps1          ← existing npm scripts, port 3000
run_training.ps1          ← every research stage (download → evaluate)
training/                 ← all research code (plain .py scripts, no notebooks)
  download_dataset.py     ← PTB-XL v1.0.3 fetch (resumable, no duplicates)
  inspect_dataset.py      ← structure, folds, missing values
  extract_lvh.py          ← LVH cohort + documented control group
  preprocess_ecg.py       ← waveforms → data/preprocessed/signals.npy (+ --check)
  create_splits.py        ← patient-wise folds 1-8 / 9 / 10 + leakage proof
  extract_features.py     ← S(V1), R(V5), R(V6), R(aVL), S(V3) + criteria
  train_perceptron.py     ← sklearn Perceptron + Platt calibration
  evaluate_perceptron.py  ← held-out test metrics
  cnn_model.py            ← 1D CNN architecture + saliency (PyTorch)
  train_cnn.py            ← training loop (CPU ok, GPU optional, resumable)
  evaluate_cnn.py         ← test metrics + validation-tuned threshold
  predict_ecg.py          ← single-ECG inference shared by CLI and backend
  evaluate_all.py         ← 4-method comparison, curves, charts, metrics.json
  metrics.py              ← the ONE definition of every reported metric
  tests/                  ← pipeline tests (pytest)
backend/                  ← FastAPI service (app/) + tests (tests/)
data/                     ← PTB-XL, preprocessed arrays, metadata, reports
src/                      ← EXISTING frontend (api layer extended, UI preserved)
```

## 3 · Installation (Windows + VS Code)

Prerequisites: **Python 3.10–3.12**, **Node.js 18+**, VS Code with the
*Python* and *Jupyter* extensions (Pylance comes along).

```powershell
cd path\to\spandana
.\setup.ps1                # .venv + pip install + npm install + data dirs
```

Manual alternative:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-vscode.txt
npm install
```

On Windows, `pip install torch` delivers the **CPU build** — exactly what
SPANDANA needs (GPU is optional; see §5). Everything runs locally.

## 4 · Dataset: official PTB-XL v1.0.3

Source: <https://physionet.org/content/ptb-xl/> (PhysioNet, CC BY 4.0).
The pipeline downloads the official archive from the PhysioNet mirror —
**resumable** after interruption and **never downloads twice** (an existing
finished archive/extract is detected and reused).

```powershell
.\run_training.ps1 download            # full archive: metadata + records100 (~1.8 GB)
.\run_training.ps1 download -MetadataOnly   # only the small CSVs (~10 MB)
python training/download_dataset.py --zip-file D:\downloads\ptb-xl-1.0.3.zip
```

Defaults live in `config.json` (`dataset.raw_dir = data/ptbxl`); override
once with `SPANDANA_RAW_DIR` if you keep PTB-XL elsewhere.

Label policy (all decisions in `config.json`, evidence printed by the
scripts):

| Group | Definition |
| --- | --- |
| `LVH_positive` | SCP statement `LVH` with likelihood ≥ 100 |
| `LVH_low_certainty` | `LVH` with likelihood < 100 — **excluded** from both classes |
| `control` | **no** LVH statement — *not* the same as healthy |

`96.3 %` of control recordings carry other cardiac statements (NORM 9437,
ASMI 1744, NDT 1723, LAFB 1495, IMI 1484 …) — control ≠ healthy. See
`data/reports/label_limitations.md`.

**Splits** use the official patient-wise `strat_fold`: folds 1–8 → train,
9 → validation (tuning/calibration only), 10 → test (touched exactly once).
`create_splits.py` asserts zero patient overlap.

## 5 · Training (plain terminal commands)

```powershell
.\run_training.ps1 pipeline            # inspect → labels → splits → features
.\run_training.ps1 perceptron          # train + evaluate (seconds, CPU)
.\run_training.ps1 cnn                 # train + evaluate (≈20 min on CPU)
.\run_training.ps1 evaluate            # 4-method comparison
```

or directly:

```powershell
python training/train_perceptron.py
python training/train_cnn.py --epochs 40 --batch-size 64 --device auto
python training/train_cnn.py --resume          # continue an interrupted run
python training/evaluate_cnn.py
python training/evaluate_all.py
```

Configurable in `config.json → cnn` or via CLI (`--epochs --batch-size
--device --lr --patience --limit`). CPU training is fully supported; a CUDA
GPU is picked automatically when present. Imbalance handling: pos_weight-ed
`BCEWithLogitsLoss` (CNN), `class_weight="balanced"` (Perceptron). Early
stopping monitors validation PR-AUC; the best checkpoint is kept
(`cnn_best.pt`). Preprocessing (per-lead z-score, stats fitted on train
only) is shared verbatim between training and inference through
`backend/models/preprocessing.json`.

## 6 · Results — measured, not claimed

Held-out **test = strat_fold 10** (2 085 recordings, 101 LVH-positive);
thresholds tuned on **validation = strat_fold 9** only. Nothing below is
copied from literature — it is what SPANDANA actually measured.

| Method | Threshold | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sokolow–Lyon (tuned) | 3.83 mV | 0.9381 | 0.4079 | 0.6139 | 0.9546 | 0.4901 | 0.9181 | 0.4384 |
| Cornell (tuned) | 2.78 mV | 0.8916 | 0.2407 | 0.5743 | 0.9078 | 0.3392 | 0.8330 | 0.2402 |
| Perceptron | 0.1745 | 0.9535 | 0.5189 | 0.5446 | 0.9743 | 0.5314 | 0.9322 | 0.5174 |
| **CNN** | 0.9561 | **0.9727** | **0.7200** | **0.7129** | **0.9859** | **0.7164** | **0.9802** | **0.7813** |

At the **classic literature thresholds** the criteria behave as published:
Sokolow–Lyon 3.5 mV → sensitivity 0.733 / specificity 0.917; Cornell
(sex-specific 2.0/2.8 mV) → sensitivity 0.792 / specificity 0.699. Full
curves, confusion matrices and JSON/CSV: `data/reports/` — the API serves
the same payload from `backend/models/metrics.json`.

The CNN leads on every metric with real experimental evidence (largest
margin: PR-AUC 0.78 vs 0.52). That statement is backed by
`data/reports/model_comparison.csv` — regenerate it any time with
`python training/evaluate_all.py`.

## 7 · Backend

```powershell
.\run_backend.ps1          # python -m uvicorn app.main:app --reload --port 8000
```

Interactive docs: <http://localhost:8000/docs>.

| Endpoint | Description |
| --- | --- |
| `GET /health` | `{status, models, version, metrics_available}` |
| `POST /api/v1/screen` | real screening: JSON `{patient, waveform}` **or** multipart `file` (+`hea`, `sex`) |
| `GET /api/v1/models/metrics` | real evaluation metrics (503 until `evaluate_all.py` has run) |
| `GET /api/v1/models/status` | artefact-level availability report |

Screening response: `predictions[]` (perceptron + cnn, each with
`probability`, `positive`, `threshold`, `latencyMs`, `simulated:false`,
real `contributions` — model coefficients for the Perceptron, gradient
saliency for the CNN), `criteria` (Sokolow–Lyon/Cornell with thresholds and
agreement), `meta`. Errors are explicit: `503` when model artefacts are
missing (with the exact training commands), `400` for unusable waveforms
(missing lead, NaN, too short), `422` for undecodable uploads.

## 8 · Frontend

```powershell
.\run_frontend.ps1         # copies .env.example → .env.local, then npm run dev
```

Set `NEXT_PUBLIC_FASTAPI_URL=http://localhost:8000` (`.env.local`) and the
existing SPANDANA UI switches from the labelled demo simulator to real
inference on upload, and from placeholder metrics to the real PTB-XL
evaluation tables/curves. DEMO/REAL indicators, loading states and error
handling are built in; the design and all 8 modules are unchanged.

## 9 · Testing

```powershell
python -m pytest training/tests -q      # 12 tests: labels, splits, leakage, features, models, inference
python -m pytest backend/tests -q       # 12 tests: endpoints, uploads, invalid inputs, 503 paths
bun run scripts/test-parsers.ts         # 24 browser-decoder tests (unchanged)
npm run lint && npx tsc --noEmit        # frontend lint + types
npm run build                           # production build
```

## 10 · Limitations

1. **Labels are ECG statements, not anatomy.** SCP annotations describe
   what the annotator saw in the ECG; agreement with echocardiographic LVH
   is moderate at best. The models learn the annotation convention.
2. **Control ≠ healthy** — see §4.
3. **Voltage-criteria thresholds**: SPANDANA measures R/S against the
   median-of-trace baseline (mirroring the frontend); published cutoffs
   were calibrated on PR-segment baselines, so classic operating points
   shift slightly. Both classic and tuned results are reported side by side.
4. **Class imbalance** (1 : 18): PR-AUC is the fairest ranking metric;
   accuracy alone would be misleading.
5. **Single dataset, single population**: PTB-XL is a German cohort
   (1990s devices); generalisation to other populations is unproven.
6. Thresholds are tuned for F1 on a small validation fold (97 LVH) —
   the operating point, not the model, is the noisiest part of the chain.

## 11 · References

- Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography
  dataset.* Scientific Data 7, 154 (2020). <https://doi.org/10.1038/s41597-020-0495-6>
- Goldberger, A. et al. *PhysioBank, PhysioToolkit, and PhysioNet.*
  Circulation 101(23), e215–e220 (2000).
- Sokolow, M. & Lyon, T. P. *The ventricular complex in left ventricular
  hypertrophy.* American Heart Journal 37(1), 161–186 (1949).
- Casale, P. N., Devereux, R. B. et al. *Improved sex-specific criteria of
  left ventricular hypertrophy for electrocardiography.* Circulation 75(3),
  565–572 (1987).
- Dataset: <https://physionet.org/content/ptb-xl/1.0.3/>
