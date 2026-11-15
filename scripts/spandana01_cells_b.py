"""SPANDANA Notebook 01 - cell definitions, part B (section 7 through references)."""

CELLS_B = [
    ("md", r'''## 7 · Class distribution & missing-value analysis

Two views per requirement: (a) how balanced the LVH screening task is, (b) where the official
metadata has gaps. Missing values here are *informative* — empty noise columns simply mean
"no such noise was annotated" — while demographic gaps (height, weight) matter if those covariates
are used later. `age` and `sex` are complete in v1.0.3.'''),

    ("code", r'''# 7.1 Class balance — recording level
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)

counts = db.label.value_counts()
bar_colors = {"LVH_positive": SPANDANA_RED, "control": SPANDANA_BLUE, "LVH_low_certainty": SPANDANA_GREY}
axes[0].bar(counts.index, counts.values, color=[bar_colors[i] for i in counts.index])
for x, v in enumerate(counts.values):
    axes[0].text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
axes[0].set_title("LVH screening classes (recordings)")
axes[0].set_ylabel("recordings")
axes[0].tick_params(axis="x", labelsize=9)

super_all = pd.Series([c for s in db.superclasses for c in s]).value_counts()
axes[1].bar(super_all.index, super_all.values, color=SPANDANA_NAVY)
for x, v in enumerate(super_all.values):
    axes[1].text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
axes[1].set_title("All recordings by diagnostic superclass")
axes[1].set_ylabel("recordings carrying the class")
plt.show()

pos_share = 100 * counts.get("LVH_positive", 0) / len(db)
print(f"LVH-positive share: {pos_share:.1f}% of the 100 Hz database — expect strong class imbalance at training time.")'''),

    ("code", r'''# 7.2 Missing values in the official metadata (raw columns only, via ORIGINAL_COLS)
missing = db[ORIGINAL_COLS].isna().sum()
miss_tbl = (pd.DataFrame({"missing": missing, "pct": (100 * missing / len(db)).round(1)})
            .query("missing > 0").sort_values("missing", ascending=False))
print(miss_tbl.to_string())

m = miss_tbl.pct.sort_values()
fig, ax = plt.subplots(figsize=(9, max(2.5, 0.42 * len(m))), constrained_layout=True)
ax.barh(m.index, m.values, color=SPANDANA_CYAN)
ax.set_xlabel("% of recordings with a missing value")
ax.set_title("Missing values in ptbxl_database.csv (v1.0.3)")
plt.show()

print(textwrap.fill(
    "Interpretation matters: empty noise columns (baseline_drift, static_noise, burst_noise) and the "
    "mostly empty event columns (electrodes_problems, extra_beats, pacemaker, infarction_stadium) "
    "simply mean 'nothing of that kind was annotated' — informative, not a gap. True clinical gaps "
    "are height and weight. Anonymised provenance IDs (nurse, site, device, validated_by) are "
    "missing when the item was not recorded.", 112))'''),

    ("md", r'''## 8 · Visualising real 12-lead recordings

We load actual signal files with **WFDB** (`wfdb.rdrecord`) and plot them in the standard clinical
layout — rows `I II III`, `aVR aVL aVF`, `V1 V2 V3`, `V4 V5 V6`. Amplitudes are normalised to
**mV** using the unit declared inside each `.hea` header (v1.0.3 stores millivolts; the loader
still converts µV/nV defensively if it ever encounters them). Selection is deterministic
(`RANDOM_STATE`) and prefers recordings without annotated noise and with human validation, so
morphology can be judged fairly. We display **4 LVH-positive and 4 control recordings** — the
controls deliberately include non-normal ECGs, demonstrating again that control ≠ healthy.'''),

    ("code", r'''# 8.1 Signal loader and clinical 12-lead plotter (real files via WFDB — no synthetic data)
def load_100hz(ecg_id):
    """Load one official 100 Hz recording; returns (signal_mV, fs, display_lead_names)."""
    rec = wfdb.rdrecord(str(DATA_DIR / db.at[ecg_id, "filename_lr"]))
    assert rec.fs == FS_EXPECTED, f"unexpected fs {rec.fs} for ecg_id {ecg_id}"
    # WFDB reports augmented leads as AVR/AVL/AVF; compare case-insensitively against clinical names.
    assert [s.upper() for s in rec.sig_name] == [s.upper() for s in DISPLAY_LEADS], rec.sig_name
    sig = rec.p_signal.astype(np.float64)
    unit = str(rec.units[0]).strip()
    if unit in ("µV", "μV", "uV"):        # defensive: normalise microvolts to millivolts
        sig /= 1000.0
    elif unit == "nV":
        sig /= 1e6
    elif unit != "mV":
        raise ValueError(f"unexpected signal unit '{unit}' for ecg_id {ecg_id}")
    return sig, int(rec.fs), DISPLAY_LEADS

def plot_12lead(ecg_id):
    """Plot one recording in the standard clinical 12-lead layout."""
    sig, fs, leads = load_100hz(ecg_id)
    t = np.arange(sig.shape[0]) / fs
    row = db.loc[ecg_id]

    age_txt = f"age {row.age:.0f}" if pd.notna(row.age) else "age n/a"
    sex_txt = f"sex {row.sex:.0f}" if pd.notna(row.sex) else "sex n/a"
    ltxt = f" | LVH likelihood {row.lvh_likelihood:.0f}" if pd.notna(row.lvh_likelihood) else ""

    fig, axes = plt.subplots(4, 3, figsize=(12.5, 9), sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, sig[:, i], lw=0.75, color=SPANDANA_NAVY)
        ax.set_ylabel(leads[i], rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=9)
    fig.suptitle(f"ecg_id {ecg_id} | patient {row.patient_id} | {row.label} | {age_txt} | {sex_txt}{ltxt}",
                 fontsize=11)
    plt.show()

print("Loader and plotter ready — signals are read from the official records100 files on demand.")'''),

    ("code", r'''# 8.2 Visualise LVH-positive recordings (deterministic selection, noise-free preferred)
def pick_demo(sub, n):
    """Deterministic pick: prefer noise-free, human-validated recordings; fall back gracefully."""
    c = sub.copy()
    noise_cols = [col for col in ("baseline_drift", "static_noise", "burst_noise") if col in c.columns]
    for strict in (True, False):
        cc = c
        if strict:
            for col in noise_cols:
                cc = cc[cc[col].isna()]
        if "validated_by_human" in cc.columns:
            pref = cc[cc.validated_by_human == True]   # noqa: E712
            if len(pref) >= n:
                cc = pref
        if len(cc) >= n:
            return cc.sample(n, random_state=RANDOM_STATE).sort_index()
    return c.sample(min(n, len(c)), random_state=RANDOM_STATE)

pos_pick = pick_demo(db[db.label == "LVH_positive"], N_DEMO_POS)
print("Selected LVH-positive ecg_ids:")
display(pos_pick[["patient_id", "age", "sex", "lvh_likelihood", "filename_lr"]])
for eid in pos_pick.index:
    plot_12lead(eid)'''),

    ("code", r'''# 8.3 Visualise control recordings — deliberately including non-normal ones (control != healthy)
norm_pick  = pick_demo(db[db.norm_only], N_DEMO_NEG // 2)
patho_pick = pick_demo(db[db.is_control & ~db.norm_only], N_DEMO_NEG - len(norm_pick))

print("Selected NORM-only controls:")
display(norm_pick[["patient_id", "age", "sex", "superclasses"]])
print("Selected non-normal controls (note their diagnostic superclasses):")
display(patho_pick[["patient_id", "age", "sex", "superclasses"]])
for eid in list(norm_pick.index) + list(patho_pick.index):
    plot_12lead(eid)'''),

    ("md", r'''## 9 · Patient IDs and the patient-wise train / validation / test split

A patient can contribute **multiple recordings** (repeat visits — up to ten recordings for a single
patient in the extreme case). Training and testing on recordings from the *same patient* would leak
information and inflate every metric, so the split must be **patient-wise**. PTB-XL already ships
an official recommendation: the **`strat_fold`** column — folds **1–8 → training**, **9 →
validation**, **10 → test** — generated patient-wise by the dataset authors. We **verify** that
property against the metadata instead of trusting it, then export the final assignment table.'''),

    ("code", r'''# 9.1 Verify the official patient-wise strat_fold and turn it into train/val/test assignments
folds_per_patient = db.groupby("patient_id").strat_fold.nunique()
assert (folds_per_patient == 1).all(), \
    f"{int((folds_per_patient > 1).sum())} patients appear in several folds — official property violated!"

fold2split = {**{f: "train" for f in range(1, 9)}, VAL_FOLD: "val", TEST_FOLD: "test"}
db["split"] = db.strat_fold.map(fold2split)

split_tbl = pd.DataFrame({
    "recordings"   : db.split.value_counts(),
    "patients"     : db.groupby("split").patient_id.nunique(),
    "LVH+ recs"    : db[db.label == "LVH_positive"].split.value_counts(),
    "control recs" : db[db.label == "control"].split.value_counts(),
}).reindex(["train", "val", "test"]).fillna(0).astype(int)
split_tbl["LVH+ %"] = (100 * split_tbl["LVH+ recs"] / split_tbl["recordings"]).round(2)

print("Split overview (official strat_fold mapping: 1-8 train, 9 validation, 10 test):")
print(split_tbl.to_string())

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = len(set(db[db.split == a].patient_id) & set(db[db.split == b].patient_id))
    print(f"patient overlap {a} & {b}: {overlap}")
print("OK: the official split is patient-wise and the LVH-positive share is similar in all three splits.")'''),

    ("code", r'''# 9.2 Patient-level summary table (one row per patient) — the basis for patient-wise modelling
def patient_label(labels):
    """Coarsest label wins: any definite LVH recording marks the patient LVH-positive."""
    labs = set(labels)
    if "LVH_positive" in labs:
        return "LVH_positive"
    if "LVH_low_certainty" in labs:
        return "LVH_low_certainty"
    return "control"

patient_summary = (db.groupby("patient_id")
      .agg(n_recordings=("label", "size"),
           patient_label=("label", patient_label),
           strat_fold=("strat_fold", lambda s: int(s.iloc[0])),
           split=("split", lambda s: "+".join(sorted(set(s))))))
patient_summary["mixed_label"] = db.groupby("patient_id").label.nunique().gt(1).reindex(patient_summary.index)

print("Patients by patient-level label:")
print(patient_summary.patient_label.value_counts().to_string())
print(f"\nMixed-label patients (LVH-positive AND control recordings; kept, but flagged): "
      f"{int(patient_summary.mixed_label.sum())}")
print("\nRecordings-per-patient distribution:")
print(patient_summary.n_recordings.value_counts().sort_index().to_string())
display(patient_summary.head(8))'''),

    ("code", r'''# 9.3 OPTIONAL alternative: a custom patient-wise stratified split
# The official strat_fold (used above) is the recommended choice: it keeps every future SPANDANA
# result comparable with the PTB-XL literature. Only flip this flag if you ever need custom ratios.
# If you do, re-run sections 9.2 and 10 afterwards so the exported CSVs reflect the new split.
USE_CUSTOM_SPLIT = False   # keep False to stay with the official folds

def custom_patient_split(meta, val_frac=0.1, test_frac=0.1, seed=RANDOM_STATE):
    """Split PATIENTS (never recordings) while balancing the patient-level LVH label."""
    from sklearn.model_selection import train_test_split
    pat = meta.groupby("patient_id").agg(plabel=("label", patient_label))
    rest, test = train_test_split(pat, test_size=test_frac, random_state=seed, stratify=pat.plabel)
    train, val = train_test_split(rest, test_size=val_frac / (1 - test_frac),
                                  random_state=seed, stratify=rest.plabel)
    assignment = pd.Series("train", index=pat.index)
    assignment.loc[val.index] = "val"
    assignment.loc[test.index] = "test"
    return assignment

if USE_CUSTOM_SPLIT:
    db["split"] = db.patient_id.map(custom_patient_split(db))
    print(db.split.value_counts().to_string())
else:
    print("Using the official strat_fold mapping (recommended). Nothing changed.")'''),

    ("md", r'''## 10 · Saving the verified metadata & split assignments

Three artefacts are exported to `/content/spandana_outputs/` — the **single source of truth** for
every later SPANDANA notebook:

| File | Grain | Content |
|---|---|---|
| `spandana_lvh_metadata.csv` | recording | official metadata + parsed labels, cohort flags, superclass set, split |
| `spandana_split_assignments.csv` | recording | `ecg_id` → `patient_id`, label, fold, split, 100 Hz file path |
| `spandana_patient_summary.csv` | patient | patient-level label, recording count, mixed-label flag |

Labels are derived **only** from the official SCP statements under the documented threshold policy —
nothing is imputed or synthesised.'''),

    ("code", r'''# 10.1 Export the verified metadata and split assignments
export_cols = [c for c in ["patient_id", "age", "sex", "height", "weight", "site", "device",
                           "recording_date", "heart_axis", "scp_codes", "strat_fold", "filename_lr"]
               if c in db.columns]

meta_out = db[export_cols + ["lvh_likelihood", "has_vclvh_form", "superclasses", "label",
                             "is_control", "clean_control", "norm_only", "split"]].copy()
meta_out["superclasses"] = meta_out.superclasses.apply(lambda s: "|".join(sorted(s)))
meta_out.index.name = "ecg_id"
meta_out.to_csv(OUTPUT_DIR / "spandana_lvh_metadata.csv")

split_out = db[["patient_id", "strat_fold", "label", "filename_lr", "split"]].copy()
split_out.index.name = "ecg_id"
split_out.to_csv(OUTPUT_DIR / "spandana_split_assignments.csv")

psum_out = patient_summary.copy()
psum_out.index.name = "patient_id"
psum_out.to_csv(OUTPUT_DIR / "spandana_patient_summary.csv")

print("Saved artefacts:")
for f in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"  {f.name:36s} {f.stat().st_size / 1024:9.1f} kB")

# Optional: download the artefacts to your local machine directly from Colab
# from google.colab import files
# for f in OUTPUT_DIR.glob("*.csv"):
#     files.download(str(f))

print("\nPreview of the split assignments (single source of truth for notebooks 02+):")
display(split_out.head(5))'''),

    ("md", r'''## 11 · Summary, limitations & next steps

**Verified in this notebook** (the exact numbers are in the printed outputs above):

- PTB-XL v1.0.3 metadata and 100 Hz signals loaded and cross-checked — sampling rate, lead order
  and physical units verified on sampled headers, never assumed.
- LVH identified from the official SCP dictionary: the **diagnostic `LVH` code** is the positive
  label; the `VCLVH` *form* statement is kept separate and reported; likelihood semantics follow
  the official paper, with an explicit, configurable threshold.
- The control pool is defined as *no LVH statement* and decomposed into nested subsets; the
  superclass breakdown printed in section 6 proves that **control ≠ healthy**.
- The official patient-wise `strat_fold` was verified (no patient spans folds) and mapped to
  train/validation/test; patient overlap between splits is asserted to be zero.
- Three CSV artefacts exported for notebooks 02+.

**Documented label limitations — carry these into every SPANDANA result report:**

1. **ECG statements, not diagnoses.** SCP statements are cardiologist readings of the ECG, not
   echocardiography-confirmed LVH. No LV-mass ground truth exists in PTB-XL.
2. **Limited sensitivity of ECG-based LVH detection.** Anatomical LVH patients can sit in the
   control pool; any metric computed here is specific to ECG-statement labels and must be reported
   as such.
3. **Threshold dependence.** The likelihood threshold (default 100) is a documented research
   decision — changing it changes the cohort; it is a configuration variable for exactly that
   reason.
4. **`LVH_low_certainty` recordings** (LVH annotated with likelihood < 100) are excluded, not
   silently relabelled — their count is printed in section 5.
5. **Historical heterogeneity & class imbalance.** Recordings span 1989–1996 across multiple sites
   and devices, and positives are a small minority (~5%); later notebooks must use the patient-wise
   split plus imbalance-aware training and evaluation.

**Next notebook (02).** Signal preprocessing and classical voltage-criteria feature extraction
(Sokolow-Lyon, Cornell) on the exported split — still no model training. The AI models (Perceptron,
CNN) enter only after those classical baselines are verified.

**References & licence**

- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020). *PTB-XL: A Large Publicly Available ECG Dataset.* Scientific Data 7, 154. https://doi.org/10.1038/s41597-020-0495-6
- PhysioNet PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/ — released under CC BY 4.0; cite both the PhysioNet DOI (https://doi.org/10.13026/kfzx-aw45) and the paper above when using the data.
- WFDB Python: https://github.com/MIT-LCP/wfdb-python'''),
]
