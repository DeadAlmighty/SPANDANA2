"""Smoke-test WFDB loading + 12-lead plotting on real PTB-XL sample records."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

sys.path.insert(0, "/tmp/ptbxl")
BASE = "/tmp/ptbxl/"

rec = wfdb.rdrecord(BASE + "records100/00000/00001_lr")
print("fs:", rec.fs, "| n_sig:", rec.n_sig, "| sig_len:", rec.sig_len)
print("sig_name:", rec.sig_name)
print("units:", rec.units)
sig = rec.p_signal.astype(np.float64)
print("shape:", sig.shape, "| min/max:", round(float(np.nanmin(sig)), 3), round(float(np.nanmax(sig)), 3))

hdr = wfdb.rdheader(BASE + "records100/01000/01500_lr")
print("header-only read ok | fs:", hdr.fs, "| units:", hdr.units, "| leads:", hdr.sig_name[:4], "...")
hdr2 = wfdb.rdheader(BASE + "records100/02000/02500_lr")
print("header-only read ok | fs:", hdr2.fs, "| units:", hdr2.units)

# --- notebook-style plot: 4x3 grid, standard clinical layout ---
EXPECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
assert [s.upper() for s in rec.sig_name] == [s.upper() for s in EXPECTED_LEADS], rec.sig_name
t = np.arange(sig.shape[0]) / rec.fs
fig, axes = plt.subplots(4, 3, figsize=(12.5, 9), sharex=True, sharey=True, constrained_layout=True)
for i, ax in enumerate(axes.flat):
    ax.plot(t, sig[:, i], lw=0.75, color="#0B1728")
    ax.set_ylabel(rec.sig_name[i], rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
for ax in axes[-1]:
    ax.set_xlabel("time (s)", fontsize=9)
fig.suptitle("ecg_id 1 | test plot from real PTB-XL record", fontsize=11)
fig.savefig("/tmp/test_12lead.png", dpi=70)
print("plot saved OK -> /tmp/test_12lead.png")
print("DONE")
