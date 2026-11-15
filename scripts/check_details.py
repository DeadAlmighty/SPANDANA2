"""Check VCLVH form statement, control-pool composition, and positivity per split."""
import ast

import pandas as pd

pd.set_option("display.width", 220)

db = pd.read_csv("/tmp/ptbxl/ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv("/tmp/ptbxl/scp_statements.csv", index_col=0)

print("=== LVH vs VCLVH rows in scp_statements ===")
print(scp.loc[["LVH", "VCLVH"]][["description", "diagnostic", "form", "rhythm", "diagnostic_class"]].to_string())

db["scp_dict"] = db.scp_codes.apply(ast.literal_eval)
db["lvh"] = db.scp_dict.apply(lambda d: d.get("LVH"))
db["has_vclvh"] = db.scp_dict.apply(lambda d: "VCLVH" in d)
print("\nrecordings carrying VCLVH (form statement):", int(db.has_vclvh.sum()))
print("recordings with LVH stmt AND VCLVH:", int((db.has_vclvh & db.lvh.notna()).sum()))
print("recordings with VCLVH but no LVH stmt:", int((db.has_vclvh & db.lvh.isna()).sum()))

diag = scp[scp.diagnostic == 1]
db["sc"] = db.scp_dict.apply(lambda d: {diag.at[c, "diagnostic_class"] for c in d if c in diag.index})

pos = db.lvh >= 100
unc = db.lvh.notna() & (db.lvh < 100)
ctrl = db.lvh.isna()
print("\npositives (LVH=100):", int(pos.sum()), "| low-certainty LVH:", int(unc.sum()), "| controls:", int(ctrl.sum()))

ctrl_df = db[ctrl]
print("controls carrying other HYP-class codes (RVH/LAO/RAO/SEHYP):",
      int(ctrl_df.sc.apply(lambda s: "HYP" in s).sum()))
print("controls carrying MI:", int(ctrl_df.sc.apply(lambda s: "MI" in s).sum()))
print("controls carrying STTC:", int(ctrl_df.sc.apply(lambda s: "STTC" in s).sum()))
print("controls carrying CD:", int(ctrl_df.sc.apply(lambda s: "CD" in s).sum()))
print("NORM-only controls:", int(ctrl_df.sc.apply(lambda s: s == {"NORM"}).sum()))
print("controls with EMPTY diagnostic superclass set:", int(ctrl_df.sc.apply(lambda s: len(s) == 0).sum()))

db["split"] = db.strat_fold.map(lambda f: "train" if f <= 8 else ("val" if f == 9 else "test"))
print("\nrecordings per split:", db.split.value_counts().to_dict())
print("positives per split:", db[pos].split.value_counts().to_dict())
print("positive share per split:")
print((100 * db.groupby("split").apply(lambda g: (g.lvh >= 100).mean(), include_groups=False)).round(2).to_string())

mixed = db.groupby("patient_id").apply(
    lambda g: bool({"pos", "ctrl"} <= set(g.lvh.notna().map({True: "pos", False: "ctrl"}))),
    include_groups=False)
print("\npatients with both positive and control recordings:", int(mixed.sum()))
unc_pat = db[unc].patient_id.nunique()
print("patients with low-certainty LVH recordings:", unc_pat)
print("\nDONE")
