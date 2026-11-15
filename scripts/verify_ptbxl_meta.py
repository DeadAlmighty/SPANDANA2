"""Verify PTB-XL metadata facts before writing the SPANDANA_01 notebook.

Runs against the REAL official CSVs downloaded from physionet.org.
"""
import ast
from collections import Counter

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

db = pd.read_csv("/tmp/ptbxl/ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv("/tmp/ptbxl/scp_statements.csv", index_col=0)

print("=== ptbxl_database.csv ===")
print("shape:", db.shape)
print("columns:", list(db.columns))
print("index dtype/name:", db.index.dtype, db.index.name, "| first ids:", list(db.index[:3]))

print("\nsex counts:", db.sex.value_counts(dropna=False).to_dict())
print("sex==0 share:", round((db.sex == 0).mean(), 4), "(paper documents 52% male)")
print("strat_fold counts:", db.strat_fold.value_counts().sort_index().to_dict())
print("unique patients:", db.patient_id.nunique())
print("recordings per patient:", db.groupby("patient_id").size().value_counts().to_dict())
print("max folds per patient:", db.groupby("patient_id").strat_fold.nunique().max())

print("\nmissing values per column:")
miss = db.isna().sum()
print(miss[miss > 0].sort_values(ascending=False).to_string())

for col in ("validated_by_human", "second_opinion"):
    if col in db.columns:
        print(f"\n{col}:", db[col].value_counts(dropna=False).to_dict())

print("\n=== scp_statements.csv ===")
print("shape:", scp.shape, "| columns:", list(scp.columns))
print("diagnostic:", scp.diagnostic.value_counts(dropna=False).to_dict())
print("diagnostic_class:", scp.diagnostic_class.value_counts(dropna=False).to_dict())

hyp = scp[scp.diagnostic_class == "HYP"]
print("\nAll HYP-class codes:")
print(hyp[["description", "diagnostic", "diagnostic_subclass"]].to_string())

print("\n=== LVH extraction ===")
db["scp_dict"] = db.scp_codes.apply(ast.literal_eval)
db["lvh"] = db.scp_dict.apply(lambda d: d.get("LVH"))
print("recordings with any LVH statement:", int(db.lvh.notna().sum()))
print("LVH likelihood distribution:", db.lvh.value_counts(dropna=False).to_dict())
print("definite LVH (>=100):", int((db.lvh >= 100).sum()))
print("definite LVH patients:", db.loc[db.lvh >= 100, "patient_id"].nunique())
print("LVH stmt but likelihood==0:", int((db.lvh == 0).sum()))

diag = scp[(scp.diagnostic == 1)]
print("\ndiagnostic==1 superclass counts:", diag.diagnostic_class.value_counts(dropna=False).to_dict())
print("diagnostic==1 rows with NaN class:", int(diag.diagnostic_class.isna().sum()))

cls_map = diag.diagnostic_class.to_dict()
allc = Counter()
for d in db.scp_dict:
    for c in d:
        if c in cls_map:
            allc[cls_map[c]] += 1
print("superclass occurrences over recordings:", dict(allc))

lv = Counter()
for d in db.scp_dict:
    lv.update(d.values())
print("\nall statement likelihood values:", dict(sorted(lv.items())))

print("\nregex-based LVH code identification (mirrors notebook logic):")
hyp_related = scp[scp.description.str.contains("hypertrophy|enlargement|overload", case=False, na=False)]
lvh_codes = hyp_related[hyp_related.description.str.contains("left ventricular", case=False, na=False)].index.tolist()
print("  'left ventricular' description codes:", lvh_codes)
print("  other hypertrophy/enlargement/overload codes:", sorted(set(hyp_related.index) - set(lvh_codes)))

print("\nfilename_lr sample:", db.filename_lr.iloc[0], "| filename_hr sample:", db.filename_hr.iloc[0])
print("age missing:", int(db.age.isna().sum()), "| sex missing:", int(db.sex.isna().sum()))
print("\nDONE")
