"""Build SPANDANA_01_Dataset_Exploration.ipynb from cell definitions, then validate it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nbformat as nbf

from spandana01_cells_a import CELLS_A
from spandana01_cells_b import CELLS_B

OUT = Path("/home/z/my-project/download/SPANDANA_01_Dataset_Exploration.ipynb")

cells = CELLS_A + CELLS_B

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "colab": {"provenance": []},
}

n_md = n_code = 0
for i, (kind, src) in enumerate(cells):
    if kind == "md":
        cell = nbf.v4.new_markdown_cell(src)
        n_md += 1
    else:
        cell = nbf.v4.new_code_cell(src)
        n_code += 1
    cell["id"] = f"spandana-01-cell-{i:02d}"
    nb.cells.append(cell)

nb["nbformat"] = 4
nb["nbformat_minor"] = 5

# ---- validation -------------------------------------------------------------
nbf.validate(nb)

problems = []
for i, cell in enumerate(nb.cells):
    src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
    if cell.cell_type == "code":
        # IPython magics (%pip, !wget) are valid in Jupyter but not plain Python — strip before compile
        py = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!")))
        try:
            compile(py, f"cell_{i:02d}", "exec")
        except SyntaxError as e:
            problems.append(f"cell {i:02d}: SyntaxError: {e}")
    for token in ("TODO", "FIXME", "XXX", "placeholder", "【", "待补充"):
        if token.lower() in src.lower():
            problems.append(f"cell {i:02d}: leftover token '{token}'")

total_lines = sum(len((c.source if isinstance(c.source, str) else "".join(c.source)).splitlines())
                  for c in nb.cells)
code_lines = sum(len((c.source if isinstance(c.source, str) else "".join(c.source)).splitlines())
                 for c in nb.cells if c.cell_type == "code")

if problems:
    print("VALIDATION PROBLEMS:")
    for p in problems:
        print(" -", p)
    sys.exit(1)

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)

print(f"Notebook written : {OUT}")
print(f"Size             : {OUT.stat().st_size / 1024:.1f} kB")
print(f"Cells            : {len(nb.cells)} ({n_md} markdown, {n_code} code)")
print(f"Lines            : {total_lines} total ({code_lines} in code cells)")
print("nbformat validate: OK | all code cells compile: OK | no leftover placeholders: OK")
print("\nCell map:")
for i, cell in enumerate(nb.cells):
    first = (cell.source if isinstance(cell.source, str) else "".join(cell.source)).strip().splitlines()[0]
    print(f"  [{i:02d}] {cell.cell_type:4s} | {first[:86]}")
