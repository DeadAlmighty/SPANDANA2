"""Patch SPANDANA_01_Dataset_Exploration.ipynb so the SAME file runs on
Google Colab AND locally in VS Code / Jupyter.

Changes (research logic untouched):
  cell 0  : intro markdown - artefact path + runtime notes now cover both runtimes
  cell 1  : markdown - mention local PTBXL_DIR option
  cell 2  : runtime auto-detection, BASE_DIR-relative paths, imports for the
            pure-Python downloader (shutil, urllib.request)
  cell 3  : wget when available (Colab/Linux), else resumable pure-Python
            Range-request download (Windows-safe)
  cell 29 : clarifying comment for local runs
"""
import json
from pathlib import Path

NB = Path("/home/z/my-project/download/SPANDANA_01_Dataset_Exploration.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def set_src(c, text):
    c["source"] = text.splitlines(keepends=True)
    c["outputs"] = []
    c["execution_count"] = None


# ---------------------------------------------------------------- cell 0 (md)
c = nb["cells"][0]
t = src(c)
t = t.replace(
    "**Artefacts produced** (in `/content/spandana_outputs/`):",
    "**Artefacts produced** (in `spandana_outputs/` next to the notebook — "
    "`/content/spandana_outputs/` on Colab):",
)
t = t.replace(
    "**Colab runtime notes.** A plain CPU runtime is sufficient. The first run downloads the official\n"
    "1.8 GB archive from PhysioNet and extracts the metadata plus the 100 Hz signals (~0.6 GB);\n"
    "later runs skip the download. Run the notebook top-to-bottom (\"Runtime → Run all\").",
    "**Runtime notes — Google Colab OR VS Code (local).** A plain CPU runtime is sufficient for this\n"
    "notebook; no GPU is needed. The same file auto-detects where it runs. The first run downloads the\n"
    "official 1.8 GB archive from PhysioNet and extracts the metadata plus the 100 Hz signals (~0.6 GB);\n"
    "later runs skip the download. Run the notebook top-to-bottom (Colab: \"Runtime → Run all\";\n"
    "VS Code: \"Run All\" — see the VS Code setup guide for the one-time environment steps).",
)
set_src(c, t)

# ---------------------------------------------------------------- cell 1 (md)
c = nb["cells"][1]
t = src(c)
t = t.replace(
    "If you already host\nPTB-XL on Google Drive, mount it, set `PTBXL_DIR` to that folder, and the download step is skipped.",
    "If you already host\nPTB-XL on Google Drive (Colab) or on your machine (VS Code), set `PTBXL_DIR` to that folder and the\n"
    "download step is skipped.",
)
set_src(c, t)

# ---------------------------------------------------------------- cell 2 (code)
c = nb["cells"][2]
t = src(c)
t = t.replace(
    "# 1.1 Install the waveform library (pandas / numpy / matplotlib are preinstalled in Colab)\n%pip install -q wfdb\n\n# 1.2 Imports\nimport ast, sys, textwrap, warnings, zipfile",
    "# 1.1 Install the waveform library (pandas / numpy / matplotlib are preinstalled on Colab;\n"
    "#      in VS Code they come from your virtual environment - see requirements-vscode.txt)\n"
    "%pip install -q wfdb\n\n"
    "# 1.2 Imports\nimport ast, shutil, sys, textwrap, urllib.request, warnings, zipfile",
)
t = t.replace(
    '''PTBXL_VERSION = "1.0.3"
ZIP_URL     = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"  # official PhysioNet get-zip target
ZIP_PATH    = Path("/content/ptbxl.zip")
EXTRACT_ROOT = Path("/content/ptbxl")
PTBXL_DIR   = None              # e.g. Path("/content/drive/MyDrive/ptb-xl-1.0.3") to skip the download
OUTPUT_DIR  = Path("/content/spandana_outputs")''',
    '''PTBXL_VERSION = "1.0.3"
ZIP_URL     = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"  # official PhysioNet get-zip target

# 1.3.1 Runtime detection — the SAME notebook runs on Google Colab AND locally in VS Code / Jupyter
IN_COLAB = "google.colab" in sys.modules
BASE_DIR = Path("/content") if IN_COLAB else Path.cwd()

ZIP_PATH     = BASE_DIR / "ptbxl.zip"            # 1.8 GB archive (removed after extraction unless KEEP_ZIP)
EXTRACT_ROOT = BASE_DIR / "ptbxl"                # extracted metadata + records100/ signals
PTBXL_DIR    = None              # already have PTB-XL on disk? point at the folder that contains
                                # ptbxl_database.csv, e.g. Path("D:/data/ptb-xl-1.0.3") - skips the download
OUTPUT_DIR   = BASE_DIR / "spandana_outputs"     # every CSV artefact is written here''',
)
t = t.replace(
    'KEEP_ZIP = False                # delete the 1.8 GB archive after extraction to save Colab disk',
    'KEEP_ZIP = False                # delete the 1.8 GB archive after extraction to save disk space',
)
t = t.replace(
    '''print("Python :", sys.version.split()[0])
print("Pandas :", pd.__version__, "| NumPy :", np.__version__, "| WFDB :", wfdb.__version__)''',
    '''print("Runtime :", "Google Colab" if IN_COLAB else f"Local (VS Code / Jupyter) - {BASE_DIR}")
print("Python  :", sys.version.split()[0])
print("Pandas  :", pd.__version__, "| NumPy :", np.__version__, "| WFDB :", wfdb.__version__)''',
)
set_src(c, t)

# ---------------------------------------------------------------- cell 3 (code)
c = nb["cells"][3]
t = src(c)
t = t.replace(
    '''    if not ZIP_PATH.exists():
        print(f"Downloading PTB-XL v{PTBXL_VERSION} (1.8 GB) from the official PhysioNet mirror ...")
        !wget -c --progress=dot:giga -O {ZIP_PATH} {ZIP_URL}
        assert ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1e9, "Download failed - retry, or fetch the zip manually from physionet.org."''',
    '''    if not ZIP_PATH.exists():
        print(f"Downloading PTB-XL v{PTBXL_VERSION} (1.8 GB) from the official PhysioNet mirror ...")
        if shutil.which("wget"):
            # Colab / Linux / macOS with wget: resumable command-line download
            !wget -c --progress=dot:giga -O {ZIP_PATH} {ZIP_URL}
        else:
            # Windows and systems without wget: resumable pure-Python download (Range requests)
            done = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with urllib.request.urlopen(urllib.request.Request(ZIP_URL, headers=headers)) as resp:
                if done and resp.status != 206:      # server ignored the Range header - restart cleanly
                    done = 0
                total = done + int(resp.headers.get("Content-Length") or 0)
                with open(ZIP_PATH, "ab" if done else "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)   # 1 MiB at a time
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if done % (64 << 20) < (1 << 20):
                            print(f"  {done / 1e9:5.2f} / {total / 1e9:.2f} GB", flush=True)
            print()
        assert ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1e9, "Download failed - retry, or fetch the zip manually from physionet.org."''',
)
set_src(c, t)

# --------------------------------------------------------------- cell 29 (code)
c = nb["cells"][29]
t = src(c)
t = t.replace(
    "# Optional: download the artefacts to your local machine directly from Colab\n# from google.colab import files",
    "# Optional: on Colab you can pull the artefacts straight to your machine:\n"
    "# from google.colab import files",
)
set_src(c, t)

# ------------------------------------------------- cell 2 fix: define before use
# The Runtime print references IN_COLAB, so runtime detection must come BEFORE it.
c = nb["cells"][2]
t = src(c)
detect_old = (
    "# 1.3.1 Runtime detection — the SAME notebook runs on Google Colab AND locally in VS Code / Jupyter\n"
    'IN_COLAB = "google.colab" in sys.modules\n'
    'BASE_DIR = Path("/content") if IN_COLAB else Path.cwd()\n'
)
detect_new = (
    "# 1.2.1 Runtime detection — the SAME notebook runs on Google Colab AND locally in VS Code / Jupyter\n"
    'IN_COLAB = "google.colab" in sys.modules\n'
    'BASE_DIR = Path("/content") if IN_COLAB else Path.cwd()\n'
)
if detect_old in t and 'print("Runtime :"' in t:
    t = t.replace("\n" + detect_old, "", 1)
    t = t.replace('print("Runtime :"', detect_new + "\nprint(\"Runtime :\"", 1)
    set_src(c, t)
    print("Moved runtime detection above the Runtime print.")
elif detect_new in t:
    print("Runtime detection already correctly placed.")

# ------------------------------------------------------------------ validate
# Replace IPython magics (%pip, !wget) with `pass` for validation - they are valid
# inside Jupyter but not pure Python, and may sit inside if-blocks.
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        py = "\n".join(
            (l[: len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("%", "!")) else l
            for l in src(cell).splitlines()
        )
        compile(py, f"<cell {i}>", "exec")
print("All code cells compile.")
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Patched:", NB)
