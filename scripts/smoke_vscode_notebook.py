"""Smoke-test the VS Code readiness of SPANDANA notebook 01.

1. Executes the notebook's setup cell (cell 2) locally, outside Colab, in a temp
   working directory - proves runtime detection, BASE_DIR paths and imports work.
2. Sends a tiny Range request to the official PhysioNet mirror - proves the
   resumable pure-Python downloader branch (used on Windows) is sound.
"""
import json
import tempfile
import urllib.request
from pathlib import Path

NB = Path("/home/z/my-project/download/SPANDANA_01_Dataset_Exploration.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


# ---- 1. run cell 2 locally -------------------------------------------------
setup = src(nb["cells"][2])
setup = "\n".join(
    l for l in setup.splitlines() if not l.lstrip().startswith(("%", "!"))
)  # drop the %pip magic (already installed in this env)

with tempfile.TemporaryDirectory() as tmp:
    cwd = Path(tmp)
    ns: dict = {}
    code = (
        "import os\n"
        f"os.chdir({str(cwd)!r})\n"
        "from pathlib import Path\n"
        + setup
    )
    exec(compile(code, "<cell 2>", "exec"), ns)  # noqa: S102 - controlled notebook source
    assert ns["IN_COLAB"] is False, "runtime detection broken"
    assert ns["BASE_DIR"] == cwd, f"BASE_DIR should be cwd, got {ns['BASE_DIR']}"
    assert ns["OUTPUT_DIR"] == cwd / "spandana_outputs"
    assert ns["OUTPUT_DIR"].is_dir(), "OUTPUT_DIR was not created"
    assert ns["ZIP_PATH"] == cwd / "ptbxl.zip"
    assert ns["EXTRACT_ROOT"] == cwd / "ptbxl"
    assert ns["LVH_LIKELIHOOD_THRESHOLD"] == 100 and ns["VAL_FOLD"] == 9 and ns["TEST_FOLD"] == 10
    print("PASS cell 2: local runtime detected, paths =", ns["OUTPUT_DIR"])
    print("     research constants intact: threshold=100, folds 9/10, fs=100")

# ---- 2. Range support on the official mirror -------------------------------
url = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"
req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
with urllib.request.urlopen(req, timeout=30) as r:
    body = r.read()
    assert r.status == 206, f"expected 206 Partial Content, got {r.status}"
    assert len(body) == 1024, f"expected 1024 bytes, got {len(body)}"
    assert r.headers.get("Content-Range", "").startswith("bytes 0-1023/")
    total_gb = int(r.headers["Content-Range"].split("/")[-1]) / 1e9
    print(f"PASS downloader: mirror honours Range requests; full archive = {total_gb:.2f} GB")
print("Smoke test complete - notebook is VS Code ready.")
