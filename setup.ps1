# ============================================================================
#  SPANDANA - one-time environment setup for VS Code on Windows
#  Usage:  .\setup.ps1            (from the project root, in any terminal)
#  Re-run safely at any time - it skips everything already done.
# ============================================================================
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- Python ----
Step "Checking Python installation"
$pyExe = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $pyExe = "python" }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $pyExe = "py" }
if (-not $pyExe) {
    Write-Host "Python not found. Install Python 3.10-3.12 from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "IMPORTANT: tick 'Add python.exe to PATH' in the installer." -ForegroundColor Red
    exit 1
}
$ver = if ($pyExe -eq "py") { & py -3 --version } else { & python --version }
Ok "Found: $ver"

# ---------------------------------------------------------------- venv ------
Step "Creating .venv (if missing)"
if (-not (Test-Path "$root\.venv")) {
    if ($pyExe -eq "py") { & py -3 -m venv "$root\.venv" } else { & python -m venv "$root\.venv" }
    Ok ".venv created"
} else { Ok ".venv already exists" }

Step "Installing Python dependencies (requirements-vscode.txt)"
& "$root\.venv\Scripts\python.exe" -m pip install -q --upgrade pip
& "$root\.venv\Scripts\python.exe" -m pip install -q -r "$root\requirements-vscode.txt"
Ok "Python packages installed (torch on Windows = CPU build, ideal for SPANDANA)"

# ---------------------------------------------------------------- Node ------
Step "Checking Node.js and npm"
$node = Get-Command node -ErrorAction SilentlyContinue
$npm  = Get-Command npm  -ErrorAction SilentlyContinue
if (-not $node -or -not $npm) {
    Warn "Node.js/npm not found - the frontend needs Node.js 18+ (https://nodejs.org/)."
} else {
    Ok "Found: node $(node --version), npm $(npm --version)"
    Step "Installing frontend dependencies"
    if (-not (Test-Path "$root\node_modules")) {
        npm install
        Ok "npm packages installed"
    } else { Ok "node_modules already present (run 'npm install' manually if package.json changed)" }
}

# ------------------------------------------------------------- folders ------
Step "Creating data / artifact directories"
foreach ($d in @("data\ptbxl", "data\metadata", "data\preprocessed",
                 "data\reports", "data\cnn_runs", "backend\models")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $d) | Out-Null
}
Ok "data/, backend/models/ ready"

# ---------------------------------------------------------------- done ------
Write-Host ""
Write-Host "Setup complete. Next steps:" -ForegroundColor Magenta
Write-Host "  1) .\run_training.ps1 download      # fetch PTB-XL (resumable)"
Write-Host "  2) .\run_training.ps1 pipeline      # inspect -> labels -> splits -> features"
Write-Host "  3) .\run_training.ps1 perceptron    # train + evaluate the Perceptron"
Write-Host "  4) .\run_training.ps1 cnn           # train + evaluate the CNN (CPU ok)"
Write-Host "  5) .\run_training.ps1 evaluate      # 4-method comparison"
Write-Host "  6) .\run_backend.ps1                # FastAPI on :8000"
Write-Host "  7) .\run_frontend.ps1               # Next.js on :3000"
