# ============================================================================
#  SPANDANA - start the FastAPI backend on http://localhost:8000
#  Usage:  .\run_backend.ps1 [-Port 8000] [-NoReload]
# ============================================================================
param(
    [int]$Port = 8000,
    [switch]$NoReload
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not (Test-Path "$root\.venv\Scripts\python.exe")) {
    Write-Host ".venv not found - run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

Set-Location "$root\backend"
$reload = if ($NoReload) { "" } else { "--reload" }
Write-Host "Starting SPANDANA backend on http://localhost:$Port  (docs at /docs)" -ForegroundColor Cyan
& "$root\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port $Port
