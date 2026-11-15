# ============================================================================
#  SPANDANA - start the existing Next.js frontend
#  Usage:  .\run_frontend.ps1
#  (uses the project's existing npm scripts - nothing is rebuilt or renamed)
# ============================================================================
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npm) {
    Write-Host "npm not found - install Node.js 18+ from https://nodejs.org/" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$root\node_modules")) {
    Write-Host "node_modules missing - installing..." -ForegroundColor Cyan
    npm install
}

# Tell the frontend where the FastAPI backend lives (see .env.example).
if (-not (Test-Path "$root\.env.local") -and (Test-Path "$root\.env.example")) {
    Copy-Item "$root\.env.example" "$root\.env.local"
    Write-Host "Created .env.local from .env.example (NEXT_PUBLIC_FASTAPI_URL=http://localhost:8000)" -ForegroundColor Cyan
}

Write-Host "Starting SPANDANA frontend on http://localhost:3000" -ForegroundColor Cyan
npm run dev
