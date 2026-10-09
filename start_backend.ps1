# Deepfake Detection v2.0 — Backend Startup Script
Write-Host "=== Deepfake Detection Engine v2.0 ===" -ForegroundColor Cyan

# Kill anything on port 8000
$existing = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Clearing existing process on port 8000..." -ForegroundColor Yellow
    Get-Process -Id $existing.OwningProcess | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

$python = Join-Path $PSScriptRoot "backend\.venv\Scripts\python.exe"
if (-Not (Test-Path $python)) {
    Write-Host "ERROR: Python environment not found at $python. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

Set-Location (Join-Path $PSScriptRoot "backend")
Write-Host "Starting DeepGuard FastAPI backend on http://localhost:8000 ..." -ForegroundColor Green
& $python -m uvicorn main:app --host 0.0.0.0 --port 8000
