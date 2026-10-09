# Deepfake Detection v2.0 — Frontend Startup Script
Write-Host "=== Deepfake Detection Web UI v2.0 ===" -ForegroundColor Cyan

# Kill anything on port 8080
$existing = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Clearing existing process on port 8080..." -ForegroundColor Yellow
    Get-Process -Id $existing.OwningProcess | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

Set-Location (Join-Path $PSScriptRoot "frontend")
Write-Host "Starting Vite UI on port 8080 (available on Localhost & LAN for phone camera)..." -ForegroundColor Green
cmd /c "npm run dev -- --host"
