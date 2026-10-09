# Deepfake Detection v2.0 — Setup Script
# Run once to install all dependencies

Write-Host "=== Setting up Deepfake Detection v2.0 ===" -ForegroundColor Cyan

$root = $PSScriptRoot
$backendPath = Join-Path $root "backend"
$frontendPath = Join-Path $root "frontend"

# ── Backend Setup ──────────────────────────────────────────────────────────
Write-Host "`n[1/3] Setting up Python backend..." -ForegroundColor Yellow
Set-Location $backendPath

if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Installing Python packages (this may take 5-10 minutes)..."
.\.venv\Scripts\pip install --upgrade pip --quiet
.\.venv\Scripts\pip install fastapi "uvicorn[standard]" python-multipart aiofiles `
    timm transformers Pillow "opencv-python-headless" numpy `
    librosa soundfile scikit-learn scipy requests torch torchvision

Write-Host "✅ Backend dependencies installed" -ForegroundColor Green

# ── Frontend Setup ─────────────────────────────────────────────────────────
Write-Host "`n[2/3] Setting up React frontend..." -ForegroundColor Yellow
Set-Location $frontendPath

if (-Not (Test-Path "node_modules")) {
    Write-Host "Installing npm packages..."
    npm install
} else {
    Write-Host "node_modules already exists, running npm install to update..."
    npm install
}
Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green

# ── Models Check ──────────────────────────────────────────────────────────
Write-Host "`n[3/3] Checking models directory..." -ForegroundColor Yellow
$modelsPath = Join-Path $backendPath "models"
if (-Not (Test-Path (Join-Path $modelsPath "efficientnet_deepfake.pt"))) {
    Write-Host "⚠️  Custom trained model not found at backend/models/efficientnet_deepfake.pt" -ForegroundColor Yellow
    Write-Host "    Run the Google Colab notebook first, then copy the downloaded .pt file here." -ForegroundColor Yellow
    Write-Host "    The ViT model will be auto-downloaded on first run." -ForegroundColor Cyan
} else {
    Write-Host "✅ Custom EfficientNet model found" -ForegroundColor Green
}

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "To start the system:"
Write-Host "  Terminal 1 (backend): .\start_backend.ps1"
Write-Host "  Terminal 2 (frontend): cd frontend; npm run dev"
Write-Host "  Browser: http://localhost:5173"
Write-Host "  Phone: Scan QR code shown in the app"
