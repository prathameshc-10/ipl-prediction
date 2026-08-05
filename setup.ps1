# Windows PowerShell Setup Script for Third Umpire IPL Predictor
$ErrorActionPreference = "Stop"

Write-Host "🏏 Starting Third Umpire Setup Script..." -ForegroundColor Cyan

# 1. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment 'venv'..." -ForegroundColor Green
    python -m venv venv
} else {
    Write-Host "📦 Virtual environment 'venv' already exists." -ForegroundColor Yellow
}

# 2. Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Green
# Using the full path to activate.ps1
. venv\Scripts\Activate.ps1

# 3. Upgrade pip and install requirements
Write-Host "📥 Installing dependencies from requirements.txt..." -ForegroundColor Green
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Set PYTHONPATH environment variable temporarily
$env:PYTHONPATH = (Get-Item .).FullName

# 5. Run training pipeline to generate model and visual plots
Write-Host "⚙️ Running model training pipeline..." -ForegroundColor Green
python src/model_training.py

# 6. Run unit tests
Write-Host "🧪 Running unit tests..." -ForegroundColor Green
python -m pytest

Write-Host "🎉 Setup complete! The model has been trained and validated successfully." -ForegroundColor Cyan
Write-Host "--------------------------------------------------------" -ForegroundColor Gray
Write-Host "To run the FastAPI backend server:" -ForegroundColor DarkYellow
Write-Host "  . venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  uvicorn backend.main:app --reload" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------" -ForegroundColor Gray
Write-Host "To run the Streamlit frontend:" -ForegroundColor DarkYellow
Write-Host "  . venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  streamlit run frontend/app.py" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------" -ForegroundColor Gray
