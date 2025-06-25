# Setup script for NeuroRiskLogic deployment (PowerShell)

Write-Host "Setting up NeuroRiskLogic deployment..." -ForegroundColor Green

# Check if .env exists, if not copy from template
if (-Not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.template" ".env"
    Write-Host "Please edit .env file with your actual values before proceeding." -ForegroundColor Yellow
    Write-Host "Press any key to continue after editing..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
$directories = @("../data/models", "../data/synthetic", "../logs", "../reports")
foreach ($dir in $directories) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Check if model file exists
if (-Not (Test-Path "../data/models/model_current.pkl")) {
    Write-Host "WARNING: No trained model found at ../data/models/model_current.pkl" -ForegroundColor Red
    Write-Host "You need to train a model first using: python scripts/train_model.py" -ForegroundColor Red
}

# Build images
Write-Host "Building Docker images..." -ForegroundColor Yellow
docker-compose build

# Start services
Write-Host "Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to be ready
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check health
Write-Host "Checking service health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
    Write-Host "API is healthy!" -ForegroundColor Green
} catch {
    Write-Host "API not yet ready" -ForegroundColor Red
}

Write-Host "`nDeployment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f api"
Write-Host "  docker-compose logs -f retraining"
Write-Host ""
Write-Host "To stop services:" -ForegroundColor Cyan
Write-Host "  docker-compose down"