# CUDA Environment Setup Script for PowerShell
# Run this before building OpticXT to ensure CUDA is properly configured

Write-Host "🚀 Setting up CUDA environment for OpticXT..." -ForegroundColor Green

# Set CUDA environment variables
$env:CUDA_ROOT = "/usr/local/cuda-12.5"
$env:PATH = "$env:CUDA_ROOT/bin:$env:PATH"
$env:LD_LIBRARY_PATH = "$env:CUDA_ROOT/lib64:$env:LD_LIBRARY_PATH"

# Additional CUDA variables that may be needed
$env:CUDA_PATH = $env:CUDA_ROOT
$env:CUDA_TOOLKIT_ROOT_DIR = $env:CUDA_ROOT

# Verify CUDA installation
Write-Host "🔍 Verifying CUDA installation..." -ForegroundColor Yellow

if (Test-Path "$env:CUDA_ROOT/bin/nvcc") {
    Write-Host "✅ NVCC found at: $env:CUDA_ROOT/bin/nvcc" -ForegroundColor Green
    
    # Show CUDA version
    try {
        $cudaVersion = & "$env:CUDA_ROOT/bin/nvcc" --version 2>$null
        Write-Host "📋 CUDA Version:" -ForegroundColor Cyan
        Write-Host $cudaVersion -ForegroundColor White
    } catch {
        Write-Host "⚠️ Could not get CUDA version" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ NVCC not found at: $env:CUDA_ROOT/bin/nvcc" -ForegroundColor Red
    Write-Host "Please verify your CUDA installation path" -ForegroundColor Yellow
}

# Check for CUDA libraries
if (Test-Path "$env:CUDA_ROOT/lib64") {
    Write-Host "✅ CUDA libraries found at: $env:CUDA_ROOT/lib64" -ForegroundColor Green
} else {
    Write-Host "❌ CUDA libraries not found at: $env:CUDA_ROOT/lib64" -ForegroundColor Red
}

Write-Host "`n🎯 Environment variables set:" -ForegroundColor Cyan
Write-Host "   CUDA_ROOT: $env:CUDA_ROOT" -ForegroundColor White
Write-Host "   CUDA_PATH: $env:CUDA_PATH" -ForegroundColor White
Write-Host "   PATH includes: $env:CUDA_ROOT/bin" -ForegroundColor White
Write-Host "   LD_LIBRARY_PATH includes: $env:CUDA_ROOT/lib64" -ForegroundColor White

Write-Host "`n✨ CUDA environment setup complete!" -ForegroundColor Green
Write-Host "You can now run: cargo build --release" -ForegroundColor Cyan
