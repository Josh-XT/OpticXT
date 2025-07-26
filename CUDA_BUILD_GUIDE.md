# CUDA Build Setup Guide

This guide helps you set up the CUDA environment for building OpticXT with GPU acceleration.

## Quick Setup

### Option 1: Use the Setup Scripts

**For PowerShell (Windows/WSL):**
```powershell
./setup_cuda.ps1
cargo build --release
```

**For Bash/Zsh (Linux/macOS):**
```bash
source ./setup_cuda.sh
cargo build --release
```

### Option 2: Manual Environment Setup

**PowerShell:**
```powershell
$env:CUDA_ROOT = "/usr/local/cuda-12.5"
$env:PATH = "$env:CUDA_ROOT/bin:$env:PATH" 
$env:LD_LIBRARY_PATH = "$env:CUDA_ROOT/lib64:$env:LD_LIBRARY_PATH"
$env:CUDA_PATH = $env:CUDA_ROOT
$env:CUDA_TOOLKIT_ROOT_DIR = $env:CUDA_ROOT
```

**Bash/Zsh:**
```bash
export CUDA_ROOT="/usr/local/cuda-12.5"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"
export CUDA_PATH="$CUDA_ROOT"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
```

## VS Code Tasks

Use the predefined VS Code tasks for easy building:

1. **Setup CUDA Environment** - Sets up CUDA environment variables
2. **Build OpticXT** - Builds with CUDA support
3. **Build OpticXT (No CUDA)** - Builds CPU-only version

## Troubleshooting

### CUDA Not Found
- Verify CUDA installation: `nvcc --version`
- Check CUDA path: adjust `/usr/local/cuda-12.5` to match your installation
- Common CUDA paths:
  - `/usr/local/cuda-12.5`
  - `/usr/local/cuda`
  - `/opt/cuda`

### Build Failures
If CUDA build fails, try CPU-only mode:
```bash
cargo build --release --no-default-features
```

## Verification

After setting up CUDA, verify with:
```bash
echo $CUDA_ROOT
nvcc --version
cargo build --release
```

The build should complete without CUDA-related errors.
