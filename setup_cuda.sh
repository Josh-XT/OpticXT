#!/bin/bash
# CUDA Environment Setup Script for Bash/Zsh
# Run this before building OpticXT to ensure CUDA is properly configured

echo "🚀 Setting up CUDA environment for OpticXT..."

# Set CUDA environment variables
export CUDA_ROOT="/usr/local/cuda-12.5"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"

# Additional CUDA variables that may be needed
export CUDA_PATH="$CUDA_ROOT"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"

# Verify CUDA installation
echo "🔍 Verifying CUDA installation..."

if [ -f "$CUDA_ROOT/bin/nvcc" ]; then
    echo "✅ NVCC found at: $CUDA_ROOT/bin/nvcc"
    
    # Show CUDA version
    echo "📋 CUDA Version:"
    "$CUDA_ROOT/bin/nvcc" --version 2>/dev/null || echo "⚠️ Could not get CUDA version"
else
    echo "❌ NVCC not found at: $CUDA_ROOT/bin/nvcc"
    echo "Please verify your CUDA installation path"
fi

# Check for CUDA libraries
if [ -d "$CUDA_ROOT/lib64" ]; then
    echo "✅ CUDA libraries found at: $CUDA_ROOT/lib64"
else
    echo "❌ CUDA libraries not found at: $CUDA_ROOT/lib64"
fi

echo ""
echo "🎯 Environment variables set:"
echo "   CUDA_ROOT: $CUDA_ROOT"
echo "   CUDA_PATH: $CUDA_PATH"
echo "   PATH includes: $CUDA_ROOT/bin"
echo "   LD_LIBRARY_PATH includes: $CUDA_ROOT/lib64"

echo ""
echo "✨ CUDA environment setup complete!"
echo "You can now run: cargo build --release"

# Export variables for the current session
echo ""
echo "💡 To make these changes persistent, add the following to your shell profile:"
echo "   export CUDA_ROOT=\"/usr/local/cuda-12.5\""
echo "   export PATH=\"\$CUDA_ROOT/bin:\$PATH\""
echo "   export LD_LIBRARY_PATH=\"\$CUDA_ROOT/lib64:\$LD_LIBRARY_PATH\""
