#!/bin/bash

# CUDA Debug Script for OpticXT
echo "🔍 CUDA Environment Debug Script"
echo "================================"

echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "🔧 CUDA Environment Variables:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "CUDA_LAUNCH_BLOCKING: ${CUDA_LAUNCH_BLOCKING:-not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"

echo ""
echo "📁 CUDA Library Paths:"
find /usr/local/cuda* -name "libcudart.so*" 2>/dev/null || echo "No CUDA libraries found in /usr/local/cuda*"
find /usr/lib -name "libcudart.so*" 2>/dev/null || echo "No CUDA libraries found in /usr/lib"

echo ""
echo "🚀 Running OpticXT with CUDA environment..."
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export RUST_LOG=debug

echo "Environment set:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "RUST_LOG=$RUST_LOG"

echo ""
echo "Starting OpticXT test..."
cargo run -- --test-simple
