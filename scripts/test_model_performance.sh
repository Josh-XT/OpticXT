#!/bin/bash

echo "🧪 Testing ISQ vs UQFF model loading performance"
echo "=============================================="

echo ""
echo "📊 System Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

echo "🚀 Test 1: ISQ Model Loading"
echo "----------------------------"
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export RUST_LOG=info

echo "Starting ISQ model test..."
timeout 120 cargo run -- --model-path "google/gemma-2-2b-it" --test-simple

echo ""
echo "🚀 Test 2: UQFF Model Loading (current)"
echo "---------------------------------------"
echo "Starting UQFF model test..."
timeout 120 cargo run -- --test-simple

echo ""
echo "✅ Test completed!"
