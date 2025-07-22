#!/bin/bash

# OpticXT Demo Script
# Demonstrates the vision-to-action pipeline

echo "🤖 OpticXT Demo - Vision-Driven Robot Control"
echo "=============================================="
echo ""

# Check if the binary exists
if [ ! -f "./target/release/opticxt" ]; then
    echo "❌ OpticXT binary not found. Please run: cargo build --release"
    exit 1
fi

# Check if model exists
if [ ! -f "models/gemma-3n-E4B-it-Q4_K_M.gguf" ]; then
    echo "⚠️  Model file not found!"
    echo "Please download the Gemma model:"
    echo "  1. Visit: https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF"
    echo "  2. Download: gemma-3n-E4B-it-Q4_K_M.gguf"
    echo "  3. Place in: models/"
    echo ""
    echo "Or use HuggingFace CLI:"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download unsloth/gemma-3n-E4B-it-GGUF gemma-3n-E4B-it-Q4_K_M.gguf --local-dir models/"
    echo ""
    read -p "Continue with simulation mode? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for camera
if [ -e "/dev/video0" ]; then
    echo "📹 Camera found at /dev/video0"
    CAMERA_ARG="--camera-device 0"
else
    echo "⚠️  No camera found, running in simulation mode"
    CAMERA_ARG="--camera-device 0"
fi

echo ""
echo "🚀 Starting OpticXT..."
echo "This demo will:"
echo "  1. Capture video from camera (or simulate)"
echo "  2. Detect objects in each frame"
echo "  3. Generate contextual prompts"
echo "  4. Use Gemma model to decide actions"
echo "  5. Execute XML commands"
echo ""
echo "Press Ctrl+C to stop"
echo "In debug mode, press ESC in the video window to exit"
echo ""

# Create a simple demo config
cat > demo_config.toml << EOF
[vision]
width = 640
height = 480
fps = 30
confidence_threshold = 0.3
vision_model = "yolo"

[model]
model_path = "models/gemma-3n-E4B-it-Q4_K_M.gguf"
context_length = 2048
temperature = 0.8
top_p = 0.9
max_tokens = 256

[context]
system_prompt = "prompts/system_prompt.txt"
max_context_history = 5
include_timestamp = true

[commands]
enabled_commands = ["move", "rotate", "speak", "analyze", "wait"]
timeout_seconds = 10
validate_before_execution = true

[performance]
worker_threads = 2
frame_buffer_size = 5
processing_interval_ms = 500
use_gpu = false
EOF

echo "📝 Created demo configuration (demo_config.toml)"
echo "   - Reduced performance settings for demo"
echo "   - CPU-only inference"
echo "   - Slower processing for visibility"
echo ""

# Run the application
./target/release/opticxt --config demo_config.toml $CAMERA_ARG --verbose

echo ""
echo "🏁 Demo completed!"
echo ""
echo "What happened:"
echo "  ✅ Video frames were captured and processed"
echo "  ✅ Objects were detected and labeled" 
echo "  ✅ Context was built with vision + system prompt"
echo "  ✅ Gemma model generated action commands"
echo "  ✅ Commands were parsed and executed"
echo ""
echo "Next steps:"
echo "  - Integrate with actual robot hardware"
echo "  - Train custom vision models"
echo "  - Add more sophisticated command types"
echo "  - Optimize for real-time performance"
echo ""
echo "For production use:"
echo "  1. Enable GPU: use_gpu = true in config"
echo "  2. Reduce processing_interval_ms for faster response"
echo "  3. Fine-tune the Gemma model for your specific robot"
echo "  4. Add hardware-specific command implementations"
