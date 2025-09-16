# OpticXT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)](https://developer.nvidia.com/cuda-zone)
[![GitHub Stars](https://img.shields.io/github/stars/Josh-XT/OpticXT)](https://github.com/Josh-XT/OpticXT/stargazers)

*Vision-Driven Autonomous Robot Control System*

## Overview

OpticXT is a high-performance, real-time robot control system combining computer vision, audio processing, and multimodal AI. Powered by GPU-accelerated ISQ quantization on NVIDIA hardware for edge deployment.

OpticXT transforms visual and audio input into contextual understanding and immediate robotic actions through GPU-accelerated inference.

## Key Features

### 🚀 GPU-Accelerated AI Inference

- **ISQ Quantization**: In-Situ Quantization with Q4K precision for optimal speed/quality balance
- **CUDA Acceleration**: Full GPU acceleration on NVIDIA RTX 4090 and compatible hardware
- **Fast Model Loading**: 22-second model loading with optimized memory footprint
- **Multimodal Support**: Text, image, and audio processing with [`unsloth/gemma-3n-E4B-it`](https://huggingface.co/unsloth/gemma-3n-E4B-it) vision model
- **Real-time Inference**: 36-38% GPU utilization with 6.8GB VRAM usage for continuous processing

### 🌐 Remote Model Support (NEW)

- **OpenAI-Compatible APIs**: Support for GPT-4o, Claude, Groq, and custom endpoints
- **Minimal Hardware**: Run on Raspberry Pi Zero 2 W with remote inference
- **Vision Support**: Full multimodal capabilities via remote models
- **Seamless Integration**: Same interface for local and remote models
- **Provider Flexibility**: Switch between OpenAI, Anthropic, Groq, or self-hosted models

### 🎯 Real-Time Vision & Audio Processing

- Real camera input with automatic device detection and hardware fallback
- Optimized object detection with spam prevention (max 10 high-confidence objects)
- Concise scene descriptions: "Environment contains: person, 9 rectangular objects"
- Real-time audio input from microphone with voice activity detection
- Text-to-speech output with configurable voice options

### 🤖 Autonomous Robot Control

- **OpenAI Tool Calling**: Modern function call interface for precise robot actions
- **Action-First Architecture**: Direct translation of visual context to robot commands
- **Context-Aware Responses**: Real model computation influences tool calls based on multimodal input
- **Safety Constraints**: Built-in collision avoidance and human detection
- **Hardware Integration**: Real motor control and sensor feedback loops

### ⚡ Performance Optimized

- **CUDA Detection**: Automatic GPU detection with CPU fallback
- **Memory Efficient**: ISQ reduces memory footprint compared to full-precision models
- **Edge-Ready**: Optimized for NVIDIA Jetson Nano and desktop GPU deployment
- **Real-time Pipeline**: Sub-second inference with continuous processing

## Project Structure

OpticXT follows a clean, organized structure that separates core functionality from supporting files:

### Core Application
- **`src/`** - Main source code
  - `main.rs` - Application entry point and CLI
  - `models.rs` - AI model management (local/remote)
  - `config.rs` - Configuration management
  - `pipeline.rs` - Vision-action processing pipeline
  - `camera.rs` - Camera input handling
  - `audio.rs` - Audio input/output processing
  - `vision_basic.rs` - Computer vision processing
  - `remote_model.rs` - OpenAI-compatible API client

### Testing & Examples
- **`tests/`** - Comprehensive test suite
- **`examples/`** - Usage examples and configurations
- **`scripts/`** - Utility and testing scripts

### Documentation & Configuration
- **`docs/`** - Comprehensive documentation
- **`config.toml`** - Main configuration file
- **`models/`** - Model storage directory
- **`prompts/`** - System prompts and templates

This organization provides clear separation between operational code, testing, examples, and documentation, making the project easy to navigate and maintain.

## Core Concepts

### Vision-to-Action Pipeline

```

1. **Visual Input**: Real-time camera stream with automatic device detection
2. **Object Detection**: Optimized computer vision with spam prevention (max 10 objects)
3. **AI Processing**: GPU-accelerated ISQ inference with multimodal understanding
4. **Action Output**: OpenAI-style function calls for immediate robot execution

### ISQ Quantization System

OpticXT uses In-Situ Quantization (ISQ) for optimal performance:

- **Q4K Precision**: 4-bit quantization with optimal speed/quality balance
- **In-Memory Processing**: Weights quantized during model loading (reduced memory footprint)
- **GPU Acceleration**: Full CUDA support with 36-38% GPU utilization on RTX 4090
- **Fast Loading**: 22-second model initialization vs. slower UQFF alternatives

### Command Execution Framework

Building on modern OpenAI tool calling, OpticXT translates model outputs into robot actions:

- Physical movements and navigation commands
- Sensor integration and feedback loops
- Environmental awareness and safety constraints
- Audio/visual output generation

## Architecture Philosophy

OpticXT represents a paradigm shift from conversational AI to action-oriented intelligence. By eliminating the conversational layer and focusing purely on vision-driven decision-making, we achieve the low-latency response times critical for real-world robotics applications.

The system acts as a remote AGiXT agent, maintaining compatibility with the broader ecosystem while operating independently on edge hardware. This hybrid approach enables sophisticated behaviors through local processing while retaining the ability to offload complex tasks when needed.

## Use Cases

- Autonomous navigation in dynamic environments
- Real-time object interaction and manipulation
- Surveillance and monitoring applications
- Assistive robotics with visual understanding
- Industrial automation with adaptive behavior

## Getting Started

### Prerequisites

#### System Requirements

- Any Linux system with camera and microphone (for video chat mode)
- NVIDIA Jetson Nano (16GB) or Go2 robot (for full robot mode)
- USB camera, webcam, or CSI camera module
- Microphone and speakers/headphones for audio
- Rust 1.70+ installed

#### Dependencies

```bash
# Ubuntu/Debian - Basic dependencies
sudo apt update
sudo apt install -y build-essential cmake pkg-config

# Audio system dependencies (required)
sudo apt install -y libasound2-dev portaudio19-dev

# TTS support (required for voice output)
sudo apt install -y espeak espeak-data libespeak-dev

# Optional: Additional audio codecs
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Josh-XT/OpticXT.git
cd OpticXT
```

2. **Setup CUDA Environment (Required for GPU acceleration)**

For PowerShell (Windows/WSL):

```powershell
# Set CUDA environment variables
$env:CUDA_ROOT = "/usr/local/cuda-12.5"
$env:PATH = "$env:CUDA_ROOT/bin:$env:PATH"
$env:LD_LIBRARY_PATH = "$env:CUDA_ROOT/lib64:$env:LD_LIBRARY_PATH"

# Or run the setup script
./setup_cuda.ps1
```

For Bash/Zsh (Linux/macOS):

```bash
# Set CUDA environment variables
export CUDA_ROOT="/usr/local/cuda-12.5"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"

# Or run the setup script
source ./setup_cuda.sh
```

3. **Build with CUDA support for GPU acceleration**

```bash
# For NVIDIA GPU acceleration (recommended) - with CUDA environment
cargo build --release

# For CPU-only mode (fallback) - no CUDA dependencies
cargo build --release --no-default-features
```

4. **The system uses ISQ quantization with automatic model download**

The system automatically downloads and quantizes the `unsloth/gemma-3n-E4B-it` model:

- **Model**: unsloth/gemma-3n-E4B-it (vision-capable, no authentication required)
- **Quantization**: ISQ Q4K (in-situ quantization during loading)
- **Loading Time**: ~22 seconds with GPU acceleration
- **Memory Usage**: ~6.8GB VRAM on RTX 4090

**Note**: Models are downloaded automatically from HuggingFace on first run. No manual model installation required.

### Configuration

Edit `config.toml` to match your setup:

```toml
[vision]
width = 640           # Camera resolution
height = 480
confidence_threshold = 0.5

[audio]
input_device = "default"    # Microphone device
output_device = "default"   # Speaker device
voice = "en"               # TTS voice language
enable_vad = true          # Voice activity detection

[model]
model_path = "models/gemma-3n-E4B-it-Q4_K_M.gguf"
temperature = 0.7     # Lower = more deterministic

# Remote model configuration (optional)
# Uncomment to use remote API instead of local model
# [model.remote]
# base_url = "https://api.openai.com/v1"
# api_key = "your-api-key-here"
# model_name = "gpt-4o"
# supports_vision = true

[performance]
use_gpu = true        # Set to false if no CUDA
processing_interval_ms = 100  # Adjust for performance
```

#### Remote Model Support

OpticXT supports remote model inference via OpenAI-compatible APIs, enabling deployment on low-power hardware like Raspberry Pi Zero 2 W. When a remote model is configured, OpticXT will use it instead of local inference, dramatically reducing hardware requirements while maintaining full functionality.

**Supported Remote Providers:**
- **OpenAI**: GPT-4o with vision support for high-quality multimodal inference
- **Groq**: Ultra-fast inference with Llama models (text-only)
- **Anthropic**: Claude models via OpenAI-compatible endpoints
- **Local APIs**: LM Studio, Ollama, or any OpenAI-compatible server

**Example Remote Configurations:**

```toml
# OpenAI GPT-4o with vision
[model.remote]
base_url = "https://api.openai.com/v1"
api_key = "your-openai-key"
model_name = "gpt-4o"
supports_vision = true

# Groq (very fast, text-only)
[model.remote]
base_url = "https://api.groq.com/openai/v1"
api_key = "your-groq-key"
model_name = "llama-3.1-70b-versatile"
supports_vision = false

# Local LM Studio server
[model.remote]
base_url = "http://localhost:1234/v1"
api_key = "not-needed"
model_name = "local-model"
supports_vision = false
```

**Benefits of Remote Models:**
- **Minimal Hardware**: Run on Pi Zero 2 W or any low-power device
- **No GPU Required**: Offload inference to powerful remote servers
- **Latest Models**: Access to cutting-edge models without local storage
- **Scalability**: Handle multiple robot instances without per-device model loading

### Examples and Scripts

OpticXT includes comprehensive examples and utility scripts:

**Configuration Examples** (`examples/`):
- `examples/remote_model_examples.toml` - Remote model configurations for various providers
- `examples/example_api_client.py` - Python client demonstrating API usage

**Utility Scripts** (`scripts/`):
- `scripts/test_api.sh` - Test API endpoints
- `scripts/debug_cuda.sh` - CUDA environment debugging
- `scripts/demo.sh` - System demonstration
- `scripts/test_model_performance.sh` - Performance benchmarking

### Running OpticXT

OpticXT runs as an autonomous robot control system with GPU-accelerated AI inference:

#### Basic Usage

```bash
# Run with CUDA acceleration (recommended)
cargo run --release --features cuda -- --verbose

# Monitor GPU utilization (separate terminal)
watch -n 1 nvidia-smi

# Check real-time inference performance
cargo run --release --features cuda -- --verbose 2>&1 | grep "GPU Utilization\|GPU Memory"
```

#### Command Line Options

```bash
USAGE:
    opticxt [OPTIONS]

OPTIONS:
    -c, --config <CONFIG>              Configuration file path [default: config.toml]
    -d, --camera-device <DEVICE>       Camera device index [default: 0]
    -m, --model-path <PATH>            Override model path from config
        --video-chat                   Run in video chat/assistant mode
        --chat-mode <MODE>             Chat mode: assistant, monitoring [default: assistant]
    -v, --verbose                      Enable verbose logging with GPU monitoring
        --benchmark                    Run model performance benchmark
        --benchmark-iterations <N>     Number of benchmark iterations [default: 50]
        --api-server                   Start API server mode
        --api-port <PORT>              API server port [default: 8080]
    -h, --help                         Print help information
```

**Note**: All test commands have been moved to standard Cargo test commands. Use `cargo test` instead of command-line test flags.

#### Expected Performance (RTX 4090)

- **Model Loading**: ~22 seconds (ISQ quantization)
- **GPU Memory Usage**: ~6.8GB VRAM / 24.5GB total (28% utilization)
- **Inference Speed**: 36-38% GPU utilization during processing
- **Object Detection**: Max 10 high-confidence objects per frame
- **Response Time**: 3-6 seconds per inference cycle

## Usage

### Robot Control Mode

After installation, run the system:

```bash
# Start in video chat mode (default)
cargo run --release --features cuda

# Run with custom config
cargo run --release --features cuda -- --config custom.toml --camera-device 1
```

### API Server Mode

OpticXT can also run as a REST API server for integration with web applications, mobile apps, or other services:

```bash
# Start API server on default port 8080
cargo run --release --features cuda -- --api-server

# Start on custom port
cargo run --release --features cuda -- --api-server --api-port 3000

# CPU-only mode (no CUDA required)
cargo run --release --no-default-features -- --api-server
```

**API Endpoint**: `POST /v1/inference`

- Accepts multipart form data with optional text, images, or video files
- Returns JSON responses with model-generated text
- Supports real-time task status monitoring
- See `API_DOCUMENTATION.md` for detailed usage examples

Test the API:

```bash
# Simple text inference
curl -X POST http://localhost:8080/v1/inference -F "text=What do you see?"

# Image analysis
curl -X POST http://localhost:8080/v1/inference -F "text=Describe this image" -F "image=@photo.jpg"
```

For robot control mode, edit `config.toml` to enable hardware integration.

**Quick Test**: Verify your setup with tests:

```bash
# Run basic functionality tests
cargo test --test test_simple

# Test with real camera (if available)
cargo test --test test_camera_vision
```

## Hardware Requirements & Compatibility

### Minimum Requirements (Video Chat Mode)

- Any Linux system with USB ports
- Webcam or built-in camera
- Microphone and speakers/headphones
- 4GB RAM minimum (8GB recommended)
- Rust 1.70+ toolchain

### Full Robot Mode Requirements

- NVIDIA Jetson Nano (16GB) or Go2 robot platform
- CSI or USB camera
- Microphone and speaker system
- Motor controllers and actuators
- Optional: LiDAR sensors (falls back to simulation)

### Audio System

OpticXT includes a complete real audio pipeline:

- **Input**: Real-time microphone capture with voice activity detection
- **Output**: Text-to-speech synthesis with multiple voice options
- **Processing**: Audio filtering and noise reduction
- **Fallback**: Graceful degradation when audio hardware is unavailable

### Camera System

Flexible camera support with automatic detection:

- **USB Cameras**: Automatic detection of any UVC-compatible camera
- **Built-in Cameras**: Laptop/desktop integrated cameras
- **CSI Cameras**: Jetson Nano camera modules
- **Fallback**: Simulation mode when no camera is detected

## Testing & Development

OpticXT includes comprehensive testing capabilities organized in the `tests/` directory:

### Running Tests

```bash
# Run all tests using standard Cargo commands
cargo test

# Run unit tests only
cargo test --lib

# Run integration tests  
cargo test --test integration_tests

# Run specific test files
cargo test --test test_simple
cargo test --test test_multimodal
cargo test --test test_camera_vision
```

### Test Organization

Tests are organized in the following structure:
- **`tests/integration_tests.rs`** - Main integration tests
- **`tests/test_simple.rs`** - Basic text inference tests
- **`tests/test_multimodal.rs`** - Multimodal (text + image + audio) tests
- **`tests/test_image.rs`** - Image processing tests
- **`tests/test_camera_vision.rs`** - Real camera vision tests
- **`tests/test_remote_model.rs`** - Remote model configuration tests
- **`tests/test_vision_flow.rs`** - Vision pipeline flow tests
- **`tests/test_vision_pipeline.rs`** - Vision processing pipeline tests

### Development Testing Workflow

1. **Start with basic tests**: `cargo test --lib` for unit tests
2. **Test specific components**: `cargo test --test test_simple` for text inference
3. **Test camera integration**: `cargo test --test test_camera_vision` (requires camera)
4. **Test multimodal features**: `cargo test --test test_multimodal`
5. **Test remote models**: `cargo test --test test_remote_model`
6. **Full integration**: `cargo test --test integration_tests`

### Test Scripts and Examples

The `scripts/` directory contains utility scripts for testing and development:
- **`scripts/test_api.sh`** - API endpoint testing
- **`scripts/test_model_performance.sh`** - Model performance benchmarks
- **`scripts/debug_cuda.sh`** - CUDA debugging utilities
- **`scripts/demo.sh`** - System demonstration script

### Example Usage and Configuration

The `examples/` directory contains practical examples and configurations:
- **`examples/example_api_client.py`** - Python API client example
- **`examples/remote_model_examples.toml`** - Remote model configuration examples

For detailed remote model configuration examples, see `examples/remote_model_examples.toml`.

### Expected Test Results

- **Model Loading**: 15-25 seconds depending on hardware
- **Text Generation**: 50-200ms per response
- **Image Processing**: 200-800ms per image
- **Audio Processing**: 100-500ms per audio segment
- **Tool Call Generation**: Valid JSON format with proper function structure

## Troubleshooting

### CUDA Issues
- **CUDA Not Detected**: 
  - Verify drivers: `nvidia-smi`
  - Rebuild: `cargo clean && cargo build --release --features cuda`
- Ensure NVIDIA drivers are installed and up to date
- Verify CUDA toolkit installation (see `CUDA_BUILD_GUIDE.md`)
- Check that your GPU supports the CUDA version
- Try running without CUDA features first: `cargo run --release`

### Model Loading Issues
- Verify internet connection for model downloads
- Check available disk space (models can be several GB)
- Ensure sufficient system memory (8GB+ recommended)
- Clear model cache if corruption suspected: `rm -rf ~/.cache/huggingface`

### Camera/Audio Issues
- **No Camera Detected**:
  ```bash
  # Check available cameras
  ls /dev/video*
  v4l2-ctl --list-devices
  
  # Fix permissions
  sudo usermod -a -G video $USER
  # Log out and back in
  ```
- **Audio Issues**:
  ```bash
  # Check audio devices
  aplay -l    # List playback devices
  arecord -l  # List capture devices
  
  # Test microphone
  arecord -d 5 test.wav && aplay test.wav
  
  # Fix audio permissions
  sudo usermod -a -G audio $USER
  ```
- For multiple cameras, try different device indices (0, 1, 2...)

#### Model Loading (Real AI Inference - Current Status)

OpticXT successfully loads and runs AI models with real neural network inference:

```bash
# 1. Current Status Check
ls -lh models/
# Should show both gemma-3n-E4B-it-Q4_K_M.gguf and tokenizer.json

# 2. What's Working:
# ✅ Model file loading (GGUF format)
# ✅ Tokenizer loading and text processing
# ✅ Model architecture initialization
# ✅ Neural network forward pass with real inference
# ✅ Real token generation from model logits
# ✅ Context-aware function call output generation from actual model output
# ✅ Complete removal of all hardcoded/simulation fallbacks

# 3. Current Behavior:
# - Models load successfully with real tokenizer and UQFF quantization
# - Real model inference with mistral.rs and multimodal support (text/image/audio)
# - OpenAI-style function calls generated from genuine model output
# - System fails gracefully with clear error messages when models unavailable
# - All functionality (camera, audio, movement) works with real hardware input
# - NO hardcoded responses or simulation fallbacks whatsoever

# 4. Expected Log Messages:
# ✅ "✅ Successfully loaded HuggingFace Gemma 3n model with multimodal support" 
# ✅ "Real multimodal model generated text in XXXms"
# ✅ "Running model inference with X modalities"
# ❌ "Model inference timed out after 180 seconds" (when model performance issues)

# 5. Error Handling:
# The system now properly fails with informative errors when:
# - Model files are missing or corrupted
# - Tokenizer cannot be loaded
# - Real inference fails
# This ensures complete authenticity - no fake responses under any circumstances
```

**Current Status**: The system uses exclusively real neural network inference with genuine GGUF model loading and authentic tokenizer processing. All simulation logic, hardcoded responses, and fallback mechanisms have been completely removed. The system will only operate with actual model inference or fail gracefully with clear error messages.

#### Build Issues

```bash
# Install missing dependencies
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libasound2-dev portaudio19-dev
sudo apt install -y espeak espeak-data libespeak-dev

# Clean and rebuild
cargo clean
cargo build --release
```

#### Performance Issues

```bash
# Monitor system resources
htop
# Watch for CPU/memory usage

# Adjust processing interval in config.toml
# Increase processing_interval_ms for lower resource usage
```

### Debug Mode

Enable comprehensive logging:

```bash
RUST_LOG=debug cargo run -- --verbose
```

This shows:

- Camera detection and initialization
- Audio device enumeration
- Model loading status (real or simulation)
- Frame processing times
- Audio input/output status
- Error details and fallback triggers

## Architecture & Technical Details

### Real Hardware Integration

OpticXT is built around real hardware components with intelligent fallbacks:

#### Vision System
- **Primary**: Real camera capture via nokhwa library
- **Fallback**: Simulated visual input when no camera detected
- **Support**: USB, CSI, and built-in cameras with automatic detection

#### Audio System
- **Input**: Real microphone capture via cpal library
- **Output**: Text-to-speech via tts/espeak integration
- **Processing**: Voice activity detection and audio filtering
- **Fallback**: Silent operation when audio hardware unavailable

#### AI Model System

- **Model Loading**: Successfully loads UQFF models with mistral.rs VisionModelBuilder
- **Real Inference**: Full multimodal neural network inference with text, image, and audio support
- **Context-Aware Responses**: Real model computation influences function call output based on multimodal input
- **Tool Call Generation**: OpenAI-style function calls generated from actual model outputs
- **Intelligent Fallback**: Graceful degradation when models unavailable

### Dual Mode Architecture

#### Video Chat Assistant Mode
```
Camera → Vision Processing → AI Model → TTS Response
   ↑                                      ↓
Microphone ← Audio Processing ← User Interaction
```

#### Robot Control Mode
```
Sensors → Context Assembly → AI Decision → Function Calls
   ↑                                         ↓
Environment ← Robot Actions ← Motor Control ← Tool Call Output
```

### Command System (Robot Mode)

OpticXT generates OpenAI-style function calls for precise robot control:

```json
[{
  "id": "call_1",
  "type": "function",
  "function": {
    "name": "move",
    "arguments": "{\"direction\": \"forward\", \"distance\": 1.0, \"speed\": \"slow\", \"reasoning\": \"Moving forward to investigate detected object\"}"
  }
}]

[{
  "id": "call_1",
  "type": "function",
  "function": {
    "name": "speak",
    "arguments": "{\"text\": \"I can see someone approaching\", \"voice\": \"default\", \"reasoning\": \"Alerting about detected human presence\"}"
  }
}]

[{
  "id": "call_1",
  "type": "function",
  "function": {
    "name": "analyze",
    "arguments": "{\"target\": \"obstacle\", \"detail_level\": \"detailed\", \"reasoning\": \"Need to assess navigation path\"}"
  }
}]
```

## Roadmap

- **Integrate Enhanced Model Support**: Add Llama variants and other state-of-the-art models
- **Add ROS2 Integration**: Advanced robotics framework support for complex robot control
- **Enhance Real-time Audio**: Improved speech recognition and audio processing capabilities
- **Create Web Interface**: Browser-based control panel for remote operation
- **Add Docker Support**: Containerized deployment for easier setup and scaling
- **Enable Multi-Robot Coordination**: Support for controlling multiple robots simultaneously

## Deployment Guide

### Desktop/Laptop Development

1. **Install dependencies** (audio system required)
2. **Build project**: `cargo build --release`
3. **Run**: `cargo run --release`
4. **Mode**: Automatically runs as video chat assistant

### Jetson Nano Deployment

1. **Flash Jetson with Ubuntu 20.04**
2. **Install Rust toolchain**
3. **Install system dependencies** (including CUDA if available)
4. **Build with optimizations**: `cargo build --release`
5. **Configure hardware** in `config.toml`
6. **Deploy**: Copy binary and config to target system

### Go2 Robot Integration

1. **Cross-compile** for ARM64 architecture
2. **Install on Go2** via SDK deployment tools
3. **Configure sensors** for robot hardware
4. **Enable robot mode** in configuration
5. **Test control commands** before full deployment

## Key Features Summary

✅ **GPU-Accelerated AI**: ISQ quantization with CUDA acceleration on NVIDIA RTX 4090  
✅ **Real Camera Input**: Works with any USB/CSI/built-in camera  
✅ **Real Audio I/O**: Microphone input and TTS output  
✅ **Multimodal AI**: unsloth/gemma-3n-E4B-it model with text, image, and audio support  
✅ **OpenAI Tool Calling**: Robot control commands via modern function call interface  
✅ **Optimized Vision**: Spam prevention with max 10 high-confidence objects per frame  
✅ **Hardware Auto-Detection**: Real hardware integration with CUDA detection and CPU fallback  
✅ **Edge Deployment Ready**: Optimized for NVIDIA Jetson and desktop GPU platforms  
✅ **Production Ready**: 22-second model loading, 36-38% GPU utilization, 6.8GB VRAM usage  

**Performance Status**: Full GPU acceleration achieved with ISQ quantization. System loads `unsloth/gemma-3n-E4B-it` model in 22 seconds, utilizes 28% of RTX 4090 VRAM, and processes inference at 36-38% GPU utilization with optimized vision processing.

## Contributing

We welcome contributions! Please see our [contributing guidelines](docs/CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

For detailed documentation, see:
- **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference
- **[CUDA Build Guide](docs/CUDA_BUILD_GUIDE.md)** - GPU setup instructions
- **[Remote Model Implementation](docs/REMOTE_MODEL_IMPLEMENTATION.md)** - Remote API integration guide

---

**⭐ Found OpticXT useful?** Star this repository to show your support!  
**❓ Have questions?** [Open an issue](https://github.com/Josh-XT/OpticXT/issues) - we're here to help!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support the Project

If OpticXT has helped you, consider supporting development:

- ⭐ **Star the repo** to show your support!
- 💬 **Join discussions** or report issues on GitHub
- ☕ **Support development**: CA: `Ga9P2TZcxsHjYmXdEyu9Z7wL1QAowjBAZwRQ41gBbonk`

---

## OpticXT: GPU-Accelerated Vision Robot Control

**Real-time robot control with ISQ quantization and CUDA acceleration - from autonomous navigation to precise manipulation.**
