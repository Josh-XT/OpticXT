# OpticXT

Vision-Driven Autonomous Robot Control & Video Chat Assistant

## Overview

OpticXT is a specialized real-time robot control system and video chat assistant that bridges computer vision, audio processing, and AI decision-making. Built as a focused evolution of AGiXT's architecture, OpticXT is designed for edge deployment on devices like the NVIDIA Jetson Nano (16GB) and Go2 robots, while also functioning as a standalone video chat assistant.

The system operates in two primary modes:
- **Robot Mode**: Full autonomous robot control with real-time vision, audio, and action capabilities
- **Video Chat Mode**: Standalone AI assistant using webcam and microphone for interactive conversations

OpticXT transforms visual and audio input into contextual understanding and immediate responses—whether through robot actions or conversational interaction.

## Key Features

### 🎯 Real-Time Vision & Audio Processing

- Real camera input with automatic device detection and fallback
- Continuous video stream analysis with object detection
- Real-time audio input from microphone with voice activity detection
- Text-to-speech output with configurable voice options
- Dynamic object labeling and context overlay

### 🤖 Dual Mode Operation

- **Robot Mode**: Action-first architecture for autonomous robot control
- **Video Chat Mode**: Conversational AI assistant with webcam/microphone
- Seamless mode switching based on hardware availability
- Pure action output system for robot mode (no conversational overhead)
- Interactive conversation system for video chat mode

### 🧠 Intelligent AI System

- **Real Gemma Model Integration**: Uses actual Gemma models for inference
- **Graceful Fallback**: Simulation mode when models are unavailable
- **Movement Control**: Direct motor and actuator commands (robot mode)
- **Voice Synthesis**: Real-time text-to-speech with multiple voice options
- **Audio Processing**: Real microphone input with noise filtering

### ⚡ Edge-Optimized Performance

- Minimal footprint design for Jetson Nano and Go2 deployment
- Efficient single-agent architecture with real hardware integration
- Real-time inference pipeline optimization
- Automatic hardware detection with intelligent fallbacks

## Core Concepts

### Vision-to-Action Pipeline
1. **Visual Input**: Raw video stream from robot cameras
2. **Context Enrichment**: Vision model adds labels and understanding
3. **Decision Layer**: Context-aware model processes enriched visuals
4. **Action Output**: XML-formatted commands for immediate execution

### Mandatory Context System
Inherited from AGiXT, OpticXT implements a mandatory context system that functions as a persistent system prompt, ensuring consistent behavior patterns and safety constraints across all operations.

### Command Execution Framework
Building on AGiXT's proven command system, OpticXT translates model outputs into real-world actions through a structured XML interface, enabling precise control over:
- Physical movements and navigation
- Sensor integration and feedback loops
- External system communication
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
git clone https://github.com/yourusername/OpticXT.git
cd OpticXT
```

2. **Install/Build the project (model download is optional)**

```bash
# Build the project - it will work with or without models
cargo build --release
```

3. **Optional: Download the Gemma Model for enhanced AI capabilities**

```bash
# Create models directory
mkdir -p models

# Option 1: Download manually from Hugging Face
# Visit: https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF
# Download gemma-3n-E4B-it-Q4_K_M.gguf to models/

# Option 2: Use wget (if direct download available)
wget -O models/gemma-3n-E4B-it-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q4_K_M.gguf"
```

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

[performance]
use_gpu = true        # Set to false if no CUDA
processing_interval_ms = 100  # Adjust for performance
```

### Running OpticXT

OpticXT automatically detects available hardware and runs in the appropriate mode:

- **Video Chat Mode**: When only webcam/microphone are available
- **Robot Mode**: When connected to robot hardware (Jetson/Go2)

#### Basic Usage

```bash
# Run with automatic hardware detection
cargo run --release

# Specify camera device
cargo run --release -- --camera-device 1

# Use custom model path (optional - will fallback to simulation if not found)
cargo run --release -- --model-path "path/to/your/model.gguf"

# Enable verbose logging
cargo run --release -- --verbose

# Use custom config file
cargo run --release -- --config "custom_config.toml"
```

#### Command Line Options

```text
USAGE:
    opticxt [OPTIONS]

OPTIONS:
    -c, --config <CONFIG>              Configuration file path [default: config.toml]
    -d, --camera-device <DEVICE>       Camera device index [default: 0]
    -m, --model-path <MODEL_PATH>      Model path for GGUF file (optional)
    -v, --verbose                      Verbose logging
    -h, --help                         Print help information
```

### System Modes

#### Video Chat Assistant Mode

When running on a standard computer with webcam and microphone:

- Uses any available camera (webcam, built-in camera)
- Real-time audio input from microphone
- Text-to-speech responses through speakers
- Interactive conversation with AI assistant
- No robot control commands

#### Robot Control Mode

When deployed on robot hardware (Jetson Nano, Go2):

- Full robot control capabilities
- Real-time vision processing for navigation
- Audio input/output for voice commands
- Movement and action execution
- Environmental sensor integration

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

## Troubleshooting

### Common Issues

#### No Camera Detected

```bash
# Check available cameras
ls /dev/video*
v4l2-ctl --list-devices

# Fix permissions
sudo usermod -a -G video $USER
# Log out and back in
```

#### Audio Issues

```bash
# Check audio devices
aplay -l    # List playback devices
arecord -l  # List capture devices

# Test microphone
arecord -d 5 test.wav && aplay test.wav

# Fix audio permissions
sudo usermod -a -G audio $USER
```

#### Model Loading (Optional)

```bash
# Check if model exists (optional - system works without it)
ls -lh models/
# Should show gemma-3n-E4B-it-Q4_K_M.gguf if downloaded

# System will use simulation mode if model is missing
# No action required - this is normal operation
```

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
- **Primary**: Real Gemma model inference via candle framework
- **Fallback**: Intelligent simulation responses when model unavailable
- **Support**: GGUF format models with tokenizer integration

### Dual Mode Architecture

#### Video Chat Assistant Mode
```
Camera → Vision Processing → AI Model → TTS Response
   ↑                                      ↓
Microphone ← Audio Processing ← User Interaction
```

#### Robot Control Mode
```
Sensors → Context Assembly → AI Decision → Action Commands
   ↑                                         ↓
Environment ← Robot Actions ← Motor Control ← XML Output
```

### Command System (Robot Mode)

OpticXT generates XML-structured commands for precise robot control:

```xml
<!-- Movement commands -->
<move direction="forward" distance="1.0" speed="slow">
  Moving forward to investigate detected object
</move>

<!-- Audio output -->
<speak>I can see someone approaching</speak>

<!-- Environmental analysis -->
<analyze target="obstacle" detail_level="detailed">
  Need to assess navigation path
</analyze>
```

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

✅ **Real Camera Input** - Works with any USB/CSI/built-in camera  
✅ **Real Audio I/O** - Microphone input and TTS output  
✅ **AI Model Integration** - Gemma model with simulation fallback  
✅ **Dual Mode Operation** - Video chat assistant or robot control  
✅ **Hardware Auto-Detection** - Intelligent fallbacks for missing hardware  
✅ **Edge Deployment Ready** - Optimized for Jetson/Go2 platforms  
✅ **Production Ready** - Real hardware integration, not simulation  

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**OpticXT: Real-world AI vision and voice assistant, ready for deployment**

*From desktop video chat to autonomous robotics - OpticXT bridges the gap between AI and the physical world.*
