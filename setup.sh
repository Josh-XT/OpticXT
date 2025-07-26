#!/bin/bash

# OpticXT Build and Setup Script

set -e

echo "🔧 Setting up OpticXT Development Environment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported system
print_status "Checking system compatibility..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_success "Linux system detected"
else
    print_warning "This script is optimized for Linux. Some features may not work on other systems."
fi

# Check Rust installation
print_status "Checking Rust installation..."
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    print_success "Rust found: $RUST_VERSION"
else
    print_error "Rust not found. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check for required system packages
print_status "Checking system dependencies..."

MISSING_PACKAGES=()

# Check for OpenCV
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    MISSING_PACKAGES+=("libopencv-dev")
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    MISSING_PACKAGES+=("ffmpeg")
fi

# Check for build tools
if ! command -v cmake &> /dev/null; then
    MISSING_PACKAGES+=("cmake")
fi

if ! command -v pkg-config &> /dev/null; then
    MISSING_PACKAGES+=("pkg-config")
fi

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    print_error "Missing required packages: ${MISSING_PACKAGES[*]}"
    print_status "Install them with: sudo apt install ${MISSING_PACKAGES[*]}"
    
    read -p "Would you like to install them now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt update
        sudo apt install -y "${MISSING_PACKAGES[@]}"
        print_success "Packages installed successfully"
    else
        print_error "Please install the required packages and run this script again"
        exit 1
    fi
else
    print_success "All required system packages are installed"
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
print_success "Directories created"

# Check for GPU support
print_status "Checking for GPU support..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    print_warning "No NVIDIA GPU detected. Will use CPU for inference."
fi

# Build the project
print_status "Building OpticXT..."
if cargo build --release; then
    print_success "Build completed successfully"
else
    print_error "Build failed. Please check the error messages above."
    exit 1
fi

# Check for model file
MODEL_PATH="models/gemma-3n-E4B-it-Q4_K_M.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    print_warning "Model file not found at $MODEL_PATH"
    print_status "You need to download the Gemma model manually:"
    echo "  1. Visit: https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF"
    echo "  2. Download: gemma-3n-E4B-it-Q4_K_M.gguf"
    echo "  3. Place it in: $MODEL_PATH"
    echo ""
    print_status "Or try using the HuggingFace CLI:"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download unsloth/gemma-3n-E4B-it-GGUF gemma-3n-E4B-it-Q4_K_M.gguf --local-dir models/"
else
    print_success "Model file found"
fi

# Test camera access
print_status "Testing camera access..."
if [ -e "/dev/video0" ]; then
    print_success "Camera device /dev/video0 found"
else
    print_warning "No camera device found at /dev/video0"
    print_status "Available video devices:"
    ls /dev/video* 2>/dev/null || echo "  None found"
fi

# Create a simple test script
print_status "Creating test script..."
cat > test_opticxt.sh << 'EOF'
#!/bin/bash
echo "🚀 Testing OpticXT..."
echo "This will run OpticXT in test mode for 30 seconds"
echo "Press Ctrl+C to stop early"
echo ""

# Run with test parameters
timeout 30s ./target/release/opticxt --verbose || echo "Test completed"
EOF
chmod +x test_opticxt.sh
print_success "Test script created: ./test_opticxt.sh"

# Summary
echo ""
print_success "🎉 OpticXT setup completed!"
echo ""
echo "Next steps:"
echo "  1. Download the Gemma model (see instructions above)"
echo "  2. Connect a camera to your system"
echo "  3. Run: ./test_opticxt.sh"
echo "  4. Or run directly: ./target/release/opticxt"
echo ""
echo "Configuration file: config.toml"
echo "Logs will be saved to: logs/"
echo ""
print_status "For help, run: ./target/release/opticxt --help"
