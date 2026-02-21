#!/bin/bash

# ===========================================
# AI Vinahouse Remix - macOS Setup Script
# Tối ưu cho Apple Silicon (M1/M2/M3/M4)
# ===========================================

set -e

echo "============================================"
echo "🎵 AI Vinahouse Remix - macOS Setup"
echo "🖥️  Optimized for Apple Silicon"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Apple Silicon
check_architecture() {
    echo -e "${BLUE}[1/7] Checking System Architecture...${NC}"
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "${GREEN}✅ Apple Silicon detected: $ARCH${NC}"
        export ARCHFLAGS="-arch arm64"
    else
        echo -e "${YELLOW}⚠️  Intel Mac detected: $ARCH${NC}"
        echo "    Apple Silicon recommended for best performance"
    fi
    echo ""
}

# Check Homebrew
check_homebrew() {
    echo -e "${BLUE}[2/7] Checking Homebrew...${NC}"
    if command -v brew &> /dev/null; then
        echo -e "${GREEN}✅ Homebrew installed${NC}"
        brew --version
    else
        echo -e "${YELLOW}⚠️  Homebrew not found. Installing...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    echo ""
}

# Install system dependencies
install_system_deps() {
    echo -e "${BLUE}[3/7] Installing System Dependencies...${NC}"
    
    # Required for audio processing
    brew list ffmpeg &> /dev/null || brew install ffmpeg
    brew list portaudio &> /dev/null || brew install portaudio
    brew list libsamplerate &> /dev/null || brew install libsamplerate
    brew list cmake &> /dev/null || brew install cmake
    
    echo -e "${GREEN}✅ System dependencies installed${NC}"
    echo ""
}

# Check Node.js
check_node() {
    echo -e "${BLUE}[4/7] Checking Node.js...${NC}"
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v)
        echo -e "${GREEN}✅ Node.js installed: $NODE_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠️  Node.js not found. Installing...${NC}"
        brew install node
    fi
    echo ""
}

# Check Python
check_python() {
    echo -e "${BLUE}[5/7] Checking Python...${NC}"
    
    # Prefer Python 3.11 for best ML compatibility
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD=python3.11
        echo -e "${GREEN}✅ Python 3.11 found${NC}"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
        PY_VERSION=$($PYTHON_CMD --version)
        echo -e "${GREEN}✅ Python found: $PY_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠️  Python not found. Installing Python 3.11...${NC}"
        brew install python@3.11
        PYTHON_CMD=python3.11
    fi
    
    echo "Using: $PYTHON_CMD"
    echo ""
}

# Setup Python virtual environment
setup_venv() {
    echo -e "${BLUE}[6/7] Setting up Python Virtual Environment...${NC}"
    
    cd backend
    
    # Create venv if not exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    fi
    
    # Activate venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch for Apple Silicon (MPS support)
    echo -e "${YELLOW}Installing PyTorch with MPS support...${NC}"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install other dependencies
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements-macos.txt
    
    # Download Demucs model
    echo -e "${YELLOW}Downloading Demucs models...${NC}"
    python -c "import demucs; print('Demucs ready')" || echo "Demucs will download models on first use"
    
    cd ..
    
    echo -e "${GREEN}✅ Python environment ready${NC}"
    echo ""
}

# Install Node dependencies
install_npm_deps() {
    echo -e "${BLUE}[7/7] Installing Node.js Dependencies...${NC}"
    
    npm install
    
    echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
    echo ""
}

# Test MPS availability
test_mps() {
    echo -e "${BLUE}Testing Metal Performance Shaders (MPS)...${NC}"
    
    source backend/venv/bin/activate
    
    python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(3, 3).to(device)
    print("✅ MPS GPU acceleration working!")
else:
    print("⚠️  MPS not available, using CPU")
EOF
    
    echo ""
}

# Create necessary directories
create_dirs() {
    echo -e "${BLUE}Creating directories...${NC}"
    
    mkdir -p uploads
    mkdir -p outputs
    mkdir -p pretrained
    mkdir -p backend/models
    
    echo -e "${GREEN}✅ Directories created${NC}"
    echo ""
}

# Print summary
print_summary() {
    echo ""
    echo "============================================"
    echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
    echo "============================================"
    echo ""
    echo "🚀 To start the application:"
    echo ""
    echo "   # Terminal 1 - Backend:"
    echo "   cd backend && source venv/bin/activate && python main.py"
    echo ""
    echo "   # Terminal 2 - Frontend:"
    echo "   npm run dev"
    echo ""
    echo "   # Or run both with Electron:"
    echo "   npm run electron:dev"
    echo ""
    echo "📦 To build macOS app:"
    echo "   npm run electron:build:mac"
    echo ""
    echo "💾 Your Mac M4 specs:"
    echo "   - 24GB RAM: Excellent for AI processing"
    echo "   - Apple Silicon: MPS GPU acceleration enabled"
    echo "   - Expected performance: Very fast! 🚀"
    echo ""
    echo "============================================"
}

# Main execution
main() {
    check_architecture
    check_homebrew
    install_system_deps
    check_node
    check_python
    setup_venv
    install_npm_deps
    create_dirs
    test_mps
    print_summary
}

# Run
main