#!/bin/bash
# HVAC Development Environment Setup Script
# This script initializes the HVAC-AI development environment with all required dependencies

set -e

echo "=========================================="
echo "HVAC-AI Development Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
echo "Checking prerequisites..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python 3 found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check Node.js version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js found: $NODE_VERSION"
else
    print_error "Node.js not found. Please install Node.js 18 or higher."
    exit 1
fi

# Check for GPU support (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    GPU_AVAILABLE=true
else
    print_info "No NVIDIA GPU detected. CPU inference will be slower."
    GPU_AVAILABLE=false
fi

echo ""
echo "Installing Python dependencies..."
cd services/hvac-ai

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_info "Installing Python dependencies (this may take several minutes)..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    exit 1
fi

cd ..

echo ""
echo "Installing Node.js dependencies..."
npm install

if [ $? -eq 0 ]; then
    print_success "Node.js dependencies installed"
else
    print_error "Failed to install Node.js dependencies"
    exit 1
fi

echo ""
echo "Setting up environment configuration..."

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "Created .env file from .env.example"
    print_info "Please edit .env and configure your settings"
else
    print_info ".env file already exists"
fi

echo ""
echo "Validating HVAC-specific setup..."

# Check for model directory
if [ ! -d "services/hvac-ai/models" ]; then
    mkdir -p services/hvac-ai/models
    print_info "Created services/hvac-ai/models directory. Please download or copy your inference model (YOLO) into services/hvac-ai/models/"
fi

# Check for HVAC datasets directory
if [ ! -d "hvac-datasets" ]; then
    mkdir -p hvac-datasets
    print_success "Created hvac-datasets directory"
fi

# Check for HVAC scripts directory
if [ ! -d "hvac-scripts" ]; then
    mkdir -p hvac-scripts
    print_success "Created hvac-scripts directory"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_success "HVAC-AI development environment is ready"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Download or copy your inference model (YOLO/Ultralytics) to the models/ directory"
echo "3. Run 'npm run dev:all' to start development servers"
echo ""
echo "For more information, see docs/GETTING_STARTED.md"
