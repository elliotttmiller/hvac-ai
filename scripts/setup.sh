#!/bin/bash
# 🛠️ HVAC Cortex Platform - End-to-End Setup Script
# Optimizes environment for Ray Serve (Backend) and Next.js (Frontend)

set -e  # Exit immediately if a command exits with a non-zero status

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}🚀 HVAC Cortex - Platform Initialization & Setup${NC}"
echo -e "${BLUE}==================================================${NC}"

# --- 1. System Prerequisite Checks ---
echo -e "\n${YELLOW}[1/6] Checking System Prerequisites...${NC}"

# Node.js Check
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed.${NC} Please install Node.js 18+."
    exit 1
fi
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}❌ Node.js 18+ required.${NC} Current: $(node -v)"
    exit 1
fi
echo -e "${GREEN}✅ Node.js $(node -v) detected${NC}"

# Python Check
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}❌ Python is not installed.${NC} Please install Python 3.10+."
    exit 1
fi
PY_VER=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python $PY_VER detected${NC}"

# --- 2. Frontend Setup ---
echo -e "\n${YELLOW}[2/6] Setting up Frontend (Next.js)...${NC}"
if [ -f "package.json" ]; then
    echo "📦 Installing Node dependencies..."
    npm install --silent
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
else
    echo -e "${RED}❌ package.json not found in root.${NC}"
    exit 1
fi

# --- 3. Backend Environment Setup ---
echo -e "\n${YELLOW}[3/6] Setting up Backend Virtual Environment...${NC}"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Creating virtual environment at $VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Activate Venv (Cross-Platform compatibility)
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Windows (Git Bash)
    source "$VENV_DIR/Scripts/activate"
else
    # Linux / Mac
    source "$VENV_DIR/bin/activate"
fi

echo "📦 Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --quiet

# --- 4. Backend Dependencies & GPU Optimization ---
echo -e "\n${YELLOW}[4/6] Installing AI & Infrastructure Dependencies...${NC}"

REQUIREMENTS_FILE="services/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📥 Installing base requirements from $REQUIREMENTS_FILE..."
    pip install -r $REQUIREMENTS_FILE --quiet
else
    echo -e "${RED}❌ Requirements file not found at $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Explicit Ray Serve Check
echo "📡 Ensuring Ray Serve is installed..."
pip install "ray[serve]" --quiet

# GPU Logic: Detect NVIDIA and force CUDA PyTorch
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}🎮 NVIDIA GPU Detected via nvidia-smi${NC}"
    echo "🔄 Enforcing PyTorch with CUDA 12.1 support..."
    
    # Check if torch is already CUDA-enabled
    CUDA_CHECK=$(python -c "import torch; print(torch.cuda.is_available())")
    
    if [ "$CUDA_CHECK" == "False" ]; then
        echo "⚠️  Current PyTorch is CPU-only. Reinstalling..."
        pip uninstall -y torch torchvision torchaudio
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
        echo -e "${GREEN}✅ PyTorch upgraded to CUDA version${NC}"
    else
        echo -e "${GREEN}✅ PyTorch is already CUDA-enabled${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. Running in CPU mode (Slower).${NC}"
fi

# --- 5. Configuration Setup ---
echo -e "\n${YELLOW}[5/6] Configuring Environment Variables...${NC}"

if [ ! -f ".env.local" ]; then
    if [ -f ".env.example" ]; then
        echo "📝 Creating .env.local from .env.example..."
        cp .env.example .env.local
        echo -e "${GREEN}✅ Created .env.local${NC}"
    else
        echo -e "${YELLOW}⚠️  No .env.example found. Skipping file creation.${NC}"
    fi
else
    echo "✅ .env.local already exists"
fi

# --- 6. Directory Structure ---
echo -e "\n${YELLOW}[6/6] Finalizing Directory Structure...${NC}"
mkdir -p services/hvac-ai/logs
mkdir -p ai_model
mkdir -p uploads

# Check for Model
MODEL_PATH="ai_model/best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  Model file missing at $MODEL_PATH${NC}"
    echo "   Please place your YOLOv11 .pt file there before starting."
fi

# --- Completion ---
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""
echo -e "To start the full platform (Backend + Frontend), run:"
echo -e "   ${GREEN}python scripts/start_unified.py${NC}"
echo ""