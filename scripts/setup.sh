#!/bin/bash
# Setup script for HVAC AI Platform development environment

set -e  # Exit on error

echo "🚀 HVAC AI Platform - Setup Script"
echo "=================================="

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION detected"

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies..."
npm install

# Setup Python environment
echo ""
echo "🐍 Setting up Python environment..."
cd services/hvac-ai

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

cd ..

# Check for .env file
if [ ! -f ".env.local" ]; then
    echo ""
    echo "⚠️  No .env.local file found"
    echo "📝 Creating .env.local from .env.example..."
    cp .env.example .env.local
    echo "✅ Created .env.local - Please update with your configuration"
fi

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p services/hvac-ai/models
mkdir -p services/hvac-ai/logs
mkdir -p datasets

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env.local with your configuration"
echo "2. Place your inference model at services/hvac-ai/models/<your_model_file>.pt (YOLO/Ultralytics recommended)"
echo "3. Run 'npm run dev' to start the frontend"
echo "4. Run 'python services/hvac_unified_service.py' to start the backend (or see services/hvac-ai for service modules)"
echo ""
echo "📚 See docs/GETTING_STARTED.md for more information"
