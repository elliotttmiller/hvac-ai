#!/bin/bash
# HVAC AI Platform Python Service Startup Script

set -e

echo "🚀 Starting HVAC AI Analysis Service..."
echo ""

# Color codes for better UX
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.9 or higher"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo "📚 Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check for .env file
if [ ! -f "../.env" ]; then
    echo -e "${YELLOW}⚠ Warning: .env file not found${NC}"
    echo "  Copy .env.example to .env and configure MODEL_PATH"
    echo "  The service will start but analysis features may not work"
    echo ""
fi

# Check MODEL_PATH if .env exists
if [ -f "../.env" ]; then
    if grep -q "^MODEL_PATH=" "../.env"; then
        MODEL_PATH=$(grep "^MODEL_PATH=" "../.env" | cut -d'=' -f2)
        if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
            MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
            echo -e "${GREEN}✓ Model found: $MODEL_PATH ($MODEL_SIZE)${NC}"
        else
            echo -e "${YELLOW}⚠ Warning: MODEL_PATH configured but model file not found${NC}"
            echo "  Analysis features will not work until model is available"
        fi
    else
        echo -e "${YELLOW}⚠ Warning: MODEL_PATH not configured in .env${NC}"
    fi
    echo ""
fi

# Start the service
echo "✅ Starting FastAPI service on http://localhost:8000"
echo "📖 API Documentation available at http://localhost:8000/docs"
echo "🏥 Health check at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

python hvac_analysis_service.py

