#!/bin/bash
# HVAC AI Platform Python Service Startup Script

echo "🚀 Starting HVAC AI Analysis Service..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/Update dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Start the service
echo "✅ Starting FastAPI service on http://localhost:8000"
echo "📖 API Documentation available at http://localhost:8000/docs"
echo ""

python hvac_analysis_service.py
