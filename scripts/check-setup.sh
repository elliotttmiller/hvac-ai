#!/bin/bash
# Setup validation script for HVAC AI Platform
# This script checks if all required components are properly configured

set -e

echo "🔍 HVAC AI Platform - Setup Validation"
echo "======================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

errors=0
warnings=0

# Function to print colored status
print_status() {
    if [ "$1" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $2"
    elif [ "$1" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $2"
        ((warnings++))
    else
        echo -e "${RED}✗${NC} $2"
        ((errors++))
    fi
}

# 1. Check Node.js
echo "1. Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "OK" "Node.js installed: $NODE_VERSION"
else
    print_status "ERROR" "Node.js not found. Please install Node.js 18+"
fi
echo ""

# 2. Check Python
echo "2. Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    # Extract version number (e.g., "Python 3.10.0" -> "3.10")
    PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 9 ]; then
        print_status "OK" "Python installed: $PYTHON_VERSION"
    else
        print_status "WARN" "Python $PYTHON_VERSION found, but 3.9+ recommended"
    fi
else
    print_status "ERROR" "Python 3 not found. Please install Python 3.9+"
fi
echo ""

# 3. Check frontend dependencies
echo "3. Checking frontend dependencies..."
if [ -d "node_modules" ]; then
    print_status "OK" "Node modules installed"
else
    print_status "WARN" "Node modules not installed. Run: npm install"
fi
echo ""

# 4. Check Python dependencies
echo "4. Checking Python dependencies..."
if [ -d "python-services/venv" ]; then
    print_status "OK" "Python virtual environment exists"
else
    print_status "WARN" "Python venv not found. Run: cd python-services && python3 -m venv venv"
fi
echo ""

# 5. Check environment files
echo "5. Checking environment configuration..."
if [ -f ".env.local" ] || [ -f ".env" ]; then
    if [ -f ".env.local" ]; then
        ENV_FILE=".env.local"
    else
        ENV_FILE=".env"
    fi
    print_status "OK" "Environment file found: $ENV_FILE"
    
    # Check for required frontend variables
    if grep -q "NEXT_PUBLIC_API_BASE_URL" "$ENV_FILE"; then
        API_URL=$(grep "NEXT_PUBLIC_API_BASE_URL" "$ENV_FILE" | cut -d'=' -f2)
        if [ -z "$API_URL" ]; then
            print_status "WARN" "NEXT_PUBLIC_API_BASE_URL is empty in $ENV_FILE"
        else
            print_status "OK" "NEXT_PUBLIC_API_BASE_URL configured: $API_URL"
        fi
    else
        print_status "ERROR" "NEXT_PUBLIC_API_BASE_URL not found in $ENV_FILE"
    fi
    
    # Check for MODEL_PATH
    if grep -q "MODEL_PATH" "$ENV_FILE"; then
        MODEL_PATH=$(grep "^MODEL_PATH" "$ENV_FILE" | cut -d'=' -f2)
        if [ -z "$MODEL_PATH" ]; then
            print_status "WARN" "MODEL_PATH is empty in $ENV_FILE"
        else
            print_status "OK" "MODEL_PATH configured: $MODEL_PATH"
            
            # Check if model file exists
            if [ -f "$MODEL_PATH" ]; then
                MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
                print_status "OK" "Model file found: $MODEL_SIZE"
            else
                print_status "ERROR" "Model file not found at: $MODEL_PATH"
            fi
        fi
    else
        print_status "ERROR" "MODEL_PATH not found in $ENV_FILE"
    fi
else
    print_status "ERROR" "No .env or .env.local file found. Copy .env.example to .env"
fi
echo ""

# 6. Check backend structure
echo "6. Checking backend structure..."
if [ -f "python-services/hvac_analysis_service.py" ]; then
    print_status "OK" "Backend service file exists"
else
    print_status "ERROR" "Backend service file not found"
fi

if [ -f "python-services/requirements.txt" ]; then
    print_status "OK" "Requirements file exists"
else
    print_status "ERROR" "Requirements file not found"
fi

if [ -f "python-services/core/ai/sam_inference.py" ]; then
    print_status "OK" "SAM inference module exists"
else
    print_status "ERROR" "SAM inference module not found"
fi
echo ""

# 7. Check if backend is running
echo "7. Checking if services are running..."
if [ -f ".env.local" ]; then
    API_URL=$(grep "NEXT_PUBLIC_API_BASE_URL" ".env.local" | cut -d'=' -f2)
elif [ -f ".env" ]; then
    API_URL=$(grep "NEXT_PUBLIC_API_BASE_URL" ".env" | cut -d'=' -f2)
else
    API_URL="http://localhost:8000"
fi

# Store health response to avoid multiple curl calls
HEALTH=$(curl -s "${API_URL}/health" 2>/dev/null)
if [ -n "$HEALTH" ]; then
    if echo "$HEALTH" | grep -q '"status":"healthy"'; then
        print_status "OK" "Backend is running and healthy at $API_URL"
    else
        print_status "WARN" "Backend is running but model not loaded. Check backend logs."
    fi
else
    print_status "WARN" "Backend not responding at $API_URL (may not be started yet)"
fi

# Check frontend
if curl -s "http://localhost:3000" > /dev/null 2>&1; then
    print_status "OK" "Frontend is running at http://localhost:3000"
else
    print_status "WARN" "Frontend not responding (may not be started yet)"
fi
echo ""

# Summary
echo "======================================="
echo "Summary:"
echo ""
if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "Your setup looks good. You can start the services:"
    echo "  Terminal 1: cd python-services && python hvac_analysis_service.py"
    echo "  Terminal 2: npm run dev"
elif [ $errors -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup complete with $warnings warning(s)${NC}"
    echo "Review the warnings above. The platform may still work but some features might be limited."
else
    echo -e "${RED}✗ Found $errors error(s) and $warnings warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before starting the platform."
    echo "See docs/TROUBLESHOOTING.md for detailed help."
fi
echo ""
echo "For help, see:"
echo "  - docs/GETTING_STARTED.md"
echo "  - docs/TROUBLESHOOTING.md"
echo "  - docs/SAM_DEPLOYMENT.md"
