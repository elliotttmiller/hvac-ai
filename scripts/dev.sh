#!/bin/bash
# Development start script for HVAC AI Platform

echo "🚀 Starting HVAC AI Platform Development Environment"
echo "===================================================="

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "⚠️  Warning: .env.local not found. Using default configuration."
    echo "Run 'npm run setup' or './scripts/setup.sh' for initial setup."
fi

# Start backend in background
echo ""
echo "🐍 Starting Python backend service..."
cd services/hvac-analysis

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Run './scripts/setup.sh' first."
    exit 1
fi

python hvac_analysis_service.py &
BACKEND_PID=$!

cd ..

# Wait for backend to start
echo "Waiting for backend to be ready..."
sleep 3

# Start frontend
echo ""
echo "⚛️  Starting Next.js frontend..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Services started!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Handle cleanup on exit
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM

# Wait for processes
wait
