# HVAC AI Platform - Scripts Directory

This directory contains utility scripts for development and deployment.

## Available Scripts

### `setup.sh`
Initial setup script for the development environment.

```bash
./scripts/setup.sh
```

**What it does:**
- Checks for Node.js and Python installations
- Installs frontend dependencies (npm)
- Creates Python virtual environment
- Installs Python dependencies
- Creates .env.local from .env.example
- Sets up necessary directories

### `dev.sh`
Starts both frontend and backend development servers.

```bash
./scripts/dev.sh
```

**What it does:**
- Starts Python backend on http://localhost:8000
- Starts Next.js frontend on http://localhost:3000
- Handles graceful shutdown with Ctrl+C

## Usage

### First Time Setup
```bash
# Make scripts executable (if needed)
chmod +x scripts/*.sh

# Run setup
./scripts/setup.sh

# Start development servers
./scripts/dev.sh
```

# Daily Development
```bash
# Start both servers
./scripts/dev.sh

# Or start them separately:
npm run dev                      # Frontend only
cd services/hvac-analysis && python hvac_analysis_service.py  # Backend only
```

## Adding New Scripts

When adding new scripts:
1. Make them executable: `chmod +x scripts/your-script.sh`
2. Add documentation to this README
3. Follow the existing naming conventions
4. Include error handling and helpful messages
