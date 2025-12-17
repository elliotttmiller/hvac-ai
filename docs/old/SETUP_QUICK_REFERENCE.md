# HVAC AI Platform - Quick Setup Reference

## Prerequisites Check

```bash
node --version    # Should be 18+
python3 --version # Should be 3.9+
```

## Quick Setup (5 Steps)

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/elliotttmiller/hvac-ai.git
cd hvac-ai

# Install frontend dependencies
npm install

# Install backend dependencies
cd python-services
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env.local

# Edit .env.local and set these REQUIRED variables:
# NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
# MODEL_PATH=./models/sam_hvac_finetuned.pth
```

**Important:** The `NEXT_PUBLIC_API_BASE_URL` must be set for the frontend to communicate with the backend.

### 3. Get SAM Model File

You need a SAM model checkpoint file (.pth) to run analysis features:

```bash
# Create models directory
mkdir -p models

# Option A: If you have a trained model, copy it
cp /path/to/your/sam_model.pth models/sam_hvac_finetuned.pth

# Option B: Download a pretrained SAM model (if available)
# [Add download instructions here when model is available]

# Verify file exists
ls -lh models/sam_hvac_finetuned.pth
```

**Note:** Without this file, the backend will start but analysis features will not work.

### 4. Validate Setup

```bash
# Run the setup validation script
npm run check

# Or directly:
./scripts/check-setup.sh
```

This will check:
- ✓ Node.js and Python installed
- ✓ Dependencies installed
- ✓ Environment variables configured
- ✓ Model file exists
- ✓ Services running (if started)

### 5. Start Services

**Terminal 1 - Backend:**
```bash
cd python-services
source venv/bin/activate  # On Windows: venv\Scripts\activate
python hvac_analysis_service.py
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

## Verify Setup

### Check Backend Health

Open in browser: http://localhost:8000/health

**Good Response (Model Loaded):**
```json
{
  "status": "healthy",
  "service": "running",
  "model_loaded": true,
  "model_path": "./models/sam_hvac_finetuned.pth",
  "device": "cuda"
}
```

**Warning Response (No Model):**
```json
{
  "status": "degraded",
  "service": "running",
  "model_loaded": false,
  "error": "Model file not found at: ./models/sam_hvac_finetuned.pth",
  "troubleshooting": {...}
}
```

### Check Frontend

Open in browser: http://localhost:3000

- If you see a **red warning banner** → Backend connection issue
- If you see the **upload interface** → Setup successful!

## Common Issues

| Issue | Solution |
|-------|----------|
| "API URL not configured" | Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` and restart frontend |
| "Cannot connect to backend" | Start backend server in python-services directory |
| "Model file not found" | Place SAM model .pth file in `models/` directory |
| "SAM engine not available" | Check MODEL_PATH points to valid .pth file |

For detailed troubleshooting, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

## Environment Variables Reference

### Frontend (Next.js)

Must be in `.env.local` or `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | **YES** | - | Backend API URL (e.g., http://localhost:8000) |
| `NEXT_PUBLIC_SUPABASE_URL` | No | - | Supabase project URL (if using auth) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | No | - | Supabase anonymous key (if using auth) |

### Backend (Python)

Must be in `.env` or `.env.local`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_PATH` | **YES** | - | Path to SAM model .pth file |
| `SAM_MODEL_PATH` | No | - | Backward compatibility alias for MODEL_PATH |

**Note:** Frontend variables starting with `NEXT_PUBLIC_` are embedded at build time. Restart dev server after changes!

## Quick Commands Reference

```bash
# Validate setup
npm run check

# Start both services (convenience script)
npm run dev:all

# Start frontend only
npm run dev

# Start backend only
cd python-services && python hvac_analysis_service.py

# Check backend status
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

## What's Working After Setup?

✅ **Backend:**
- `/health` endpoint showing model status
- `/docs` interactive API documentation
- Graceful startup (even without model)
- Clear error messages

✅ **Frontend:**
- API connectivity check on page load
- Warning banners when backend unavailable
- Upload interface (works when backend is healthy)
- Clear error messages with troubleshooting steps

## Next Steps

1. Upload a test image to verify analysis works
2. Try the SAM analysis features
3. Review [API Documentation](http://localhost:8000/docs)
4. Read [Full Documentation](./README.md) for advanced features

## Need Help?

- Quick issues → [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- Setup details → [GETTING_STARTED.md](./GETTING_STARTED.md)
- Architecture → [ARCHITECTURE.md](./ARCHITECTURE.md)
- SAM features → [SAM_DEPLOYMENT.md](./SAM_DEPLOYMENT.md)
