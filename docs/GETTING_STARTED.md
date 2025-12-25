# Getting Started with HVAC AI Platform

Complete guide to set up and run the HVAC AI Blueprint Analysis Platform.

## Prerequisites

### Required Software

- **Node.js** 18+ (for frontend)
- **Python** 3.9+ (for backend)
- **npm** or **yarn** (package manager)
- **Git** (for cloning the repository)

### Optional but Recommended

- **CUDA-capable GPU** for faster inference (NVIDIA GPU with CUDA 11.8+)
- **Virtual environment** tool for Python (venv or conda)

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
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
# Copy the example environment file
cp .env.example .env

# Edit .env and configure the following:
# - MODEL_PATH: Path to your trained YOLO model file
# - NEXT_PUBLIC_AI_SERVICE_URL: Backend URL (http://localhost:8000 for local)
```

**Required Environment Variables:**

```env
# Backend Configuration
MODEL_PATH=./models/your-yolo-model.pt

# Frontend Configuration
NEXT_PUBLIC_AI_SERVICE_URL=http://localhost:8000
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### 3. Start Services

**Option A: Using startup scripts (Recommended)**

```bash
# Terminal 1 - Start Backend
cd python-services
./start.sh

# Terminal 2 - Start Frontend
npm run dev
```

**Option B: Manual start**

```bash
# Terminal 1 - Backend
cd python-services
source venv/bin/activate
python hvac_analysis_service.py

# Terminal 2 - Frontend
npm run dev
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Detailed Setup

### Frontend Setup

1. **Install Node.js dependencies:**

```bash
npm install
```

2. **Configure environment:**

Create `.env.local` or `.env` file:

```env
NEXT_PUBLIC_AI_SERVICE_URL=http://localhost:8000
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_ENFORCE_AUTH=false  # Set to true for production
```

3. **Run development server:**

```bash
npm run dev
```

The frontend will be available at http://localhost:3000

### Backend Setup

1. **Create virtual environment:**

```bash
cd python-services
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Configure environment:**

Create or edit `.env` file in project root:

```env
MODEL_PATH=./models/your-yolo11-model.pt
```

4. **Obtain or train a YOLO model:**

You need a trained YOLO11 model for HVAC component detection. Options:

- **Use pre-trained model**: Place your trained model in `./models/` directory
- **Train your own**: See training documentation (coming soon)

5. **Start the backend service:**

```bash
python hvac_analysis_service.py
```

Or use the startup script:

```bash
./start.sh
```

The backend will be available at http://localhost:8000

## Validating Your Setup

### Quick Validation

Run the setup check script:

```bash
./scripts/check-setup.sh
```

This will verify:
- ✓ Node.js and Python are installed
- ✓ Dependencies are installed
- ✓ Environment files are configured
- ✓ Model file exists and is accessible
- ✓ Services are running

### Manual Validation

1. **Check backend health:**

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",  // or "cpu"
  "num_classes": 2
}
```

2. **Check frontend:**

Open http://localhost:3000 in your browser. You should see the HVAC AI Platform homepage.

3. **Test analysis:**

- Navigate to the Documents page
- Upload a sample blueprint image
- Click "Analyze HVAC System"
- You should see detected components

## Common Issues

### Backend won't start

**Issue:** `MODEL_PATH invalid` or `Model file not found`

**Solution:**
1. Ensure MODEL_PATH is set in `.env`
2. Verify the model file exists at the specified path
3. Check file permissions

```bash
# Check if file exists
ls -la ./models/your-model.pt

# Check .env configuration
cat .env | grep MODEL_PATH
```

### Frontend can't connect to backend

**Issue:** `Failed to fetch` or `503 Service Unavailable`

**Solution:**
1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check NEXT_PUBLIC_AI_SERVICE_URL in `.env`
3. Verify no firewall blocking port 8000

### Model loaded but no detections

**Issue:** Analysis completes but finds 0 components

**Solution:**
1. Check confidence threshold (default 0.50)
2. Ensure image is a valid blueprint
3. Verify model is trained for your specific components
4. Try lowering confidence threshold in UI

### CUDA/GPU issues

**Issue:** `CUDA not available` or GPU memory errors

**Solution:**
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. The system will fall back to CPU if CUDA is unavailable

## Production Deployment

For production deployment:

1. **Build frontend:**

```bash
npm run build
npm run start
```

2. **Use production WSGI server for backend:**

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker hvac_analysis_service:app
```

3. **Configure environment:**

- Set `NODE_ENV=production`
- Enable authentication: `NEXT_PUBLIC_ENFORCE_AUTH=true`
- Use HTTPS
- Configure proper CORS origins

4. **Use reverse proxy:**

Consider using nginx or similar for:
- SSL/TLS termination
- Load balancing
- Static file serving
- Rate limiting

## Next Steps

- 📚 Read the [README](../README.md) for feature overview
- 🎯 Try the [examples](../examples/README.md) for usage patterns
- 🔧 Review [API documentation](http://localhost:8000/docs) for integration
- 📖 See [architecture documentation](./adr/) for technical details

## Getting Help

- Check the [Troubleshooting Guide](./TROUBLESHOOTING.md)
- Review [GitHub Issues](https://github.com/elliotttmiller/hvac-ai/issues)
- Read the [documentation](./README.md)

## Development

### Running tests

```bash
# Frontend tests
npm test

# Backend tests
cd python-services
pytest
```

### Code quality

```bash
# Frontend linting
npm run lint

# Frontend formatting
npm run format

# TypeScript checking
npx tsc --noEmit
```

---

**Happy Building! 🚀**
