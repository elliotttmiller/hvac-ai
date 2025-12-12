# HVAC AI Platform - Troubleshooting Guide

This guide helps you diagnose and fix common issues with image upload and SAM model analysis.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Frontend Issues](#frontend-issues)
3. [Backend Issues](#backend-issues)
4. [Model Loading Issues](#model-loading-issues)
5. [Network/CORS Issues](#networkcors-issues)

---

## Quick Diagnostics

### 1. Check Backend Health

Visit the health endpoint in your browser:
```
http://localhost:8000/health
```

**Healthy Response:**
```json
{
  "status": "healthy",
  "service": "running",
  "model_loaded": true,
  "model_path": "./models/sam_hvac_finetuned.pth",
  "device": "cuda",
  "timestamp": 1234567890.123
}
```

**Unhealthy Response:**
```json
{
  "status": "degraded",
  "service": "running",
  "model_loaded": false,
  "error": "Model file not found at: ./models/sam_hvac_finetuned.pth",
  "troubleshooting": {...}
}
```

### 2. Check API Documentation

Visit the interactive API docs:
```
http://localhost:8000/docs
```

This confirms the backend is running and shows all available endpoints.

### 3. Check Frontend Configuration

Open the browser console on your frontend page. Look for errors or warnings about:
- "API URL not configured"
- "Cannot connect to backend"
- "Backend service unavailable"

---

## Frontend Issues

### Issue: "API URL not configured"

**Symptoms:**
- Red warning banner appears on the page
- Cannot upload images or analyze diagrams
- Toast notification: "API URL not configured"

**Cause:**
The `NEXT_PUBLIC_API_BASE_URL` environment variable is not set.

**Solution:**

1. Create `.env.local` in the project root:
   ```bash
   cp .env.example .env.local
   ```

2. Edit `.env.local` and set:
   ```env
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   ```

3. Restart the frontend server:
   ```bash
   npm run dev
   ```

**Important:** Environment variables starting with `NEXT_PUBLIC_` are embedded at build time. You **must** restart the dev server after changing them.

### Issue: "Cannot connect to backend"

**Symptoms:**
- Red warning banner appears
- Toast notification about connection failure
- Browser console shows network errors

**Causes & Solutions:**

1. **Backend not running:**
   - Start the backend: `cd python-services && python hvac_analysis_service.py`
   - Verify it's running: visit `http://localhost:8000`

2. **Wrong URL configured:**
   - Check `.env.local` has the correct URL
   - Default should be `http://localhost:8000` for local dev
   - Verify the backend is actually listening on that port

3. **Firewall blocking connection:**
   - Check your firewall settings
   - Try disabling firewall temporarily to test

### Issue: Images don't render or upload

**Symptoms:**
- Image appears to upload but doesn't show in the canvas
- Canvas remains blank

**Solutions:**

1. **Check browser console** for JavaScript errors
2. **Check file format:** Only PNG, JPG, JPEG are supported
3. **Check file size:** Very large images may cause issues
4. **Try a different browser** to rule out browser-specific issues

---

## Backend Issues

### Issue: Backend won't start

**Symptoms:**
- Python error on startup
- ImportError or ModuleNotFoundError

**Solutions:**

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.9 or higher
   ```

2. **Reinstall dependencies:**
   ```bash
   cd python-services
   pip install -r requirements.txt
   ```

3. **Check virtual environment:**
   ```bash
   # Create and activate venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Issue: "SAM engine is not available" (503 error)

**Symptoms:**
- Backend starts but API returns 503 errors
- Health endpoint shows `"model_loaded": false`
- Backend logs show model loading errors

**Cause:**
The SAM model file is missing or MODEL_PATH is not configured.

**Solutions:**

See [Model Loading Issues](#model-loading-issues) section below.

---

## Model Loading Issues

### Issue: MODEL_PATH not set

**Symptoms:**
Backend logs show:
```
❌ CRITICAL: MODEL_PATH environment variable is not set!
```

**Solution:**

1. Create `.env` in project root (or `.env.local`):
   ```bash
   cp .env.example .env
   ```

2. Set MODEL_PATH in `.env`:
   ```env
   MODEL_PATH=./models/sam_hvac_finetuned.pth
   ```

3. Restart the backend

### Issue: Model file not found

**Symptoms:**
Backend logs show:
```
❌ Cannot load SAM engine: Model file not found at: ./models/sam_hvac_finetuned.pth
```

**Cause:**
The model file doesn't exist at the specified path.

**Solutions:**

1. **Create the models directory:**
   ```bash
   mkdir -p models
   ```

2. **Obtain the model file:**
   - If you trained a custom model, copy the `.pth` file to `models/`
   - If using a pretrained model, download it and place it in `models/`
   - The file must be a valid PyTorch checkpoint (`.pth` file)

3. **Verify the file exists:**
   ```bash
   ls -lh models/sam_hvac_finetuned.pth
   ```

4. **Update MODEL_PATH if needed:**
   If your model has a different name or location:
   ```env
   MODEL_PATH=/absolute/path/to/your/model.pth
   ```

### Issue: Model loads but crashes

**Symptoms:**
- Backend crashes during model loading
- Python traceback with torch/CUDA errors

**Common Causes & Solutions:**

1. **CUDA/GPU issues:**
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   
   If False, the model will run on CPU (slower but works).
   If you want GPU acceleration, ensure CUDA is properly installed.

2. **Corrupted model file:**
   - Re-download or re-export the model
   - Verify file size is reasonable (SAM models are ~350-2500MB)

3. **Incompatible PyTorch version:**
   - Check PyTorch version: `pip show torch`
   - Model might require a specific PyTorch version
   - Try updating: `pip install --upgrade torch`

4. **Out of memory:**
   - SAM models require significant RAM/VRAM
   - Close other applications
   - Use a smaller model variant if available

---

## Network/CORS Issues

### Issue: CORS errors in browser console

**Symptoms:**
Browser console shows:
```
Access to fetch at 'http://localhost:8000/api/analyze' from origin 'http://localhost:3000' 
has been blocked by CORS policy
```

**Cause:**
Backend CORS configuration is too restrictive.

**Solution:**

The backend already allows all origins (`allow_origins=["*"]`), so this shouldn't happen. If it does:

1. **Verify backend CORS middleware** in `python-services/hvac_analysis_service.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Check for proxy/reverse proxy** that might be blocking requests

3. **Try without credentials:**
   In the fetch call, remove credentials if present

### Issue: Requests time out

**Symptoms:**
- Requests hang indefinitely
- Eventually fail with timeout error

**Solutions:**

1. **For count/analysis operations:** These can take time (30-120+ seconds)
   - This is normal for large images or complex analysis
   - Wait for the operation to complete
   - Check backend logs to see progress

2. **Increase timeout:**
   The count endpoint has a configurable timeout (default 120s)

3. **Optimize image size:**
   - Resize very large images before upload
   - Recommended max: 2048x2048 pixels

---

## Still Having Issues?

### Collect Debug Information

1. **Backend logs:**
   - Check terminal where backend is running
   - Look for ERROR or WARNING messages

2. **Frontend console:**
   - Open browser DevTools (F12)
   - Check Console tab for errors
   - Check Network tab for failed requests

3. **Health check output:**
   - Visit `/health` endpoint
   - Copy the full JSON response

4. **Environment configuration:**
   ```bash
   # Check frontend env
   cat .env.local | grep -v "SECRET\|KEY"
   
   # Check backend env
   cat .env | grep -v "SECRET\|KEY"
   ```

### Common Command Summary

```bash
# Start backend (from project root)
cd python-services
python hvac_analysis_service.py

# Start frontend (from project root)
npm run dev

# Check backend health
curl http://localhost:8000/health

# Check backend is running
curl http://localhost:8000/

# Reinstall dependencies
cd python-services && pip install -r requirements.txt
cd .. && npm install
```

---

## Additional Resources

- [Getting Started Guide](./GETTING_STARTED.md)
- [SAM Deployment Guide](./SAM_DEPLOYMENT.md)
- [API Documentation](http://localhost:8000/docs) (when backend is running)
- [Architecture Overview](./ARCHITECTURE.md)
