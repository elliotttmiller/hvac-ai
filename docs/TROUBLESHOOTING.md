# Troubleshooting Guide

Common issues and solutions for HVAC AI Platform.

## Table of Contents

- [Backend Issues](#backend-issues)
- [Frontend Issues](#frontend-issues)
- [Model Issues](#model-issues)
- [Performance Issues](#performance-issues)
- [Deployment Issues](#deployment-issues)

## Backend Issues

### Backend won't start

#### Symptom
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
cd python-services
source venv/bin/activate
pip install -r requirements.txt
```

#### Symptom
```
❌ MODEL_PATH invalid: None
```

**Solution:**
1. Create `.env` file in project root if it doesn't exist:
```bash
cp .env.example .env
```

2. Set MODEL_PATH:
```env
MODEL_PATH=./models/your-model.pt
```

3. Ensure the model file exists:
```bash
ls -la ./models/
```

#### Symptom
```
RuntimeError: Could not initialize YOLO model
```

**Solution:**
- Check if model file is corrupted: try downloading again
- Verify model is compatible with ultralytics YOLO
- Check Python/PyTorch versions match model requirements

### Backend crashes during inference

#### Symptom
```
CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** (if processing multiple images)
2. **Use CPU instead:**
```python
# In yolo_inference.py, force CPU
device = 'cpu'
```

3. **Reduce image resolution** before processing:
```python
# Resize large images before analysis
max_size = 2048
if width > max_size or height > max_size:
    scale = max_size / max(width, height)
    image = cv2.resize(image, None, fx=scale, fy=scale)
```

4. **Free GPU memory:**
```bash
# Kill other processes using GPU
nvidia-smi
kill <PID>
```

### Backend runs but analysis fails

#### Symptom
```
503 Service Unavailable: Model not loaded
```

**Solution:**
- Check health endpoint: `curl http://localhost:8000/health`
- If `model_loaded: false`, check:
  1. MODEL_PATH is correct
  2. Model file exists and is readable
  3. Backend logs for initialization errors

#### Symptom
```
400 Bad Request: Invalid image file
```

**Solution:**
- Verify image format (PNG, JPG, TIFF supported)
- Check file is not corrupted
- Ensure file size < 500MB
- Try converting image to PNG first

## Frontend Issues

### Frontend won't start

#### Symptom
```
Error: Cannot find module 'next'
```

**Solution:**
```bash
npm install
```

#### Symptom
```
Port 3000 is already in use
```

**Solutions:**

1. **Kill existing process:**
```bash
# Linux/Mac
lsof -ti:3000 | xargs kill -9

# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

2. **Use different port:**
```bash
PORT=3001 npm run dev
```

### Frontend can't connect to backend

#### Symptom
```
Failed to fetch
Network error
```

**Solutions:**

1. **Verify backend is running:**
```bash
curl http://localhost:8000/health
```

2. **Check environment variable:**
```bash
# In .env or .env.local
NEXT_PUBLIC_AI_SERVICE_URL=http://localhost:8000
```

3. **Check for CORS issues** (in browser console):
- Backend should allow frontend origin
- Check `CORSMiddleware` configuration in `hvac_analysis_service.py`

4. **Check firewall:**
```bash
# Test connectivity
telnet localhost 8000
```

### Upload fails

#### Symptom
```
File size exceeds 500MB limit
```

**Solution:**
- Reduce image resolution before upload
- Use image compression tools
- Consider increasing limit in code (both frontend and backend)

#### Symptom
```
Please upload a valid blueprint file
```

**Solution:**
- Ensure file extension is: .png, .jpg, .jpeg, .tiff, .pdf, .dwg, or .dxf
- Check file is not corrupted
- Try converting to PNG format

### Analysis hangs or times out

#### Symptom
Analysis starts but never completes

**Solutions:**

1. **Check backend logs** for errors
2. **Verify backend is processing:**
```bash
# Check CPU/GPU usage
top  # or nvidia-smi for GPU
```

3. **Try smaller image** to verify it's not a timeout
4. **Check streaming connection** in browser DevTools Network tab

## Model Issues

### No detections found

#### Symptom
Analysis completes but finds 0 components

**Solutions:**

1. **Lower confidence threshold:**
   - In UI: adjust confidence slider
   - Default is 0.50, try 0.25 or 0.30

2. **Verify image quality:**
   - Ensure blueprint is clear and not too low resolution
   - Check that components are visible

3. **Check model training:**
   - Verify model was trained on similar blueprints
   - Check model classes match expected components

4. **Test with known good image:**
   - Use a test image you know has components
   - Verify model can detect anything

### Wrong detections

#### Symptom
Model detects text as components or misclassifies objects

**Solutions:**

1. **Check aspect ratio filter:**
   - Code already filters out wide objects (aspect ratio > 3.0)
   - Adjust in `yolo_inference.py` if needed

2. **Increase confidence threshold:**
   - Higher threshold = fewer false positives
   - Try 0.60 or 0.70

3. **Retrain model:**
   - Use more diverse training data
   - Add hard negative examples

### Model loads but gives poor results

#### Symptom
Inconsistent or low-quality detections

**Solutions:**

1. **Verify model version:**
```bash
# Check model file
ls -lh ./models/
# Should be recent and appropriate size
```

2. **Check preprocessing:**
   - Ensure image preprocessing matches training
   - Verify normalization is correct

3. **Test inference parameters:**
```python
# In yolo_inference.py, try adjusting:
conf=0.25,  # Lower to get more detections
iou=0.45,   # Adjust NMS threshold
```

## Performance Issues

### Slow inference

#### Symptom
Analysis takes > 30 seconds per image

**Solutions:**

1. **Use GPU acceleration:**
```bash
# Verify CUDA available
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Reduce image size:**
   - Process at lower resolution
   - YOLO works well at 640-1280px

3. **Optimize model:**
   - Export to ONNX or TensorRT
   - Use model pruning/quantization

4. **Check system resources:**
```bash
top
nvidia-smi  # For GPU
```

### High memory usage

#### Symptom
System runs out of RAM or VRAM

**Solutions:**

1. **Close other applications**

2. **Process smaller images:**
```python
# Resize before processing
max_size = 1280
```

3. **Clear cache periodically:**
```python
# In Python
import gc
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

4. **Use CPU if GPU memory insufficient:**
```python
device = 'cpu'
```

### Frontend lag

#### Symptom
UI becomes unresponsive during analysis

**Solutions:**

1. **Use streaming endpoint:**
   - Already implemented
   - Shows progress updates

2. **Reduce visualization complexity:**
   - Disable fill when many segments
   - Lower rendering quality if needed

3. **Clear browser cache:**
```javascript
// In browser console
localStorage.clear();
sessionStorage.clear();
```

## Deployment Issues

### Build failures

#### Symptom
```
npm run build fails
```

**Solutions:**

1. **Clear cache:**
```bash
rm -rf .next node_modules
npm install
npm run build
```

2. **Check TypeScript errors:**
```bash
npx tsc --noEmit
```

3. **Fix linting errors:**
```bash
npm run lint
```

### Production performance

#### Symptom
Slow in production compared to development

**Solutions:**

1. **Enable production optimizations:**
```bash
NODE_ENV=production npm run build
npm run start
```

2. **Use CDN for static assets**

3. **Enable caching:**
   - Redis for API responses
   - Browser caching for images

4. **Use production WSGI server:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  hvac_analysis_service:app
```

### CORS errors in production

#### Symptom
```
CORS policy: No 'Access-Control-Allow-Origin' header
```

**Solution:**

Update `hvac_analysis_service.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Getting More Help

### Check logs

**Backend logs:**
```bash
cd python-services
python hvac_analysis_service.py 2>&1 | tee backend.log
```

**Frontend logs:**
- Check browser console (F12)
- Check terminal where `npm run dev` is running

### Enable debug mode

**Backend:**
```python
# In hvac_analysis_service.py
logging.basicConfig(level=logging.DEBUG)
```

**Frontend:**
```javascript
// In browser console
localStorage.setItem('debug', 'true');
```

### Diagnostic commands

```bash
# Check Python packages
pip list | grep -E "(torch|ultralytics|fastapi)"

# Check Node packages
npm list next react

# Check system resources
free -h              # RAM
df -h                # Disk
nvidia-smi           # GPU

# Test API
curl -v http://localhost:8000/health
```

### Report an issue

If you can't resolve the issue:

1. **Collect information:**
   - Error messages (full stack trace)
   - System info (OS, Python version, Node version)
   - Configuration (`.env` file, sanitized)
   - Steps to reproduce

2. **Check existing issues:**
   - [GitHub Issues](https://github.com/elliotttmiller/hvac-ai/issues)

3. **Create new issue with:**
   - Clear title describing the problem
   - Complete error message
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

---

**Still stuck? Check the [Getting Started Guide](./GETTING_STARTED.md) or review [Documentation](./README.md)**
