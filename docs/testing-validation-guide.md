# Infrastructure Integration - Testing & Validation Guide

## Prerequisites

Before starting the platform, ensure dependencies are installed:

```bash
# Navigate to repository root
cd /path/to/hvac-ai

# Install Python dependencies
pip install -r services/requirements.txt

# Install Node.js dependencies (for frontend)
npm install
```

## Environment Configuration

Create a `.env.local` file in the repository root:

```bash
# Model Configuration
MODEL_PATH=/path/to/your/ai_model/best.pt
SKIP_MODEL=0  # Set to 1 to skip model loading for testing

# Pricing Configuration (optional)
ENABLE_PRICING=1  # Set to 0 to disable pricing integration
DEFAULT_LOCATION=default  # Default location for regional pricing

# Ray Configuration (optional)
RAY_USE_GPU=1
FORCE_CPU=0

# Port Configuration (optional)
PORT=8000
FRONTEND_PORT=3000
```

## Starting the Platform

### Option 1: Full Stack (Frontend + Backend)
```bash
python scripts/start_unified.py
```

### Option 2: Backend Only
```bash
python scripts/start_unified.py --no-frontend
```

## Expected Startup Sequence

### 1. Ray Serve Initialization
```
[AI-ENGINE] Initializing Ray cluster...
[AI-ENGINE] Starting Serve instance...
[AI-ENGINE] Deploying Inference Graph...
```

### 2. Model Loading
```
[LOAD] Loading detection model from: /path/to/model
[OK] Model loaded successfully
   Classes: ['vav_box', 'thermostat', 'diffuser_square', ...]
   Supports OBB: True
```

### 3. Service Initialization
```
[AI-ENGINE] Initializing ObjectDetectorDeployment...
[AI-ENGINE] ObjectDetectorDeployment ready
[AI-ENGINE] Initializing TextExtractorDeployment...
[AI-ENGINE] TextExtractorDeployment ready
[AI-ENGINE] Initializing InferenceGraphIngress...
[OK] Pricing Engine initialized with 12 component types
[AI-ENGINE] InferenceGraphIngress ready
```

### 4. Ready State
```
[AI-ENGINE] Inference Graph Deployed Successfully!
[AI-ENGINE]   - ObjectDetector (GPU)
[AI-ENGINE]   - TextExtractor (GPU)
[AI-ENGINE]   - Ingress (HTTP :8000)

Platform running. Press Ctrl+C to stop all services.
```

## Testing the Integration

### Basic Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy"
}
```

### Test Inference with Pricing

```bash
# Prepare a test image (base64-encoded)
python -c "
import base64
import requests

# Read and encode image
with open('test_blueprint.png', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post(
    'http://localhost:8000/analyze',
    json={
        'image_base64': img_data,
        'project_id': 'TEST-001',
        'location': 'Chicago, IL',
        'conf_threshold': 0.5
    }
)

print(response.json())
"
```

Expected response structure:
```json
{
  "status": "success",
  "total_detections": 15,
  "detections": [
    {
      "label": "vav_box",
      "score": 0.92,
      "bbox": [100, 200, 300, 400],
      "textContent": "VAV-01"
    }
  ],
  "quote": {
    "quote_id": "Q-TEST-001",
    "currency": "USD",
    "summary": {
      "subtotal_materials": 12450.00,
      "subtotal_labor": 8750.00,
      "total_cost": 21200.00,
      "final_price": 25440.00
    },
    "line_items": [...]
  }
}
```

## Troubleshooting

### Issue: Import Errors
**Symptoms:**
```
ImportError: No module named 'pricing'
```

**Solution:**
Ensure dependencies are installed:
```bash
pip install -r services/requirements.txt
```

### Issue: Model Not Found
**Symptoms:**
```
WARNING: MODEL_PATH not set or invalid
```

**Solution:**
Set MODEL_PATH in `.env.local`:
```bash
MODEL_PATH=/absolute/path/to/ai_model/best.pt
```

Or skip model loading for testing:
```bash
SKIP_MODEL=1
```

### Issue: Pricing Disabled
**Symptoms:**
```
[AI-ENGINE] Pricing Engine disabled (not available or disabled by config)
```

**Solution:**
1. Check that catalog exists: `services/hvac-domain/pricing/catalog.json`
2. Verify pydantic is installed: `pip install pydantic`
3. Check logs for import errors

### Issue: Windows Encoding Errors
**Symptoms:**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Solution:**
This should be fixed! All emoji characters have been removed.
If you still see this, set:
```bash
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
```

### Issue: Port Already in Use
**Symptoms:**
```
Address already in use: 8000
```

**Solution:**
1. Kill existing processes:
   ```bash
   # Linux/Mac
   lsof -ti:8000 | xargs kill -9
   
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <pid> /F
   ```
2. Or use a different port:
   ```bash
   PORT=8001 python scripts/start_unified.py
   ```

## Graceful Shutdown

To stop the platform:
1. Press `Ctrl+C` in the terminal
2. Wait for graceful shutdown:
   ```
   [STOP] Stopping platform...
   Terminating processes...
   [OK] Shutdown complete.
   ```

## Verifying No Zombie Processes

After shutdown, verify no processes remain:

```bash
# Linux/Mac
ps aux | grep "ray\|python.*start"

# Windows
tasklist | findstr "python.exe"
```

## Performance Monitoring

### Ray Dashboard
Access the Ray Dashboard for monitoring:
```
http://localhost:8265
```

### Memory Usage
```bash
# Check GPU memory
nvidia-smi

# Check system memory
free -h  # Linux
```

### Response Times
Typical latency (varies by image size):
- Detection: 200-500ms
- OCR (per text region): 50-100ms  
- Pricing: 5-10ms
- **Total**: 500-1000ms for typical blueprint

## Integration Test

Run the integration test:
```bash
python scripts/test_pricing_integration.py
```

Expected output:
```
============================================================
Testing Pricing Engine Integration
============================================================

[TEST 1] Testing PricingEngine import...
[OK] Successfully imported PricingEngine

[TEST 2] Testing PricingEngine initialization...
[OK] PricingEngine initialized with 12 components

[TEST 3] Testing quote generation from detections...
[OK] Quote generated successfully
     Quote ID: Q-TEST-001
     Line Items: 3
     Materials: $2275.00
     Labor: $3187.50
     Final Price: $6555.00

[TEST 4] Testing fallback pricing for unknown components...
[OK] Fallback pricing works
     Default material cost: $100.00

[TEST 5] Testing inference_graph can import pricing modules...
[OK] inference_graph import pattern works

============================================================
All tests passed!
============================================================
```

## Production Considerations

Before deploying to production:

1. **Security:**
   - Add authentication (API keys, OAuth)
   - Enable HTTPS/TLS
   - Implement rate limiting

2. **Scalability:**
   - Use Ray cluster for distributed inference
   - Add Redis caching for quotes
   - Implement load balancing

3. **Monitoring:**
   - Set up logging aggregation
   - Add metrics (Prometheus/Grafana)
   - Configure alerts

4. **Data:**
   - Back up catalog.json
   - Version control pricing data
   - Implement audit trails

5. **Performance:**
   - Consider ThreadPoolExecutor for pricing
   - Optimize image preprocessing
   - Enable model quantization
