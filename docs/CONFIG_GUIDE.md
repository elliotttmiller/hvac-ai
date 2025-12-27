# Configuration Guide - HVAC Cortex Platform

## Overview

The HVAC Cortex platform is fully configured via environment variables from `.env.local`. This guide explains all available configuration options and how they are used throughout the startup pipeline.

## Configuration File: `.env.local`

Located at the project root (`d:\AMD\hvac-ai\.env.local`), this file controls:
- Backend/frontend ports
- Model paths and inference settings
- Ray cluster configuration
- GPU/CPU resource allocation
- Pricing engine settings

### Loading Mechanism

Configuration is loaded at multiple stages:

1. **`scripts/start_unified.py`** - Main launcher
   - Loads `.env.local` 
   - Passes environment to backend subprocess

2. **`scripts/start_ray_serve.py`** - Ray Serve initialization
   - Loads `.env.local` explicitly at startup
   - Extracts: `BACKEND_PORT`, `RAY_ADDRESS`, `RAY_HEAD`, `RAY_USE_GPU`, `SKIP_MODEL`, `FORCE_CPU`
   - Configures Ray cluster and HTTP server accordingly

3. **`services/hvac-ai/inference_graph.py`** - Model initialization
   - Loads `.env.local` at import time
   - Extracts: `SKIP_MODEL`, `FORCE_CPU`, `MODEL_PATH`, `CONF_THRESHOLD`
   - Configures object detector device (GPU/CPU) and model path
   - Applies during startup event for deployment binding

## Configuration Variables

### Port Configuration (CRITICAL: Two Separate Ports!)

⚠️ **Important:** Ray uses TWO separate ports for different purposes:

```
BACKEND_PORT="8000"       # Ray Serve HTTP port - WHERE FRONTEND SENDS REQUESTS
RAY_PORT="10001"          # Ray cluster communication port - FOR RAY INTERNAL PROTOCOL
FRONTEND_PORT="3000"      # Next.js frontend port (default: 3000)
```

**Port Breakdown:**

| Port | Component | Purpose | Protocol |
|------|-----------|---------|----------|
| 3000 | Next.js Frontend | User interface, sends requests to backend | HTTP |
| 8000 | Ray Serve HTTP | **Frontend requests go here**: `/api/hvac/analyze` | HTTP/REST |
| 8265 | Ray Dashboard | Monitoring Ray cluster (optional access) | HTTP |
| 10001 | Ray Cluster | Internal Ray communication, worker-to-worker | Ray Protocol |

**Request Flow:**

```
Frontend (port 3000)
    ↓ POST /api/hvac/analyze
Backend (port 8000) ← BACKEND_PORT
    ↓ Communicates with Ray workers via Ray Protocol (port 10001)
Ray Workers (port 10001) ← RAY_PORT
```

**Used by:**
- `start_ray_serve.py`:
  - `ray.init(_redis_port=RAY_PORT)` - Configures Ray cluster port
  - `serve.start(http_options={"port": BACKEND_PORT})` - Configures HTTP server
- `start_unified.py`: Health check at `http://127.0.0.1:{BACKEND_PORT}/health`
- Frontend: Routes to `http://127.0.0.1:{BACKEND_PORT}/api/hvac/analyze`

---

### Backend Ports (Details)

```
BACKEND_PORT="8000"       # Ray Serve HTTP port (default: 8000)
PORT="8000"               # Fallback if BACKEND_PORT not set
FRONTEND_PORT="3000"      # Next.js frontend port (default: 3000)
```

---

### Model Configuration

```
MODEL_PATH="D:\AMD\hvac-ai\ai_model\models\hvac_obb_l_20251224_214011\weights\best.pt"
CONF_THRESHOLD="0.5"      # Detection confidence threshold (0.0-1.0)
SKIP_MODEL="0"            # Skip model loading (dev/testing): 0=load, 1=skip
FORCE_CPU="0"             # Force CPU inference: 0=auto (CUDA if available), 1=CPU only
```

**Used by:**
- `inference_graph.py` startup event:
  - `model_path` → `ObjectDetectorDeployment.bind(model_path=model_path)`
  - `FORCE_CPU` → Device selection: `device = 'cpu' if FORCE_CPU else 'cuda'`
  - `CONF_THRESHOLD` → Confidence threshold during detection

---

### Ray Cluster Configuration

```
RAY_ADDRESS=""            # Ray cluster address (e.g., "ray://127.0.0.1:10001")
                          # Leave empty for local head node
RAY_HEAD="1"              # Start local Ray head node: 0=no, 1=yes
RAY_PORT="10001"          # Ray cluster port
RAY_USE_GPU="1"           # Enable GPU resources for Ray: 0=no, 1=yes
RAY_NUM_CPUS="4"          # (Optional) CPU resources
RAY_NUM_GPUS="1"          # (Optional) GPU resources
```

**Used by:**
- `start_ray_serve.py`:
  - If `RAY_ADDRESS` is set: `ray.init(address=RAY_ADDRESS)`
  - Otherwise: `ray.init(dashboard_host="127.0.0.1", num_gpus=1 if RAY_USE_GPU else 0)`

---

### Authentication & Services

```
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="nkilDcr27v7ybH3CzKMynLpGgFfKQyJuRWe7296suKQ="
NGROK_AUTHTOKEN="36hBoLt4A3L8yOYt96wKiCxxrwp_5wFbj1Frv6GoHARRQ6H6t"
```

**Used by:**
- Next.js frontend for authentication
- ngrok tunnel (if deployed to cloud)

---

### Service URLs

```
NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
NEXT_PUBLIC_AI_SERVICE_URL="http://127.0.0.1:8000"
LOCAL_API_BASE_URL="http://127.0.0.1:8000"
```

**Used by:**
- Frontend (`src/app/api/hvac/analyze/route.ts`): Routes requests to `http://127.0.0.1:8000/api/hvac/analyze`

---

### Pricing Engine

```
ENABLE_PRICING="1"        # Enable pricing engine: 0=no, 1=yes (default)
```

**Used by:**
- `inference_graph.py` startup event: Only loads pricing if enabled and available

---

## Startup Flow with Configuration

### 1. Start Unified Script

```bash
python scripts/start_unified.py
```

```
[STARTUP] Loading .env.local...
[STARTUP] Backend port: 8000
[STARTUP] Starting Ray Serve with backend_env...
```

### 2. Start Ray Serve Script

```python
# Load .env.local
load_env_file(REPO_ROOT / ".env.local")

# Extract configuration
BACKEND_PORT = int(os.environ.get('BACKEND_PORT', '8000'))
RAY_USE_GPU = os.environ.get('RAY_USE_GPU', '1') == '1'
RAY_ADDRESS = os.environ.get('RAY_ADDRESS', '')

# Configure Ray
if RAY_ADDRESS:
    ray.init(address=RAY_ADDRESS)  # Connect to existing cluster
else:
    ray.init(num_gpus=1 if RAY_USE_GPU else 0)  # Start local head node

# Start HTTP server
serve.start(http_options={"host": "0.0.0.0", "port": BACKEND_PORT})

# Deploy FastAPI app
serve.run(APIServer.bind())
```

### 3. Inference Graph Initialization

```python
# Load .env.local at import time
load_env_file(REPO_ROOT / ".env.local")

# Extract settings
MODEL_PATH = os.environ.get('MODEL_PATH')
FORCE_CPU = os.environ.get('FORCE_CPU', '0') == '1'

# During startup event
model_path = os.getenv("MODEL_PATH", "default/path")
use_gpu = not FORCE_CPU

detector_handle = ObjectDetectorDeployment.bind(
    model_path=model_path,
    conf_threshold=conf_threshold
)
extractor_handle = TextExtractorDeployment.bind(use_gpu=use_gpu)
```

---

## Common Configuration Scenarios

### Development (CPU, No GPU)

```env
FORCE_CPU="1"
SKIP_MODEL="0"
RAY_USE_GPU="0"
BACKEND_PORT="8000"
```

**Result:**
- Object detection runs on CPU
- Ray cluster has no GPU allocation
- Model is loaded for testing

---

### Production (GPU Enabled)

```env
FORCE_CPU="0"
SKIP_MODEL="0"
RAY_USE_GPU="1"
BACKEND_PORT="8000"
MODEL_PATH="/path/to/optimized/model.pt"
```

**Result:**
- Object detection runs on CUDA GPU
- Ray cluster has 1 GPU allocated
- Uses optimized model path

---

### Testing (Skip Model Loading)

```env
SKIP_MODEL="1"
FORCE_CPU="1"
```

**Result:**
- Model is not loaded (fast startup for testing)
- All inference runs on CPU
- Useful for CI/CD or debugging without GPU

---

### Remote Ray Cluster

```env
RAY_ADDRESS="ray://10.0.0.100:10001"
RAY_HEAD="0"
```

**Result:**
- Connects to existing Ray cluster at 10.0.0.100:10001
- Doesn't start local head node
- Useful for distributed setups

---

## Debugging Configuration

### View Loaded Configuration

Check logs during startup:

```
[CONFIG] Ray Serve Configuration:
  Backend Port: 8000
  RAY_ADDRESS: (none - local head node)
  RAY_HEAD: True
  RAY_USE_GPU: True
  SKIP_MODEL: False
  FORCE_CPU: False
  MODEL_PATH: D:\AMD\hvac-ai\ai_model\models\hvac_obb_l_20251224_214011\weights\best.pt

[CONFIG] Inference Graph Configuration:
  SKIP_MODEL: False
  FORCE_CPU: False
  MODEL_PATH: D:\AMD\hvac-ai\ai_model\models\hvac_obb_l_20251224_214011\weights\best.pt
  CONF_THRESHOLD: 0.5
```

### Verify Environment Variables

```bash
# In PowerShell
$env:MODEL_PATH
$env:BACKEND_PORT
$env:RAY_USE_GPU
```

### Trace Configuration Loading

The configuration loading is logged at the start of each script:

1. `start_unified.py`: Logs `.env.local` loading
2. `start_ray_serve.py`: Logs extracted Ray configuration
3. `inference_graph.py`: Logs model and device configuration

---

## Best Practices

1. **Always use `.env.local`** for environment-specific settings
2. **Never commit `.env.local`** to version control (it's in `.gitignore`)
3. **Use sensible defaults** - scripts fall back to defaults if variables not set
4. **Log configuration** - startup scripts log loaded settings for debugging
5. **Separate concerns** - different scripts load only what they need
6. **Document changes** - update this guide when adding new variables

---

## Adding New Configuration Variables

To add a new configuration variable:

1. **Add to `.env.local`**:
   ```env
   MY_NEW_SETTING="value"
   ```

2. **Load in startup script**:
   ```python
   MY_NEW_SETTING = os.environ.get('MY_NEW_SETTING', 'default_value')
   ```

3. **Use in logic**:
   ```python
   if MY_NEW_SETTING == 'value':
       # do something
   ```

4. **Log during startup**:
   ```python
   logger.info(f"  MY_NEW_SETTING: {MY_NEW_SETTING}")
   ```

5. **Document in this guide** (under appropriate section)

---

## Files Modified for Configuration Support

- ✅ `scripts/start_ray_serve.py` - Loads `.env.local`, uses BACKEND_PORT, RAY_ADDRESS, RAY_USE_GPU, etc.
- ✅ `services/hvac-ai/inference_graph.py` - Loads `.env.local`, uses MODEL_PATH, FORCE_CPU, CONF_THRESHOLD
- ✅ `scripts/start_unified.py` - Already loads `.env.local` (no changes needed)

---

## References

- Ray Serve Configuration: https://docs.ray.io/en/latest/serve/
- FastAPI Environment: https://fastapi.tiangolo.com/advanced/settings/
- Next.js Environment: https://nextjs.org/docs/basic-features/environment-variables
