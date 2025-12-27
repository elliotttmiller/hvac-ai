import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pathlib import Path
# Heavy ML imports are deferred when SKIP_MODEL is enabled
cv2 = None
torch = None
YOLO = None
import logging
import traceback
import time
import platform
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
import uuid
import json
import os
from fastapi import Request

# --- CONFIGURATION ---
# MODEL_PATH can be overridden via the MODEL_PATH env var (or .env.local)
# Update this default to your fallback path if you prefer a hardcoded default.
MODEL_PATH = os.environ.get('MODEL_PATH')

# If MODEL_PATH was not provided via environment, try loading project .env.local
# (useful when running this script directly during development)
if not MODEL_PATH:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        env_file = repo_root / ".env.local"
        if env_file.exists():
            for raw in env_file.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key == 'MODEL_PATH' and val:
                    MODEL_PATH = val
                    os.environ['MODEL_PATH'] = val
                    logger = logging.getLogger('HVAC-Backend')
                    logger.info(f"Loaded MODEL_PATH from .env.local: {MODEL_PATH}")
                    break
    except Exception:
        pass

# Ensure repository root is on sys.path so 'services.*' imports resolve both
# when running from scripts/ and for editor language servers (Pylance/pyright).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PORT = 8000
CONF_THRES = 0.50
IOU_THRES = 0.45
IMG_SIZE = 1024
# HALF will be set after importing torch (if model is loaded); default false

# --- LOGGING SETUP ---
LOG_DIR = REPO_ROOT / "logs"
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

log_file = LOG_DIR / "backend.log"

logger = logging.getLogger("HVAC-Backend")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - [BACKEND] - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Rotating file handler so we have persistent logs to inspect
try:
    fh = RotatingFileHandler(str(log_file), maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except Exception as e:
    logger.warning(f"Could not create rotating file handler for logs: {e}")

app = FastAPI(title="HVAC Local Inference API")

# Allow CORS for localhost frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
HALF = False
pricing_engine = None
last_exception = None
start_time = time.time()

# Dev mode: skip loading heavy ML stack
# Can be enabled via environment variable SKIP_MODEL=1 or CLI arg --no-model
SKIP_MODEL = os.environ.get("SKIP_MODEL", "0") == "1"
# Optional: force CPU even if GPU is available
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"


@app.on_event("startup")
async def load_model():
    global model
    if SKIP_MODEL:
        logger.warning("SKIP_MODEL is enabled — skipping heavy ML imports and model load (dev mode).")
        return

    # Validate MODEL_PATH early and fail fast if missing to avoid silent runtime
    logger.info(f"Resolved MODEL_PATH={MODEL_PATH}")
    if not MODEL_PATH:
        logger.error("MODEL_PATH is not set. Set MODEL_PATH in your environment or .env.local to point to your trained weights.")
        # Fail fast so the launcher can detect the problem
        raise SystemExit(1)

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model not found at: {MODEL_PATH}")
        logger.error("Please update MODEL_PATH in .env.local or export MODEL_PATH before starting the service.")
        # Fail fast so the launcher can detect the problem
        raise SystemExit(1)

    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        # Import heavy dependencies here so they can be skipped in dev mode
        global torch, cv2, YOLO, HALF
        import cv2 as _cv2
        import torch as _torch
        from ultralytics import YOLO as _YOLO
        cv2 = _cv2
        torch = _torch
        YOLO = _YOLO

        # GPU availability
        gpu_available = torch.cuda.is_available()
        HALF = gpu_available and not FORCE_CPU

        try:
            torch_info = f"torch={torch.__version__}, torch.cuda={getattr(torch.version, 'cuda', None)}"
        except Exception:
            torch_info = "torch info unavailable"
        logger.info(f"PyTorch info: {torch_info}; FORCE_CPU={FORCE_CPU}")

        model = YOLO(MODEL_PATH)
        if gpu_available and not FORCE_CPU:
            try:
                model.to('cuda')
                logger.info(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"Could not move model to CUDA, falling back to CPU: {e}")
        else:
            logger.warning("⚠️ Running on CPU (FORCE_CPU set or CUDA not available).")

        # Warmup (may be heavy on CPU)
        try:
            model.predict(np.zeros((640,640,3), dtype=np.uint8), verbose=False, half=HALF)
            logger.info("🔥 Model Warmup Complete")
        except Exception as e:
            logger.warning(f"Model warmup failed (this may be expected on CPU): {e}")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        # capture last exception for diagnostics endpoint
        global last_exception
        last_exception = {"ts": time.time(), "error": str(e), "traceback": traceback.format_exc()}


@app.get("/health")
def health_check():
    status = "healthy" if model else "model_not_loaded"
    return {"status": status, "device": str(model.device) if model else "none"}


async def process_image(file_bytes):
    # If model isn't loaded (either because SKIP_MODEL or failure), raise
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    
    height, width = img.shape[:2]

    results = model.predict(
        img, 
        conf=CONF_THRES, 
        iou=IOU_THRES, 
        imgsz=IMG_SIZE,
        half=HALF
    )
    
    result = results[0]
    detections = []

    # Handle OBB
    if hasattr(result, 'obb') and result.obb is not None:
        for box in result.obb:
            poly = box.xyxyxyxy[0].cpu().numpy().tolist()
            xywhr = box.xywhr[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = result.names[cls_id]

            # Calculate standard bbox for fallback
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            detections.append({
                "id": str(uuid.uuid4()),
                "label": label,
                "score": conf,
                "bbox": bbox_rect,
                "points": poly,
                "obb": {
                    "x_center": xywhr[0],
                    "y_center": xywhr[1],
                    "width": xywhr[2],
                    "height": xywhr[3],
                    "rotation": xywhr[4]
                }
            })
            
    return {
        "success": True,
        "image": {"width": width, "height": height},
        "count": len(detections),
        "detections": detections
    }


# --- Quote generation endpoint (inference-focused) ---
try:
    from services.hvac_domain.pricing import PricingEngine, QuoteRequest  # type: ignore
except Exception:
    PricingEngine = None
    QuoteRequest = None


@app.post("/api/v1/quote/generate")
async def generate_quote(request: Request):
    """Generate a financial quote from HVAC analysis data.

    This mirrors the behavior expected by the frontend. The pricing engine
    will be lazily initialized on first request so the backend can start
    in dev mode without failing if pricing dependencies are missing.
    """
    global pricing_engine
    if PricingEngine is None or QuoteRequest is None:
        # Pricing subsystem not available in this environment
        raise HTTPException(status_code=503, detail="Pricing subsystem unavailable")

    if pricing_engine is None:
        try:
            pricing_engine = PricingEngine()
        except Exception as e:
            logger.error(f"Failed to initialize PricingEngine: {e}")
            raise HTTPException(status_code=503, detail="Pricing engine init failed")

    try:
        payload = await request.json()
        quote_request = QuoteRequest(**payload)
        quote_response = pricing_engine.generate_quote(quote_request)
        # Return serialized model (if pydantic) or dict-like response
        try:
            return quote_response.model_dump()
        except Exception:
            return dict(quote_response)
    except ValueError as e:
        logger.error(f"Quote validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Quote generation failed")
        last_exception = {"ts": time.time(), "error": str(e), "traceback": traceback.format_exc()}
        raise HTTPException(status_code=500, detail=f"Quote generation failed: {e}")


@app.get("/api/v1/quote/available")
def quote_available():
    """Return whether the pricing subsystem is present in this runtime.

    This endpoint is intentionally lightweight: it checks whether the
    PricingEngine import was successful. It does NOT instantiate the
    engine nor load heavy pricing dependencies. Frontend uses this to
    decide whether to enable quoting UI and avoid noisy 503 errors.
    """
    try:
        if PricingEngine is None:
            return JSONResponse(content={"available": False, "reason": "Pricing subsystem not installed"}, status_code=200)
        # If module is importable, report available; deeper init errors will be surfaced on generate
        return JSONResponse(content={"available": True}, status_code=200)
    except Exception as e:
        logger.warning(f"quote_available check failed: {e}")
        last_exception = {"ts": time.time(), "error": str(e), "traceback": traceback.format_exc()}
        return JSONResponse(content={"available": False, "reason": str(e)}, status_code=200)

# Match the path your frontend expects
@app.post("/api/hvac/analyze") 
async def analyze_image(request: Request, file: UploadFile = File(None), image: UploadFile = File(None)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    target_file = file or image
    if not target_file:
        raise HTTPException(status_code=422, detail="No file uploaded")
    
    try:
        contents = await target_file.read()
        response_data = await process_image(contents)
        logger.info(f"Analyzed {target_file.filename}: Found {response_data['count']} items")

        # SSE Stream support
        if request.query_params.get("stream") == "1":
            async def event_generator():
                yield f"data: {json.dumps(response_data)}\n\n"
                yield "event: close\ndata: [DONE]\n\n"
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception(f"Error analyzing image: {e}")
        global last_exception
        last_exception = {"ts": time.time(), "error": str(e), "traceback": traceback.format_exc()}
        raise HTTPException(status_code=500, detail=str(e))


# --- Diagnostics & Request logging middleware ---
def _tail_file(path: Path, lines: int = 200):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # naive tail implementation
            return ''.join(f.readlines()[-lines:])
    except Exception:
        return None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests, measure duration, and capture unhandled exceptions.

    This middleware also records the last exception so the diagnostics endpoint can
    return helpful traces when something goes wrong.
    """
    method = request.method
    url = str(request.url)
    start = time.time()
    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000.0
        logger.info(f"{method} {request.url.path} -> {response.status_code} ({duration:.1f}ms)")
        return response
    except Exception as e:
        duration = (time.time() - start) * 1000.0
        logger.exception(f"Unhandled exception for {method} {request.url.path} after {duration:.1f}ms: {e}")
        global last_exception
        last_exception = {"ts": time.time(), "error": str(e), "traceback": traceback.format_exc()}
        # Re-raise so FastAPI still handles producing an HTTP 500
        raise


@app.get("/api/v1/diagnostics")
def diagnostics():
    """Return lightweight diagnostics about the running backend useful for debugging.

    Includes model and pricing availability, uptime, environment flags, package versions,
    and the last captured exception trace and recent log tail.
    """
    # Check package versions safely
    def _ver(module_name: str):
        try:
            m = __import__(module_name)
            return getattr(m, '__version__', str(m))
        except Exception:
            return None

    python = platform.python_version()
    torch_ver = _ver('torch')
    ultralytics_ver = _ver('ultralytics')

    uptime = time.time() - start_time

    model_loaded = model is not None
    pricing_importable = PricingEngine is not None
    pricing_initialized = pricing_engine is not None

    # Read last chunk of logs to help debug race conditions or startup failures
    recent_logs = _tail_file(log_file, lines=500)

    return JSONResponse(content={
        "status": "ok",
        "uptime_seconds": int(uptime),
        "model_loaded": bool(model_loaded),
        "pricing_importable": bool(pricing_importable),
        "pricing_initialized": bool(pricing_initialized),
        "env": {
            "MODEL_PATH": MODEL_PATH,
            "SKIP_MODEL": SKIP_MODEL,
            "FORCE_CPU": FORCE_CPU,
            "PORT": PORT,
        },
        "packages": {
            "python": python,
            "torch": torch_ver,
            "ultralytics": ultralytics_ver
        },
        "last_exception": last_exception,
        "recent_logs": recent_logs
    })


if __name__ == "__main__":
    # Ensure stdout is flushed immediately for the parent process to read
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_config=None)
