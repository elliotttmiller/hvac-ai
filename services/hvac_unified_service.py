# services/hvac_unified_service.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
import io
import sys
import threading
import queue as _queue
import json
import time
import importlib.util
from typing import Optional, Any

# --- CONFIGURATION ---
# Force unbuffered output for Colab real-time logs
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore
except (AttributeError, ValueError):
    # Not supported on all systems (e.g., Windows stdout might not support reconfigure)
    pass

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("API_SERVER")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICES_ROOT = Path(__file__).resolve().parent
env_file = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_file)

MODEL_PATH = os.getenv("MODEL_PATH")

# Ensure services and hvac-ai directories are in path for imports
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

hvac_ai_path = SERVICES_ROOT / "hvac-ai"
if str(hvac_ai_path) not in sys.path:
    sys.path.insert(0, str(hvac_ai_path))

# Add hvac-domain to path for pricing imports
hvac_domain_path = SERVICES_ROOT / "hvac-domain"
if str(hvac_domain_path) not in sys.path:
    sys.path.insert(0, str(hvac_domain_path))

# Import YOLO inference engine
create_yolo_engine: Optional[Any] = None
try:
    from yolo_inference import create_yolo_engine  # type: ignore
except ImportError as e:
    logger.warning(f"Failed to import yolo_inference: {e}")

# Import pricing
PricingEngine: Optional[Any] = None
QuoteRequest: Optional[Any] = None
try:
    from pricing.pricing_service import PricingEngine, QuoteRequest  # type: ignore
except ImportError as e:
    logger.warning(f"Failed to import pricing: {e}")

# Try to import pipeline (may not be available if dependencies missing)
pipeline_router: Optional[Any] = None
initialize_pipeline: Optional[Any] = None
PipelineConfig: Optional[Any] = None
PIPELINE_AVAILABLE = False

try:
    from pipeline_api import router as pipeline_router, initialize_pipeline  # type: ignore
    from pipeline_models import PipelineConfig  # type: ignore
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Pipeline not available: {e}")
    logger.warning("   Install easyocr to enable full pipeline: pip install easyocr")
    PIPELINE_AVAILABLE = False

# --- LIFESPAN ---
ml_models = {}
pricing_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pricing_engine
    logger.info("🟢 Server Starting...")
    
    # Initialize Pricing Engine
    try:
        if PricingEngine is not None:
            logger.info("🔄 Initializing Pricing Engine...")
            pricing_engine = PricingEngine()
            logger.info("✅ Pricing Engine initialized successfully")
        else:
            logger.warning("⚠️  Pricing Engine not available")
    except Exception as e:
        logger.error(f"❌ Pricing Engine Init Failed: {e}", exc_info=True)
        logger.error("   Quote generation endpoints may not work")
    
    if not MODEL_PATH:
        logger.error("❌ MODEL_PATH environment variable not set")
        logger.error("   Please set MODEL_PATH in your .env file to point to your YOLO model")
    elif not os.path.exists(MODEL_PATH):
        logger.error(f"❌ MODEL_PATH file not found: {MODEL_PATH}")
        logger.error("   Please ensure the model file exists at the specified path")
    else:
        try:
            if create_yolo_engine is not None:
                logger.info("🔄 Initializing YOLO inference engine...")
                ml_models["yolo_engine"] = create_yolo_engine(model_path=MODEL_PATH)
                logger.info("✅ Inference Engine Attached and Ready.")
            else:
                logger.error("❌ YOLO inference engine not available")
            
            # Initialize end-to-end pipeline if available
            if PIPELINE_AVAILABLE and PipelineConfig is not None and initialize_pipeline is not None:
                try:
                    logger.info("🔄 Initializing HVAC End-to-End Pipeline...")
                    pipeline_config = PipelineConfig(
                        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
                        max_processing_time_ms=float(os.getenv("MAX_PROCESSING_TIME", "25.0")),
                        enable_gpu=os.getenv("GPU_ENABLED", "true").lower() == "true"
                    )
                    initialize_pipeline(MODEL_PATH, pipeline_config)
                    logger.info("✅ HVAC Pipeline initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Pipeline Init Failed: {e}", exc_info=True)
                    logger.error("   The server will start but pipeline endpoints may not work")
        except Exception as e:
            logger.error(f"❌ Engine Init Failed: {e}", exc_info=True)
            logger.error("   The server will start but analysis endpoints will not work")
    
    yield
    
    # Cleanup on shutdown
    if ml_models.get("yolo_engine"):
        logger.info("🧹 Cleaning up inference engine...")
    ml_models.clear()
    logger.info("🔴 Server Shutting Down.")

app = FastAPI(title="HVAC AI", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Include pipeline router if available
if PIPELINE_AVAILABLE and pipeline_router is not None:
    app.include_router(pipeline_router)
    logger.info("✅ Pipeline router registered at /api/v1/pipeline")
else:
    logger.info("ℹ️  Pipeline router not available - install easyocr to enable")

# --- ENDPOINTS ---
@app.get("/health")
async def health_check():
    """Return service health and whether the inference engine is loaded.

    Frontend expects a `model_loaded` boolean to determine whether it can
    call analysis endpoints. This endpoint returns a small JSON payload
    with status, model_loaded, and optional message for troubleshooting.
    """
    engine = ml_models.get("yolo_engine")
    model_loaded = engine is not None
    response = {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": "2.1.0",
        "device": None,
    }
    
    if model_loaded:
        try:
            # Provide device information
            if hasattr(engine, 'device'):
                response["device"] = str(engine.device)
            
            # Provide a compact model identifier if available
            model_info = getattr(engine, 'model', None)
            if model_info:
                model_name = getattr(model_info, 'model', None) or getattr(model_info, 'weights', None)
                if model_name:
                    response["model"] = str(model_name)
                
                # Add class count
                if hasattr(model_info, 'names'):
                    response["num_classes"] = len(model_info.names)

                # Expose model type (obb / standard) if the engine provides it
                if hasattr(engine, 'model_type'):
                    response["model_type"] = getattr(engine, 'model_type')
        except Exception as e:
            # best-effort only - don't fail health check
            logger.debug(f"Could not extract model info: {e}")
    else:
        response["message"] = "Inference engine not attached; check MODEL_PATH and server logs"
        response["model_path"] = MODEL_PATH or "not set"

    return response

@app.post("/api/v1/analyze")
async def analyze_blueprint(
    image: UploadFile = File(...), 
    conf_threshold: float = Form(0.50)
):
    """Analyze HVAC blueprint image for component detection.
    
    Args:
        image: Uploaded image file
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        
    Returns:
        Analysis results with detected components
    """
    logger.info(f"📡 [API] Received Request: /analyze (file={image.filename}, conf={conf_threshold})")
    
    engine = ml_models.get("yolo_engine")
    if not engine:
        logger.error("Model not loaded - cannot process request")
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check server logs and MODEL_PATH configuration."
        )

    try:
        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read and validate image
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Convert to PIL and then numpy
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        image_np = np.array(pil_image)
        logger.info(f"📷 [API] Image loaded: {image_np.shape}")

        # Pass to engine (Logs happen inside engine)
        results = engine.predict(image_np, conf_threshold=conf_threshold)

        logger.info(f"📤 [API] Sending Response: Found {results['total_objects_found']} objects.")
        
        return {
            "status": "success",
            "analysis_id": uuid.uuid4().hex,
            "file_name": image.filename,
            **results
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except ValueError as e:
        logger.error(f"❌ [API] Validation Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ [API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze/stream")
async def analyze_blueprint_stream(
    request: Request, 
    image: UploadFile = File(...), 
    conf_threshold: float = Form(0.50)
):
    """Stream analysis progress as Server-Sent Events (SSE).

    The endpoint starts the inference in a background thread and yields JSON
    events as SSE `data:` payloads. Events have a `type` field: 'status',
    'progress', and 'result'.
    
    Args:
        request: FastAPI request object (for disconnect detection)
        image: Uploaded image file
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        
    Returns:
        Streaming response with SSE events
    """
    logger.info(f"📡 [API] Received Stream Request: /analyze/stream (file={image.filename}, conf={conf_threshold})")

    engine = ml_models.get("yolo_engine")
    if not engine:
        logger.error("Model not loaded - cannot process streaming request")
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check server logs and MODEL_PATH configuration."
        )

    try:
        # Validate and read image
        if not image.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
            
        image_np = np.array(pil_image)
        logger.info(f"📷 [API] Stream image loaded: {image_np.shape}")

        q: _queue.Queue = _queue.Queue()
        done_event = threading.Event()

        def progress_callback(msg: dict):
            try:
                q.put_nowait(msg)
            except Exception:
                logger.debug("Failed to enqueue progress message", exc_info=True)

        def worker():
            try:
                # The engine will call progress_callback for interim updates
                result = engine.predict(image_np, conf_threshold=conf_threshold, progress_callback=progress_callback)
                # Ensure final result is sent
                q.put({"type": "result", "result": result})
            except Exception as e:
                logger.exception("Error during inference (stream)")
                q.put({"type": "error", "message": str(e)})
            finally:
                done_event.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        async def event_generator():
            # Yield queued messages as SSE
            try:
                while not (done_event.is_set() and q.empty()):
                    try:
                        item = q.get(timeout=0.1)
                    except _queue.Empty:
                        # If client disconnected, stop
                        if await request.is_disconnected():
                            logger.info("Client disconnected, stopping stream")
                            break
                        continue

                    # Serialize item as JSON and yield as SSE data event
                    try:
                        payload = json.dumps(item, default=str)
                    except Exception:
                        payload = json.dumps({"type": "error", "message": "serialization_failure"})

                    yield f"data: {payload}\n\n"

                # Ensure we wait for worker to finish
                thread.join(timeout=0.5)
            except GeneratorExit:
                logger.info("Event generator closed by client")
            except Exception:
                logger.exception("Unexpected error in event generator")

        return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    except Exception as e:
        logger.error(f"❌ [API] Stream Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/count")
async def count_components(image: UploadFile = File(...), conf: float = Form(0.25)):
    return await analyze_blueprint(image, conf)

@app.post("/api/v1/annotations/save")
async def save_annotation_delta(request: Request):
    """Save annotation changes using delta (differential) format.
    
    Accepts a JSON payload with:
    - added: list of new annotations
    - modified: list of changed annotations  
    - deleted: list of annotation IDs to remove
    - verification_status: optional status flag
    
    This minimizes data transfer by only sending changes, not the full dataset.
    """
    logger.info(f"📡 [API] Received Delta Save Request")
    
    try:
        payload = await request.json()
        
        added = payload.get('added', [])
        modified = payload.get('modified', [])
        deleted = payload.get('deleted', [])
        verification_status = payload.get('verification_status', 'pending')
        
        logger.info(f"📊 [DELTA] Added: {len(added)}, Modified: {len(modified)}, Deleted: {len(deleted)}")
        
        # TODO: Implement actual persistence logic (database, file storage, etc.)
        # For now, just validate and acknowledge
        
        # Validate structure
        for ann in added + modified:
            if not all(k in ann for k in ['id', 'label', 'score', 'bbox']):
                raise ValueError(f"Invalid annotation structure: {ann}")
        
        # Simulate save operation
        save_id = uuid.uuid4().hex
        
        return {
            "status": "success",
            "save_id": save_id,
            "added_count": len(added),
            "modified_count": len(modified),
            "deleted_count": len(deleted),
            "verification_status": verification_status,
            "timestamp": time.time()
        }
        
    except ValueError as e:
        logger.error(f"❌ [API] Validation Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ [API] Save Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/quote/generate")
async def generate_quote(request: Request):
    """
    Generate a financial quote from HVAC analysis data.
    
    Request body should contain:
    - project_id: Project identifier
    - location: Location string (e.g., "Chicago, IL")
    - analysis_data: Object with total_objects and counts_by_category
    - settings: Optional pricing settings (margin_percent, tax_rate, labor_hourly_rate)
    
    Returns:
    - quote_id: Generated quote identifier
    - currency: Currency code (USD)
    - summary: Cost breakdown (materials, labor, total, final_price)
    - line_items: Detailed line items for each component category
    """
    logger.info(f"📡 [API] Received Quote Generation Request")
    
    if not pricing_engine or QuoteRequest is None:
        logger.error("Pricing engine or QuoteRequest model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Pricing engine not initialized. Check server logs."
        )
    
    try:
        # Parse request body as QuoteRequest model
        payload = await request.json()
        quote_request = QuoteRequest(**payload)
        
        logger.info(f"📊 [QUOTE] Project: {quote_request.project_id}, Location: {quote_request.location}")
        logger.info(f"📊 [QUOTE] Total objects: {quote_request.analysis_data.total_objects}")
        
        # Generate quote
        quote_response = pricing_engine.generate_quote(quote_request)
        
        logger.info(f"✅ [QUOTE] Generated: {quote_response.quote_id}, Final Price: ${quote_response.summary.final_price}")
        
        # Return as dict (Pydantic models serialize automatically)
        return quote_response.model_dump()
        
    except ValueError as e:
        logger.error(f"❌ [API] Validation Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ [API] Quote Generation Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quote generation failed: {str(e)}")
