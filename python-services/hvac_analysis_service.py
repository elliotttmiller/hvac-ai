# python-services/hvac_analysis_service.py

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

# --- CONFIGURATION ---
# Force unbuffered output for Colab real-time logs
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("API_SERVER")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
env_file = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_file)

MODEL_PATH = os.getenv("MODEL_PATH")

import sys
sys.path.append(str(PROJECT_ROOT))
from core.ai.yolo_inference import create_yolo_engine

# --- LIFESPAN ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🟢 Server Starting...")
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        logger.error(f"❌ MODEL_PATH invalid: {MODEL_PATH}")
    else:
        try:
            ml_models["yolo_engine"] = create_yolo_engine(model_path=MODEL_PATH)
            logger.info("✅ Inference Engine Attached.")
        except Exception as e:
            logger.error(f"❌ Engine Init Failed: {e}")
    yield
    ml_models.clear()
    logger.info("🔴 Server Shutting Down.")

app = FastAPI(title="HVAC AI", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

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
    response = {"status": "healthy", "model_loaded": model_loaded}
    if model_loaded:
        try:
            # Provide a compact model identifier if available
            model_info = getattr(engine, 'model', None)
            model_name = getattr(model_info, 'model', None) or getattr(model_info, 'weights', None) or None
            if model_name:
                response["model"] = str(model_name)
        except Exception:
            # best-effort only
            pass
    else:
        response["message"] = "Inference engine not attached; check MODEL_PATH and server logs"

    return response

@app.post("/api/v1/analyze")
async def analyze_blueprint(
    image: UploadFile = File(...), 
    conf_threshold: float = Form(0.50) # Set Default to 0.50 here
):
    logger.info(f"📡 [API] Received Request: /analyze (file={image.filename})")
    
    engine = ml_models.get("yolo_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)

        # Pass to engine (Logs happen inside engine)
        results = engine.predict(image_np, conf_threshold=conf_threshold)

        logger.info(f"📤 [API] Sending Response: Found {results['total_objects_found']} objects.")
        
        return {
            "status": "success",
            "analysis_id": uuid.uuid4().hex,
            **results
        }

    except Exception as e:
        logger.error(f"❌ [API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/stream")
async def analyze_blueprint_stream(request: Request, image: UploadFile = File(...), conf_threshold: float = Form(0.50)):
    """Stream analysis progress as Server-Sent Events (SSE).

    The endpoint starts the inference in a background thread and yields JSON
    events as SSE `data:` payloads. Events have a `type` field: 'status',
    'progress', and 'result'.
    """
    logger.info(f"📡 [API] Received Stream Request: /analyze/stream (file={image.filename})")

    engine = ml_models.get("yolo_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)

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