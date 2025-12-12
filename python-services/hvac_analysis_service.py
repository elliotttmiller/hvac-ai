# python-services/hvac_analysis_service.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
import traceback
import io
import time
import asyncio
import functools

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Find and load the .env file from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Go up one level from 'python-services'
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

MODEL_PATH = os.getenv("MODEL_PATH")

# --- 2. CRITICAL: Import the REAL SAM Engine ---
# Add the project root to the path so we can import from 'core'
import sys
sys.path.append(str(PROJECT_ROOT))
from core.ai.sam_inference import create_sam_engine

# --- 3. MODEL LOADING & LIFESPAN MANAGEMENT ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the SAM engine on startup
    logger.info("Initializing SAM Engine...")
    ml_models["sam_engine"] = create_sam_engine(model_path=MODEL_PATH)
    yield
    # Clean up on shutdown
    ml_models.clear()
    logger.info("Cleaned up models.")

# --- 4. INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="HVAC AI Analysis Service",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and any unhandled exceptions with a traceback.
    This helps when running the server in Colab so the cell output shows full context.
    """
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code} for {request.method} {request.url}")
        return response
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"Unhandled exception for {request.method} {request.url}: {exc}\n{tb}")
        raise

# --- 5. API ENDPOINTS ---
@app.post("/api/v1/segment")
async def segment_component(image: UploadFile = File(...), coords: str = Form(...)):
    sam_engine = ml_models.get("sam_engine")
    if not sam_engine:
        raise HTTPException(status_code=503, detail="SAM engine is not available.")
    
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)

        # Coords are sent as "x,y" string
        prompt = {"type": "point", "data": {"coords": [float(c) for c in coords.split(',')]}}

        # Time the segmentation so we can return processing_time in ms and seconds
        start_time = time.perf_counter()
        results = sam_engine.segment(image_np, prompt)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Normalize response shape for frontend compatibility
        total_components = len(results) if isinstance(results, list) else 0

        return {
            "status": "success",
            "analysis_id": uuid.uuid4().hex,
            "segments": results,
            "total_components": total_components,
            "processing_time_ms": processing_time_ms,
            "processing_time_seconds": processing_time_ms / 1000.0,
        }

    except Exception as e:
        logger.error(f"Segmentation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during segmentation.")

@app.post("/api/v1/count")
async def count_components(request: Request, image: UploadFile = File(...), grid_size: int = Form(32), min_score: float = Form(0.2), debug: bool = Form(False), timeout: int = Form(120)):
    sam_engine = ml_models.get("sam_engine")
    if not sam_engine:
        raise HTTPException(status_code=503, detail="SAM engine is not available.")
        
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)
        # Offload heavy synchronous work (CPU/GPU) to a thread executor so the
        # FastAPI event loop is not blocked. Use a configurable timeout to
        # avoid UI-hangs and return a 504 if the work takes too long.
        loop = asyncio.get_running_loop()
        func = functools.partial(sam_engine.count, image_np, grid_size, min_score, debug)
        logger.info(f"Offloading count to executor (grid_size={grid_size}, min_score={min_score}, timeout={timeout}s)")
        try:
            result = await asyncio.wait_for(loop.run_in_executor(None, func), timeout=float(timeout))
        except asyncio.TimeoutError:
            logger.error(f"Counting timed out after {timeout} seconds for request {request.url}")
            raise HTTPException(status_code=504, detail="Counting timed out")
        except Exception as e:
            # Bubble up other exceptions to be handled below by the existing except
            logger.error(f"Exception while running count in executor: {e}", exc_info=True)
            raise
        # Ensure we return both ms and seconds and a frontend-friendly total_components
        processing_time_ms = result.get("processing_time_ms") if isinstance(result, dict) else None
        processing_time_seconds = (processing_time_ms / 1000.0) if processing_time_ms is not None else None

        total_components = None
        if isinstance(result, dict):
            # preserve existing keys but add aliases for frontend
            # prefer explicit key if present, allow 0 as valid count
            total_components = result.get("total_objects_found") if result.get("total_objects_found") is not None else result.get("total_components")

        response = {"status": "success", "analysis_id": uuid.uuid4().hex, **(result if isinstance(result, dict) else {} )}
        if processing_time_ms is not None:
            response["processing_time_seconds"] = processing_time_seconds
        if total_components is not None:
            response["total_components"] = total_components
        # forward raw grid scores for debugging if present
        if isinstance(result, dict) and result.get("raw_grid_scores") is not None:
            response["raw_grid_scores"] = result.get("raw_grid_scores")

        return response

    except Exception as e:
        logger.error(f"Counting failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during counting.")

# Add aliases for frontend compatibility if needed
app.post("/api/analyze")(segment_component)
app.post("/api/count")(count_components)