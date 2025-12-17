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
import io
import time

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
env_file = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_file)

MODEL_PATH = os.getenv("MODEL_PATH")

# Import the new YOLO Engine
import sys
sys.path.append(str(PROJECT_ROOT))
# Assuming you saved the previous code block as core/ai/yolo_inference.py
from core.ai.yolo_inference import create_yolo_engine

# --- 2. LIFESPAN (Load Model) ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        logger.error(f"❌ MODEL_PATH not valid: {MODEL_PATH}")
    else:
        try:
            # Initialize YOLO instead of SAM
            ml_models["yolo_engine"] = create_yolo_engine(model_path=MODEL_PATH)
            logger.info("✅ YOLO11 Engine initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO: {e}")
    yield
    ml_models.clear()

app = FastAPI(title="HVAC AI YOLO Service", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 3. ENDPOINTS ---

@app.get("/health")
async def health_check():
    engine = ml_models.get("yolo_engine")
    return {
        "status": "healthy" if engine else "degraded",
        "model_loaded": engine is not None,
        "model_type": "YOLO11-Seg"
    }

@app.post("/api/v1/analyze")
async def analyze_blueprint(
    image: UploadFile = File(...), 
    conf_threshold: float = Form(0.25)
):
    """
    Main inference endpoint. Detects, Segments, and Counts all symbols.
    """
    engine = ml_models.get("yolo_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read Image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)

        # Run Inference
        results = engine.predict(image_np, conf_threshold=conf_threshold)

        # Structure response
        return {
            "status": "success",
            "analysis_id": uuid.uuid4().hex,
            **results # Unpacks total_objects, counts, segments, timing
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Legacy support for /count endpoint (maps to same logic)
@app.post("/api/v1/count")
async def count_components(image: UploadFile = File(...), conf: float = Form(0.25)):
    return await analyze_blueprint(image, conf)