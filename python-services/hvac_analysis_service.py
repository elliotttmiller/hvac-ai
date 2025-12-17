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
import sys

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
    return {"status": "healthy", "model": "YOLO11-Seg"}

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

@app.post("/api/v1/count")
async def count_components(image: UploadFile = File(...), conf: float = Form(0.25)):
    return await analyze_blueprint(image, conf)