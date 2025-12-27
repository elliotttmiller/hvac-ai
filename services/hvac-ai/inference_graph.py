"""
Inference Graph - Distributed AI Pipeline using Ray Serve + FastAPI
Implements a Directed Acyclic Graph (DAG) with explicit FastAPI routing.
This is the main entry point for the AI engine.
"""

import logging
import os
import sys
import numpy as np
import cv2
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import time
import threading

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from starlette.responses import JSONResponse

# Ray
try:
    import ray
    from ray import serve
except ImportError:
    raise RuntimeError("Ray Serve not installed. Install with: pip install ray[serve]")

# --- PATH & CONFIGURATION SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
SERVICES_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SERVICES_ROOT.parent

# Inject paths for imports
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

# Load environment variables
MODEL_PATH = os.getenv('MODEL_PATH')
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.5'))
FORCE_CPU = os.getenv('FORCE_CPU', '0') == '1'

# Local Imports
from object_detector_service import ObjectDetector
from text_extractor_service import TextExtractor
from utils.geometry import GeometryUtils, OBB

# Cross-Service Import (Pricing) - Handle hyphenated module name
try:
    import importlib
    pricing_module = importlib.import_module('hvac-domain.pricing.pricing_service')
    PricingEngine = pricing_module.PricingEngine
    QuoteRequest = pricing_module.QuoteRequest
    AnalysisData = pricing_module.AnalysisData
    PRICING_AVAILABLE = True
except (ImportError, AttributeError):
    PRICING_AVAILABLE = False

logger = logging.getLogger("RayServe-FastAPI")
TEXT_RICH_CLASSES = {'id_letters', 'tag_number', 'text_label', 'label', 'text', 'tag'}

# --- Ray Serve Deployments (The Workers) ---

@serve.deployment(ray_actor_options={"num_gpus": 0.4}, max_ongoing_requests=5)
class ObjectDetectorDeployment:
    """Distributed object detection service."""
    def __init__(self, model_path: str):
        device = 'cpu' if FORCE_CPU else 'cuda'
        self.detector = ObjectDetector(model_path=model_path, device=device)
        logger.info(f"[ObjectDetector] Ready on device: {device}")
    
    async def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.detector.detect, image)

@serve.deployment(ray_actor_options={"num_gpus": 0.3}, max_ongoing_requests=5)
class TextExtractorDeployment:
    """Distributed text extraction service."""
    def __init__(self, use_gpu: bool = True):
        self.extractor = TextExtractor(lang='en', use_angle_cls=False, use_gpu=use_gpu)
        logger.info(f"[TextExtractor] Ready (GPU: {use_gpu})")
    
    async def extract(self, crop: np.ndarray) -> Optional[Dict[str, Any]]:
        result = await asyncio.to_thread(self.extractor.extract_single_text, crop)
        if result:
            return {'text': result[0], 'confidence': result[1]}
        return None

# --- FastAPI App (The Router/Ingress) ---
app = FastAPI(title="HVAC AI Platform")

# Thread-safe lock to prevent race conditions during startup
startup_lock = threading.Lock()

@app.on_event("startup")
async def startup_event():
    """Initialize Ray deployments and services in a thread-safe manner."""
    with startup_lock:
        if hasattr(app.state, "services_initialized") and app.state.services_initialized:
            return # Already initialized

        logger.info("[STARTUP] Initializing FastAPI + Ray Serve integration...")
        
        # Bind deployments
        app.state.detector = ObjectDetectorDeployment.bind(model_path=MODEL_PATH)
        app.state.extractor = TextExtractorDeployment.bind(use_gpu=not FORCE_CPU)
        
        # Initialize pricing
        if PRICING_AVAILABLE:
            try:
                catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
                app.state.pricing = PricingEngine(catalog_path=catalog_path)
                logger.info("[STARTUP] ✅ Pricing Engine Initialized")
            except Exception as e:
                logger.error(f"[STARTUP] ❌ Failed to init Pricing: {e}")
                app.state.pricing = None
        
        app.state.services_initialized = True
        logger.info("[STARTUP] ✅ All services ready")

@app.on_event("shutdown")
def shutdown_event():
    """Gracefully shut down Ray connections."""
    logger.info("[SHUTDOWN] Cleaning up resources...")
    if ray.is_initialized():
        ray.shutdown()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pricing_enabled": hasattr(app.state, "pricing") and app.state.pricing is not None,
        "detector_available": hasattr(app.state, "detector"),
        "extractor_available": hasattr(app.state, "extractor")
    }

@app.post("/api/hvac/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(CONF_THRESHOLD),
    project_id: Optional[str] = Form(None),
    location: Optional[str] = Form(None)
):
    """Main analysis endpoint."""
    if not hasattr(app.state, "detector") or not app.state.detector:
        raise HTTPException(status_code=503, detail="AI services not initialized")
    
    try:
        # 1. Read Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        
        # 2. Object Detection
        detections = await app.state.detector.detect.remote(image)
        
        # 3. Text Extraction (Selective)
        enriched_detections = []
        text_tasks = []
        text_indices = []
        
        for i, det in enumerate(detections):
            enriched_det = det.copy()
            enriched_detections.append(enriched_det)
            
            if det.get('label', '') in TEXT_RICH_CLASSES and 'obb' in det:
                try:
                    obb = OBB(**det['obb'])
                    crop, _ = GeometryUtils.extract_and_preprocess_obb(image, obb)
                    if crop is not None:
                        text_indices.append(i)
                        text_tasks.append(app.state.extractor.extract.remote(crop))
                except Exception as e:
                    logger.warning(f"Failed to extract crop: {e}")
        
        # Await text extraction results
        if text_tasks:
            text_results = await asyncio.gather(*text_tasks)
            for idx, result in zip(text_indices, text_results):
                if result:
                    enriched_detections[idx]['textContent'] = result['text']
                    enriched_detections[idx]['textConfidence'] = result['confidence']
        
        # 4. Pricing
        quote = None
        if hasattr(app.state, "pricing") and app.state.pricing:
            try:
                counts = {}
                for det in enriched_detections:
                    counts[det['label']] = counts.get(det['label'], 0) + 1
                
                analysis_data = AnalysisData(total_objects=len(enriched_detections), counts_by_category=counts)
                quote_request = QuoteRequest(
                    project_id=project_id or f"AUTO-{int(time.time())}",
                    location=location or "default",
                    analysis_data=analysis_data
                )
                
                quote_response = await asyncio.to_thread(app.state.pricing.generate_quote, quote_request)
                quote = quote_response.model_dump()
            except Exception as e:
                logger.warning(f"Quote generation failed: {e}")
        
        # 5. Build Response
        response_data = {
            "detections": enriched_detections,
            "quote": quote,
            "image_shape": image.shape[:2]
        }
        
        return JSONResponse(response_data)
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Ray Serve Entrypoint ---
entrypoint = serve.ingress(app)