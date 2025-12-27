"""
Inference Graph - Distributed AI Pipeline using Ray Serve + FastAPI
Implements a Directed Acyclic Graph (DAG) with explicit FastAPI routing.

FastAPI routes are mapped directly, so frontend requests to /api/hvac/analyze
are properly caught and processed.
"""

import logging
import os
import sys
import numpy as np
import cv2
import base64
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import time

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from starlette.responses import JSONResponse

# Ray
try:
    import ray
    from ray import serve
    from ray.serve import Application
except ImportError:
    raise RuntimeError("Ray Serve not installed. Install with: pip install ray[serve]")

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
SERVICES_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SERVICES_ROOT.parent

# Add paths for imports
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# --- ENVIRONMENT CONFIGURATION ---
def load_env_file(env_file_path):
    """Load environment variables from .env.local file."""
    if not env_file_path.exists():
        return
    
    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        logger = logging.getLogger("RayServe-FastAPI")
        logger.warning(f"Failed to load .env.local: {e}")

# Load .env.local BEFORE importing services
load_env_file(REPO_ROOT / ".env.local")

# Extract configuration from environment
SKIP_MODEL = os.environ.get('SKIP_MODEL', '0') == '1'
FORCE_CPU = os.environ.get('FORCE_CPU', '0') == '1'
MODEL_PATH = os.environ.get('MODEL_PATH')
CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', '0.5'))

logger = logging.getLogger("RayServe-FastAPI")
logger.info("=" * 60)
logger.info("[CONFIG] Inference Graph Configuration:")
logger.info(f"  SKIP_MODEL: {SKIP_MODEL}")
logger.info(f"  FORCE_CPU: {FORCE_CPU}")
logger.info(f"  MODEL_PATH: {MODEL_PATH or '(not set)'}")
logger.info(f"  CONF_THRESHOLD: {CONF_THRESHOLD}")
logger.info("=" * 60)

# Local Imports
from object_detector_service import ObjectDetector
from text_extractor_service import TextExtractor
from utils.geometry import GeometryUtils, OBB

# Cross-Service Import (Pricing)
try:
    from pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData
    PRICING_AVAILABLE = True
except ImportError:
    PRICING_AVAILABLE = False

TEXT_RICH_CLASSES = {'id_letters', 'tag_number', 'text_label', 'label', 'text', 'tag'}

# --- Ray Serve Deployments (The Workers) ---

@serve.deployment(ray_actor_options={"num_gpus": 0.4}, max_ongoing_requests=10)
class ObjectDetectorDeployment:
    """Distributed object detection service."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        logger.info("[ObjectDetector] Initializing...")
        
        # Determine device based on environment variables
        device = 'cpu' if FORCE_CPU else 'cuda'
        logger.info(f"[ObjectDetector] Using device: {device}")
        
        self.detector = ObjectDetector(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold
        )
        logger.info("[ObjectDetector] Ready")
    
    async def __call__(self, image_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        image = image_data['image']
        conf_threshold = image_data.get('conf_threshold', None)
        detections = await asyncio.to_thread(
            self.detector.detect,
            image,
            conf_threshold
        )
        return detections


@serve.deployment(ray_actor_options={"num_gpus": 0.3}, max_ongoing_requests=10)
class TextExtractorDeployment:
    """Distributed text extraction service."""
    
    def __init__(self, use_gpu: bool = True):
        logger.info("[TextExtractor] Initializing...")
        self.extractor = TextExtractor(lang='en', use_angle_cls=False, use_gpu=use_gpu)
        logger.info("[TextExtractor] Ready")
    
    async def __call__(self, crop_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        crop = crop_data['crop']
        conf_threshold = crop_data.get('conf_threshold', 0.5)
        
        result = await asyncio.to_thread(
            self.extractor.extract_single_text,
            crop,
            conf_threshold
        )
        
        if result:
            text, confidence = result
            return {'text': text, 'confidence': confidence}
        return None


# --- FastAPI App (The Router/Ingress) ---

app = FastAPI(title="HVAC Cortex AI Platform")

# Global handles (set by startup event)
detector_handle: Optional[Any] = None
extractor_handle: Optional[Any] = None
pricing_engine: Optional[Any] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Ray deployments and services."""
    global detector_handle, extractor_handle, pricing_engine
    
    logger.info("[STARTUP] Initializing FastAPI + Ray Serve integration...")
    
    # Get configuration from environment
    model_path = os.getenv("MODEL_PATH", str(REPO_ROOT / "ai_model" / "models" / "hvac_obb_l_20251224_214011" / "weights" / "best.pt"))
    conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.5"))
    enable_pricing = os.getenv("ENABLE_PRICING", "1").lower() not in ('0', 'false')
    use_gpu = not FORCE_CPU
    
    logger.info(f"[STARTUP] Model Path: {model_path}")
    logger.info(f"[STARTUP] Pricing Enabled: {enable_pricing}")
    logger.info(f"[STARTUP] GPU Enabled: {use_gpu}")
    logger.info(f"[STARTUP] FORCE_CPU: {FORCE_CPU}")
    
    # Bind deployments
    detector_handle = ObjectDetectorDeployment.bind(
        model_path=model_path,
        conf_threshold=conf_threshold
    )
    extractor_handle = TextExtractorDeployment.bind(use_gpu=use_gpu)
    
    # Initialize pricing if available
    if enable_pricing and PRICING_AVAILABLE:
        try:
            catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
            pricing_engine = PricingEngine(catalog_path=catalog_path)
            logger.info("[STARTUP] ✅ Pricing Engine Initialized")
        except Exception as e:
            logger.error(f"[STARTUP] ❌ Failed to init Pricing: {e}")
            pricing_engine = None
    
    logger.info("[STARTUP] ✅ All services ready")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pricing_enabled": pricing_engine is not None,
        "detections_available": detector_handle is not None,
        "text_extraction_available": extractor_handle is not None
    }


@app.post("/api/hvac/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5),
    project_id: Optional[str] = Form(None),
    location: Optional[str] = Form(None)
):
    """
    Main analysis endpoint.
    
    Accepts an image file and returns:
    - Detected objects with locations
    - Extracted text from text-rich regions
    - Optional pricing quote
    """
    
    if not detector_handle or not extractor_handle:
        raise HTTPException(status_code=503, detail="AI services not initialized")
    
    try:
        # 1. READ AND DECODE IMAGE
        logger.info("📥 Reading image...")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        logger.info(f"📊 Image shape: {image.shape}")
        
        # 2. OBJECT DETECTION
        logger.info("🔍 Running object detection...")
        detections = await detector_handle.remote({
            'image': image,
            'conf_threshold': conf_threshold
        })
        logger.info(f"✅ Found {len(detections)} objects")
        
        # 3. TEXT EXTRACTION (SELECTIVE)
        logger.info("📝 Processing text-rich regions...")
        enriched_detections = []
        text_tasks = []
        text_indices = []
        
        for i, det in enumerate(detections):
            enriched_det = det.copy()
            enriched_detections.append(enriched_det)
            
            label = det.get('label', '').lower()
            is_text_rich = label in TEXT_RICH_CLASSES or any(
                label == tc or 
                label.startswith(tc + '_') or 
                label.endswith('_' + tc) or
                ('_' + tc + '_') in label
                for tc in TEXT_RICH_CLASSES
            )
            
            if is_text_rich and 'obb' in det:
                try:
                    # Extract region using OBB
                    obb = OBB(
                        x_center=det['obb']['x_center'],
                        y_center=det['obb']['y_center'],
                        width=det['obb']['width'],
                        height=det['obb']['height'],
                        rotation=det['obb']['rotation']
                    )
                    crop, metadata = GeometryUtils.extract_and_preprocess_obb(
                        image, obb, padding=5, preprocess=True
                    )
                    
                    if 'error' not in metadata and crop is not None:
                        text_indices.append(i)
                        text_tasks.append(
                            extractor_handle.remote({
                                'crop': crop,
                                'conf_threshold': conf_threshold
                            })
                        )
                except Exception as e:
                    logger.warning(f"Failed to extract crop: {e}")
        
        # Wait for text extraction
        if text_tasks:
            logger.info(f"⏳ Extracting text from {len(text_tasks)} regions...")
            text_results = await asyncio.gather(*text_tasks)
            for idx, result in zip(text_indices, text_results):
                if result:
                    enriched_detections[idx]['textContent'] = result['text']
                    enriched_detections[idx]['textConfidence'] = result['confidence']
        
        # 4. PRICING (OPTIONAL)
        quote = None
        if pricing_engine:
            try:
                logger.info("💰 Generating quote...")
                # Count by category
                counts = {}
                for det in enriched_detections:
                    label = det.get('label', '').lower().replace(' ', '_').replace('-', '_')
                    counts[label] = counts.get(label, 0) + 1
                
                # Create quote request
                analysis_data = AnalysisData(
                    total_objects=len(enriched_detections),
                    counts_by_category=counts
                )
                quote_request = QuoteRequest(
                    project_id=project_id or f"AUTO-{int(time.time())}",
                    location=location or "default",
                    analysis_data=analysis_data,
                    settings=None
                )
                
                # Generate quote
                quote_response = await asyncio.to_thread(
                    pricing_engine.generate_quote,
                    quote_request
                )
                
                # Convert to dict
                try:
                    quote = quote_response.model_dump()
                except AttributeError:
                    quote = quote_response.dict()
                logger.info("✅ Quote generated")
            except Exception as e:
                logger.warning(f"Quote generation failed: {e}")
        
        # 5. BUILD RESPONSE
        logger.info("✨ Building response...")
        response_data = {
            "status": "success",
            "detections": enriched_detections,
            "total_detections": len(enriched_detections),
            "image_shape": image.shape[:2],
            "quote": quote
        }
        
        logger.info("✅ Analysis complete")
        return JSONResponse(response_data)
    
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Ray Serve Entrypoint ---
# The FastAPI app is the entry point
# No wrapper needed - the app handles all ASGI requests
entrypoint = app
