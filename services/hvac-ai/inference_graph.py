"""
Inference Graph - Distributed AI Pipeline using Ray Serve
Implements official Ray Serve Application Builder pattern from docs.ray.io

This module defines:
1. ObjectDetectorDeployment - GPU-bound worker for object detection
2. TextExtractorDeployment - GPU-bound worker for text extraction  
3. APIServer - Ingress deployment that handles all HTTP requests via FastAPI
4. build() - Application builder function following official Ray Serve pattern

The application graph:
    APIServer (ingress, handles HTTP)
        ├── ObjectDetectorDeployment (worker)
        └── TextExtractorDeployment (worker)

This follows the official Ray Serve documentation:
https://docs.ray.io/en/latest/serve/getting_started.html
"""

import logging
import os
import sys
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse

try:
    import ray
    from ray import serve
    from ray.serve.handle import DeploymentHandle
except ImportError:
    raise RuntimeError("Ray Serve not installed. Install with: pip install 'ray[serve]'")

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

logger = logging.getLogger("HVAC-RayServe")
TEXT_RICH_CLASSES = {'id_letters', 'tag_number', 'text_label', 'label', 'text', 'tag'}

# === RAY SERVE DEPLOYMENTS ===

@serve.deployment(ray_actor_options={"num_gpus": 0.4}, max_ongoing_requests=5)
class ObjectDetectorDeployment:
    """
    Ray Serve deployment for object detection.
    Official pattern: @serve.deployment decorator on class
    """
    def __init__(self, model_path: str, force_cpu: bool = False):
        device = 'cpu' if force_cpu else 'cuda'
        self.detector = ObjectDetector(model_path=model_path, device=device)
        logger.info(f"[ObjectDetector] Ready on device: {device}")
    
    async def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image."""
        return await asyncio.to_thread(self.detector.detect, image)


@serve.deployment(ray_actor_options={"num_gpus": 0.3}, max_ongoing_requests=5)
class TextExtractorDeployment:
    """
    Ray Serve deployment for text extraction.
    Official pattern: @serve.deployment decorator on class
    """
    def __init__(self, use_gpu: bool = True):
        self.extractor = TextExtractor(lang='en', use_angle_cls=False, use_gpu=use_gpu)
        logger.info(f"[TextExtractor] Ready (GPU: {use_gpu})")
    
    async def extract(self, crop: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract text from image crop."""
        result = await asyncio.to_thread(self.extractor.extract_single_text, crop)
        if result:
            return {'text': result[0], 'confidence': result[1]}
        return None


@serve.deployment
class APIServer:
    """
    Ingress deployment that handles HTTP requests.
    
    Official Ray Serve pattern:
    - Class decorated with @serve.deployment
    - __init__ receives DeploymentHandle objects for dependent deployments
    - __call__ method receives Starlette Request objects
    - FastAPI app created internally (avoids serialization issues)
    """
    
    def __init__(
        self, 
        detector: DeploymentHandle,
        extractor: DeploymentHandle
    ):
        """
        Initialize API server.
        
        Args:
            detector: Handle to ObjectDetectorDeployment (from .bind())
            extractor: Handle to TextExtractorDeployment (from .bind())
        """
        self.detector = detector
        self.extractor = extractor
        
        # Initialize pricing engine
        self.pricing = None
        if PRICING_AVAILABLE:
            try:
                catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
                self.pricing = PricingEngine(catalog_path=catalog_path)
                logger.info("[APIServer] ✅ Pricing Engine initialized")
            except Exception as e:
                logger.warning(f"[APIServer] ⚠️  Pricing initialization failed: {e}")
        
        # Create FastAPI app INSIDE deployment (official pattern)
        self.app = FastAPI(title="HVAC AI Platform")
        
        # Register routes
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "pricing_enabled": self.pricing is not None,
                "detector_available": self.detector is not None,
                "extractor_available": self.extractor is not None
            }
        
        @self.app.post("/api/hvac/analyze")
        async def analyze_image(
            file: UploadFile = File(...),
            conf_threshold: float = Form(CONF_THRESHOLD),
            project_id: Optional[str] = Form(None),
            location: Optional[str] = Form(None)
        ):
            """Main analysis endpoint with integrated pricing."""
            try:
                # 1. Read Image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode image")
                
                # 2. Object Detection (official pattern: use handle.method.remote())
                detections = await self.detector.detect.remote(image)
                
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
                                # Official pattern: collect .remote() calls
                                text_tasks.append(self.extractor.extract.remote(crop))
                        except Exception as e:
                            logger.warning(f"Failed to extract crop: {e}")
                
                # Await text extraction results (official pattern: await gather of remote calls)
                if text_tasks:
                    text_results = await asyncio.gather(*text_tasks)
                    for idx, result in zip(text_indices, text_results):
                        if result:
                            enriched_detections[idx]['textContent'] = result['text']
                            enriched_detections[idx]['textConfidence'] = result['confidence']
                
                # 4. Pricing
                quote = None
                if self.pricing:
                    try:
                        counts = {}
                        for det in enriched_detections:
                            counts[det['label']] = counts.get(det['label'], 0) + 1
                        
                        analysis_data = AnalysisData(
                            total_objects=len(enriched_detections), 
                            counts_by_category=counts
                        )
                        quote_request = QuoteRequest(
                            project_id=project_id or f"AUTO-{int(time.time())}",
                            location=location or "default",
                            analysis_data=analysis_data
                        )
                        
                        quote_response = await asyncio.to_thread(
                            self.pricing.generate_quote, 
                            quote_request
                        )
                        quote = quote_response.model_dump()
                    except Exception as e:
                        logger.warning(f"Quote generation failed: {e}")
                
                # 5. Build Response
                return JSONResponse({
                    "detections": enriched_detections,
                    "quote": quote,
                    "image_shape": image.shape[:2]
                })
            
            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        logger.info("[APIServer] ✅ Initialized and ready to serve")
    
    async def __call__(self, request):
        """
        Official Ray Serve + FastAPI pattern.
        Return the FastAPI app directly so Ray Serve handles ASGI routing.
        """
        return self.app



# --- Application Builder Function (The Orchestrator) ---

def build_app():
    """
    Build the Ray Serve application graph using the official Application API.
    
    This function:
    1. Creates deployments for ObjectDetector and TextExtractor
    2. Creates the APIServer ingress deployment
    3. Returns a Serve Application configured for the ingress
    
    This is the official Ray Serve "Application Builder" pattern.
    
    Returns:
        A serve.Application configured for deployment
    """
    from ray import serve
    
    logger.info("[BUILD] Constructing Ray Serve application graph...")
    
    # Validate model path
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH environment variable not set")
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"[BUILD] Using model: {model_path}")
    
    # Create detector deployment
    logger.info("[BUILD] Creating ObjectDetectorDeployment...")
    detector = ObjectDetectorDeployment.bind(
        model_path=str(model_path),
        force_cpu=FORCE_CPU
    )
    
    # Create extractor deployment
    logger.info("[BUILD] Creating TextExtractorDeployment...")
    extractor = TextExtractorDeployment.bind(use_gpu=not FORCE_CPU)
    
    # Create API server ingress with references to the deployments
    logger.info("[BUILD] Creating APIServer ingress...")
    app = APIServer.bind(
        detector=detector,
        extractor=extractor
    )
    
    logger.info("[BUILD] ✅ Application graph constructed successfully")
    
    # Return the bound app deployment directly
    # Ray Serve will use it as the ingress
    return app


# --- Module-level entrypoint for Ray Serve CLI ---
# When running: serve run inference_graph:app
# This will be called automatically by Ray Serve
# (Only execute if all environment variables are set)
if __name__ != "__main__":
    # Check if environment is properly configured before building
    if os.getenv('MODEL_PATH') and os.getenv('RAY_USE_GPU') is not None:
        try:
            app = build_app()
        except Exception as e:
            # If build fails during import, just log it
            # start_ray_serve.py will catch this
            logger.error(f"Failed to build app at module level: {e}")
            app = None
    else:
        # Environment not configured, app will be built by start_ray_serve.py
        app = None
