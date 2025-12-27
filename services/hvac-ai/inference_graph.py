"""
Inference Graph - Distributed AI Pipeline using the Application Builder Pattern.
Orchestrates Vision (YOLO), Language (OCR), and Business Logic (Pricing) into a unified API.
"""

import logging
import os
import sys
import numpy as np
import cv2
import asyncio
import time
from typing import Any, List, Dict, Optional, Union, cast
from pathlib import Path

# FastAPI & Starlette
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from starlette.responses import JSONResponse
from starlette.types import Receive, Send, Scope

# Ray Serve
try:
    import ray
    from ray import serve
    from ray.serve.handle import DeploymentHandle
except ImportError:
    raise RuntimeError("Ray Serve not installed. Install with: pip install 'ray[serve]'")

# --- Imports ---
# Assumes the startup script has injected 'services/' and 'services/hvac-ai/' into sys.path
from object_detector_service import ObjectDetector
from text_extractor_service import TextExtractor
from utils.geometry import GeometryUtils, OBB

# --- Cross-Service Import (Pricing) ---
PricingEngine: Any = None
QuoteRequest: Any = None
AnalysisData: Any = None
PRICING_AVAILABLE = False

try:
    # 1. Try direct import (if hvac-domain is installed as a package)
    from hvac_domain.pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData  # type: ignore
    PRICING_AVAILABLE = True
except ImportError:
    try:
        # 2. Try adding the parent directory to sys.path explicitly
        # This handles the case where we are running from services/hvac-ai/
        current_dir = Path(__file__).resolve().parent
        services_dir = current_dir.parent
        if str(services_dir) not in sys.path:
            sys.path.insert(0, str(services_dir))

        # Prefer importing the underscored package name if present
        try:
            from hvac_domain.pricing import pricing_service as pricing_module  # type: ignore
            PricingEngine = pricing_module.PricingEngine
            QuoteRequest = pricing_module.QuoteRequest
            AnalysisData = pricing_module.AnalysisData
            PRICING_AVAILABLE = True
        except Exception:
            # Fall back to loading the module directly from the filesystem path
            try:
                import importlib.util

                hyphen_pkg_dir = services_dir / "hvac-domain" / "pricing"
                pricing_path = hyphen_pkg_dir / "pricing_service.py"
                if pricing_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "hvac_domain_pricing_pricing_service_local",
                        str(pricing_path),
                    )
                    pricing_module = importlib.util.module_from_spec(spec)  # type: ignore
                    assert spec and spec.loader
                    spec.loader.exec_module(pricing_module)  # type: ignore

                    # Patch potential name differences in the system module used by pricing
                    try:
                        system_path = services_dir / "hvac-domain" / "hvac_system_engine.py"
                        if system_path.exists():
                            sys_path_mod_name = "hvac_domain_hvac_system_engine_local"
                            spec_sys = importlib.util.spec_from_file_location(sys_path_mod_name, str(system_path))
                            if spec_sys and spec_sys.loader:
                                sys_mod = importlib.util.module_from_spec(spec_sys)
                                spec_sys.loader.exec_module(sys_mod)  # type: ignore
                                if not hasattr(sys_mod, "SystemRelationship") and hasattr(sys_mod, "ComponentRelationship"):
                                    setattr(sys_mod, "SystemRelationship", getattr(sys_mod, "ComponentRelationship"))
                    except Exception:
                        pass

                    PricingEngine = getattr(pricing_module, "PricingEngine", None)
                    QuoteRequest = getattr(pricing_module, "QuoteRequest", None)
                    AnalysisData = getattr(pricing_module, "AnalysisData", None)
                    PRICING_AVAILABLE = PricingEngine is not None
                else:
                    raise ImportError(f"pricing_service.py not found at {pricing_path}")
            except Exception as ie:
                logging.getLogger("RayServe").warning(f"Pricing Engine filesystem import failed: {ie}")
                PRICING_AVAILABLE = False
    except Exception as e:
        # Log the specific error to help debugging
        logging.getLogger("RayServe").warning(f"Pricing Engine import failed: {e}")
        PRICING_AVAILABLE = False

# Configuration
logger = logging.getLogger("RayServe-FastAPI")
TEXT_RICH_CLASSES = {'id_letters', 'tag_number', 'text_label', 'label', 'tag'}

# --- Helper: JSON Serialization ---
def make_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    Optimized to use NumPy base classes to satisfy Pylance/MyPy.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    # Use abstract base classes for robust type checking
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# --- Deployments ---

@serve.deployment(ray_actor_options={"num_gpus": 0.4})
class ObjectDetectorDeployment:
    def __init__(self, model_path: str):
        self.detector = ObjectDetector(model_path=model_path, device='cuda')
    
    async def detect(self, image: np.ndarray):
        # Run in thread to avoid blocking the asyncio loop
        return await asyncio.to_thread(self.detector.detect, image)

@serve.deployment(ray_actor_options={"num_gpus": 0.3})
class TextExtractorDeployment:
    def __init__(self):
        self.extractor = TextExtractor(lang='en')
    
    async def extract(self, crop: np.ndarray):
        # Run in thread
        result = await asyncio.to_thread(self.extractor.extract_single_text, crop)
        if result:
            return {'text': result[0], 'confidence': result[1]}
        return None

@serve.deployment(ray_actor_options={"num_cpus": 1})
class APIServer:
    def __init__(self, detector: DeploymentHandle, extractor: DeploymentHandle):
        self.detector = detector
        self.extractor = extractor
        self.pricing_engine: Any = None

        if PRICING_AVAILABLE:
            try:
                # Resolve catalog.json via absolute path based on repository layout.
                # This avoids problems when Ray worker's cwd differs from repo root.
                # File layout: <repo>/services/hvac-domain/pricing/catalog.json
                repo_root = Path(__file__).resolve().parents[2]
                catalog_path = repo_root / "services" / "hvac-domain" / "pricing" / "catalog.json"
                if not catalog_path.exists():
                    # Fallback: older layout or relative path
                    alt = Path(__file__).resolve().parent.parent / "hvac-domain" / "pricing" / "catalog.json"
                    if alt.exists():
                        catalog_path = alt

                self.pricing_engine = PricingEngine(catalog_path=catalog_path)
                logger.info(f"✅ Pricing Engine Initialized (catalog: {catalog_path})")
            except Exception as e:
                logger.error(f"Pricing Engine init failed: {e}")
        
        # --- Internal FastAPI App ---
        self.app = FastAPI(title="HVAC AI Platform")

        @self.app.get("/health")
        def health():
            try:
                return {
                    "status": "healthy",
                    "pricing_enabled": self.pricing_engine is not None
                }
            except Exception as e:
                # Log full exception in the worker logs for debugging and return minimal info
                logger.exception("Health check failed")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/hvac/analyze")
        async def analyze(
            file: UploadFile = File(...),
            conf_threshold: float = Form(0.5),
            project_id: Optional[str] = Form(None),
            location: Optional[str] = Form(None)
        ):
            try:
                # 1. Read & Decode Image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise HTTPException(400, "Invalid image file")

                # 2. Object Detection
                detections = await self.detector.detect.remote(image)

                # 3. Text Extraction (Parallelized)
                ocr_tasks = []
                ocr_indices = []

                for i, det in enumerate(detections):
                    label = det.get('label', '')
                    # Check if this class needs OCR
                    if label in TEXT_RICH_CLASSES:
                        crop = None
                        # Prefer OBB crop, fallback to BBox
                        if 'obb' in det:
                            try:
                                obb_data = OBB(**det['obb'])
                                crop, _ = GeometryUtils.extract_and_preprocess_obb(image, obb_data)
                            except Exception:
                                pass # Fallback to bbox if geometry fails
                        
                        if crop is None and 'bbox' in det:
                            x1, y1, x2, y2 = map(int, det['bbox'])
                            crop = image[y1:y2, x1:x2]

                        if crop is not None and crop.size > 0:
                            ocr_indices.append(i)
                            # Fire off async task
                            ocr_tasks.append(self.extractor.extract.remote(crop))

                # Wait for all OCR tasks to finish
                if ocr_tasks:
                    ocr_results = await asyncio.gather(*ocr_tasks)
                    for idx, res in zip(ocr_indices, ocr_results):
                        if res:
                            detections[idx]['textContent'] = res['text']
                            detections[idx]['textConfidence'] = res['confidence']

                # 4. Pricing Logic
                quote = None
                if self.pricing_engine:
                    try:
                        # Aggregate counts
                        counts = {}
                        for d in detections:
                            label = d['label']
                            # Use OCR text to refine label if possible (e.g. valve -> valve_2in)
                            if d.get('textContent'):
                                # Simple heuristic: append text to label for SKU lookup
                                # The pricing engine should handle fuzzy matching
                                label = f"{label}_{d['textContent']}"
                            counts[label] = counts.get(label, 0) + 1
                        
                        # Build Data Objects
                        analysis_data = AnalysisData(
                            total_objects=len(detections), 
                            counts_by_category=counts
                        )
                        quote_req = QuoteRequest(
                            project_id=project_id or f"AUTO-{int(time.time())}",
                            location=location or "Default",
                            analysis_data=analysis_data
                        )
                        
                        # Generate Quote
                        quote_obj = await asyncio.to_thread(
                            self.pricing_engine.generate_quote, 
                            quote_req
                        )
                        
                        # Serialize Pydantic model
                        quote = quote_obj.model_dump() if hasattr(quote_obj, 'model_dump') else quote_obj.dict()
                        
                    except Exception as e:
                        logger.error(f"Pricing failed: {e}")
                        # Don't fail the whole request, just return null quote

                # 5. Final Response
                response_data = {
                    "status": "success",
                    "detections": make_serializable(detections),
                    "quote": make_serializable(quote),
                    "image_shape": image.shape[:2]
                }

                return JSONResponse(response_data)

            except Exception as e:
                logger.error(f"Analysis Error: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

    # --- ASGI / Callable Interface ---
    async def __call__(self, scope: Optional[Scope] = None, receive: Optional[Receive] = None, send: Optional[Send] = None) -> Any:
        """Async ASGI-compatible entrypoint.

        - If called with no args, return the ASGI app (for frameworks that
          expect a callable app factory).
        - If called with a Starlette/FastAPI Request instance, extract its
          ASGI pieces and await the internal app.
        - Otherwise, assume (scope, receive, send) were passed and await the app.
        """
        # No args -> return the ASGI app callable itself
        if scope is None:
            return self.app

        # Local import to detect Request objects
        try:
            from fastapi import Request as FastAPIRequest  # type: ignore
        except Exception:
            FastAPIRequest = None  # type: ignore

        if FastAPIRequest is not None and isinstance(scope, FastAPIRequest):
            # scope is a Request instance. Use the Request's internal _send
            # (present on Request instances) rather than the outer `send`
            # parameter which may be None in some Serve wrappers.
            send_callable = getattr(scope, "_send", None)
            await self.app(scope.scope, scope.receive, cast(Send, send_callable))
            return None

        # Standard ASGI invocation
        await self.app(scope, cast(Receive, receive), cast(Send, send))
        return None

# --- Application Builder ---
def build_app():
    """
    Constructs the Ray Serve application graph.
    Called by start_ray_serve.py.
    """
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH environment variable not set")
        
    # Bind deployments (Type ignores for static analysis tools that don't see .bind)
    detector = ObjectDetectorDeployment.bind(model_path=model_path) # type: ignore
    extractor = TextExtractorDeployment.bind() # type: ignore
    
    # Bind Ingress
    app = APIServer.bind(detector, extractor) # type: ignore
    
    return app