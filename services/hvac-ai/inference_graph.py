"""
Inference Graph - Distributed AI Pipeline using the Application Builder Pattern.
Orchestrates Vision (YOLO), Language (OCR), and Business Logic (Pricing) into a unified API.
"""

# --- CRITICAL: Disable Phone-Home Checks BEFORE ANY ML imports ---
import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import logging
import sys
import numpy as np
import cv2
import asyncio
import time
from typing import Any, List, Dict, Optional, Union, cast
from pathlib import Path

# FastAPI & Starlette
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.types import Scope

# Module-level FastAPI app used with @serve.ingress(app)
app = FastAPI(title="HVAC AI Platform")

# Configure CORS at module level so the UI can connect immediately
origins = os.environ.get("FRONTEND_ORIGINS")
if origins:
    allow_origins = [o.strip() for o in origins.split(",") if o.strip()]
else:
    allow_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        # "memory": 1024 * 1024 * 1024 # Optional: Limit to 1GB RAM if needed
    },
    max_ongoing_requests=1 # Strict limit to prevent OOM
)
class TextExtractorDeployment:
    def __init__(self):
        # Force CPU usage for OCR by setting environment variables
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for this process
        os.environ['FLAGS_use_cuda'] = 'False'   # PaddlePaddle flag to disable CUDA

        # Reduce threads to prevent CPU starvation and memory issues
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        # Explicitly disable GPU for OCR to save VRAM for YOLO
        self.extractor = TextExtractor(lang='en')
        logger.info("[TextExtractor] Initialized on CPU (Single Threaded)")
    
    async def extract(self, crop: np.ndarray):
        # Run in thread
        result = await asyncio.to_thread(self.extractor.extract_single_text, crop)
        if result:
            return {'text': result[0], 'confidence': result[1]}
        return None

@serve.deployment(ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
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
        
        # NOTE: route handlers are defined as class methods below using the module-level `app`
    # APIServer no longer defines an explicit __call__ ASGI entrypoint.
    # When using @serve.ingress(app), Ray Serve provides the ASGI integration
    # and routes are registered via the module-level `app` decorators below.

    # --- HTTP Handlers (registered on module-level `app` via serve.ingress) ---
    @app.get("/health")
    async def health(self):
        try:
            # Basic health check
            health_status = {
                "status": "healthy",
                "pricing_enabled": self.pricing_engine is not None,
                "deployments": {}
            }

            # Test ObjectDetectorDeployment readiness
            try:
                # Create a small test image (1x1 pixel)
                test_image = np.zeros((1, 1, 3), dtype=np.uint8)
                await asyncio.wait_for(self.detector.detect.remote(test_image), timeout=5.0)
                health_status["deployments"]["object_detector"] = "ready"
            except Exception as e:
                logger.warning(f"ObjectDetectorDeployment not ready: {e}")
                health_status["deployments"]["object_detector"] = "initializing"
                health_status["status"] = "initializing"

            # Test TextExtractorDeployment readiness
            try:
                # Create a small test image for OCR
                test_image = np.ones((10, 10, 3), dtype=np.uint8) * 255  # White image
                await asyncio.wait_for(self.extractor.extract.remote(test_image), timeout=5.0)
                health_status["deployments"]["text_extractor"] = "ready"
            except Exception as e:
                logger.warning(f"TextExtractorDeployment not ready: {e}")
                health_status["deployments"]["text_extractor"] = "initializing"
                health_status["status"] = "initializing"

            # Return appropriate status code
            status_code = 200 if health_status["status"] == "healthy" else 503
            return JSONResponse(health_status, status_code=status_code)

        except Exception as e:
            logger.exception("Health check failed")
            return JSONResponse({"error": "health check failed", "status": "unhealthy"}, status_code=500)

    @app.post("/api/hvac/analyze")
    async def analyze(
        self,
        file: Optional[UploadFile] = File(None),
        image: Optional[UploadFile] = File(None),
        conf_threshold: float = Form(0.5),
        project_id: Optional[str] = Form(None),
        location: Optional[str] = Form(None),
    ):
        try:
            target_file = file or image
            if not target_file:
                logger.warning("No file received in 'file' or 'image' fields")
                return JSONResponse({"error": "field 'file' is required. Please upload with key 'file' or 'image'."}, status_code=400)

            contents = await target_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_np is None:
                return JSONResponse({"error": "Invalid image file (decoding failed)"}, status_code=400)

            # Object detection (run on detector deployment)
            detections = await self.detector.detect.remote(image_np)

            # OCR for text-rich classes
            ocr_tasks: List[Any] = []
            ocr_indices: List[int] = []
            for i, det in enumerate(detections):
                label = det.get("label", "")
                if label in TEXT_RICH_CLASSES:
                    crop = None
                    if "obb" in det:
                        try:
                            obb_data = OBB(**det["obb"])  # type: ignore
                            crop, _ = GeometryUtils.extract_and_preprocess_obb(image_np, obb_data)
                        except Exception:
                            pass

                    if crop is None and "bbox" in det:
                        try:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            crop = image_np[y1:y2, x1:x2]
                        except Exception:
                            crop = None

                    if crop is not None and getattr(crop, "size", 0) > 0:
                        ocr_indices.append(i)
                        ocr_tasks.append(self.extractor.extract.remote(crop))

            if ocr_tasks:
                ocr_results = await asyncio.gather(*ocr_tasks)
                for idx, res in zip(ocr_indices, ocr_results):
                    if res:
                        detections[idx]["textContent"] = res.get("text")
                        detections[idx]["textConfidence"] = res.get("confidence")

            # Compute simple counts for frontend
            counts: Dict[str, int] = {}
            for d in detections:
                label = d.get("label", "")
                if d.get("textContent"):
                    label = f"{label}_{d['textContent']}"
                counts[label] = counts.get(label, 0) + 1

            response_data = {
                "status": "success",
                "detections": make_serializable(detections),
                "quote": None,
                "image_shape": image_np.shape[:2],
                "pricing_available": bool(self.pricing_engine is not None),
                "counts": make_serializable(counts),
            }

            return JSONResponse(response_data)

        except Exception as e:
            logger.error(f"Analysis Error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/hvac/generate_quote")
    async def generate_quote(self, request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)

        counts = payload.get("counts") or {}
        project_id = payload.get("project_id")
        location = payload.get("location")

        if not self.pricing_engine:
            return JSONResponse({"error": "Pricing engine not available"}, status_code=503)

        try:
            analysis_data = AnalysisData(
                total_objects=sum(counts.values()) if isinstance(counts, dict) else 0,
                counts_by_category=counts,
            )

            quote_req = QuoteRequest(
                project_id=project_id or f"AUTO-{int(time.time())}",
                location=location or "Default",
                analysis_data=analysis_data,
            )

            quote_obj = await asyncio.to_thread(self.pricing_engine.generate_quote, quote_req)
            quote = quote_obj.model_dump() if hasattr(quote_obj, "model_dump") else getattr(quote_obj, "dict", lambda: quote_obj)()
            return JSONResponse({"quote": make_serializable(quote)})

        except Exception as e:
            logger.error(f"Quote generation failed: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/hvac/quote_from_counts")
    async def quote_from_counts(self, payload: Dict[str, Any] = Body(...)):
        if not self.pricing_engine:
            return JSONResponse({"error": "Pricing engine not available"}, status_code=503)

        try:
            counts = payload.get("counts") or {}
            project_id = payload.get("project_id")
            location = payload.get("location")

            analysis_data = AnalysisData(
                total_objects=sum(counts.values()) if isinstance(counts, dict) else 0,
                counts_by_category=counts,
            )

            quote_req = QuoteRequest(
                project_id=project_id or f"AUTO-{int(time.time())}",
                location=location or "Default",
                analysis_data=analysis_data,
            )

            quote_obj = await asyncio.to_thread(self.pricing_engine.generate_quote, quote_req)
            quote = quote_obj.model_dump() if hasattr(quote_obj, "model_dump") else getattr(quote_obj, "dict", lambda: quote_obj)()
            return JSONResponse({"quote": make_serializable(quote)})

        except Exception as e:
            logger.error(f"Quote from counts failed: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

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