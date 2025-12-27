"""
Inference Graph - Distributed AI Pipeline using Ray Serve
Implements a Directed Acyclic Graph (DAG) of independent deployments.

Architecture:
1. Ingress (API Gateway) - Receives HTTP POST, decodes images
2. ObjectDetector Node - Detects component geometry (OBB)
3. Geometry Engine - Maps coordinates, performs perspective transforms
4. TextExtractor Node - Reads text from straightened crops
5. Fusion Layer - Merges spatial and text data into unified JSON

Design Philosophy:
- Uses Ray Serve for distributed inference
- Fractional GPU allocation for efficient resource usage
- Asynchronous processing for high throughput
- Universal naming (ObjectDetector, TextExtractor, not YOLO/Paddle)
"""

import logging
import os
import sys
import time
import numpy as np
import cv2
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

# Ray Serve imports
try:
    import ray
    from ray import serve
    from ray.serve import Application
except ImportError:
    raise RuntimeError("Ray Serve not installed. Install with: pip install ray[serve]")

# Ensure the services directory is on sys.path for cross-service imports
SCRIPT_DIR = Path(__file__).resolve().parent
SERVICES_ROOT = SCRIPT_DIR.parent
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

# Local imports (direct from hvac-ai service)
from object_detector_service import ObjectDetector
from text_extractor_service import TextExtractor
from utils.geometry import GeometryUtils, OBB

logger = logging.getLogger(__name__)

# Cross-service imports (from sibling hvac-domain service)
try:
    # Import with fallback to handle different path configurations
    try:
        from pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData, QuoteSettings
    except ImportError:
        # Try alternative import path for hvac-domain
        import sys
        hvac_domain_path = SERVICES_ROOT / "hvac-domain"
        if str(hvac_domain_path) not in sys.path:
            sys.path.insert(0, str(hvac_domain_path))
        from pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData, QuoteSettings
except ImportError as e:
    logger.warning(f"Failed to import PricingEngine: {e}")
    logger.warning("Pricing functionality will be disabled")
    PricingEngine = None
    QuoteRequest = None
    AnalysisData = None
    QuoteSettings = None

# Configuration
TEXT_RICH_CLASSES = {'id_letters', 'tag_number', 'text_label', 'label', 'text', 'tag'}


@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.4,  # 40% VRAM allocation for ObjectDetector
    },
    max_concurrent_queries=10,
)
class ObjectDetectorDeployment:
    """
    Ray Serve deployment for object detection.
    Wraps the ObjectDetector service for distributed inference.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the object detector deployment.
        
        Args:
            model_path: Path to detection model weights
            conf_threshold: Confidence threshold for detections
        """
        logger.info("[AI-ENGINE] Initializing ObjectDetectorDeployment...")
        self.detector = ObjectDetector(
            model_path=model_path,
            device='cuda',  # Will use the fractional GPU allocation
            conf_threshold=conf_threshold
        )
        logger.info("[AI-ENGINE] ObjectDetectorDeployment ready")
    
    async def __call__(self, image_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform object detection on an image.
        
        Args:
            image_data: Dictionary containing 'image' (numpy array) and optional params
            
        Returns:
            List of detections
        """
        image = image_data['image']
        conf_threshold = image_data.get('conf_threshold', None)
        
        # Run detection (synchronous call wrapped in async context)
        detections = await asyncio.to_thread(
            self.detector.detect,
            image,
            conf_threshold
        )
        
        return detections


@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.3,  # 30% VRAM allocation for TextExtractor
    },
    max_concurrent_queries=10,
)
class TextExtractorDeployment:
    """
    Ray Serve deployment for text extraction.
    Wraps the TextExtractor service for distributed inference.
    """
    
    def __init__(self, lang: str = 'en', use_gpu: bool = True):
        """
        Initialize the text extractor deployment.
        
        Args:
            lang: Language code
            use_gpu: Use GPU acceleration
        """
        logger.info("[AI-ENGINE] Initializing TextExtractorDeployment...")
        self.extractor = TextExtractor(
            lang=lang,
            use_angle_cls=False,  # We handle rotation via geometry
            use_gpu=use_gpu
        )
        logger.info("[AI-ENGINE] TextExtractorDeployment ready")
    
    async def __call__(self, crop_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract text from a cropped region.
        
        Args:
            crop_data: Dictionary containing 'crop' (numpy array) and optional params
            
        Returns:
            Extracted text result or None
        """
        crop = crop_data['crop']
        conf_threshold = crop_data.get('conf_threshold', 0.5)
        
        # Run text extraction (synchronous call wrapped in async context)
        result = await asyncio.to_thread(
            self.extractor.extract_single_text,
            crop,
            conf_threshold
        )
        
        if result:
            text, confidence = result
            return {
                'text': text,
                'confidence': confidence
            }
        return None


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,  # CPU-only for ingress/fusion
    },
    max_concurrent_queries=20,
)
class InferenceGraphIngress:
    """
    Ingress node: API Gateway for the inference graph.
    Orchestrates the entire pipeline:
    1. Decode image
    2. Call ObjectDetector
    3. Filter for TEXT_RICH_CLASSES
    4. Extract and rectify crops using GeometryUtils
    5. Call TextExtractor for text-rich regions
    6. Merge results and return
    """
    
    def __init__(
        self,
        object_detector_handle,
        text_extractor_handle,
        enable_pricing: bool = True,
        default_location: str = "default"
    ):
        """
        Initialize the ingress node with handles to downstream services.
        
        Args:
            object_detector_handle: Ray Serve handle to ObjectDetectorDeployment
            text_extractor_handle: Ray Serve handle to TextExtractorDeployment
            enable_pricing: Enable pricing engine integration
            default_location: Default location for pricing calculations
        """
        logger.info("[AI-ENGINE] Initializing InferenceGraphIngress...")
        self.object_detector = object_detector_handle
        self.text_extractor = text_extractor_handle
        self.enable_pricing = enable_pricing and PricingEngine is not None
        self.default_location = default_location
        
        # Initialize pricing engine if available
        self.pricing_engine = None
        if self.enable_pricing:
            try:
                # Find catalog path relative to hvac-domain service
                catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
                if catalog_path.exists():
                    self.pricing_engine = PricingEngine(catalog_path=catalog_path)
                    logger.info("[AI-ENGINE] Pricing Engine initialized successfully")
                else:
                    logger.warning(f"[AI-ENGINE] Catalog not found at {catalog_path}, pricing disabled")
                    self.enable_pricing = False
            except Exception as e:
                logger.error(f"[AI-ENGINE] Failed to initialize pricing engine: {e}")
                self.enable_pricing = False
        else:
            logger.info("[AI-ENGINE] Pricing Engine disabled (not available or disabled by config)")
        
        logger.info("[AI-ENGINE] InferenceGraphIngress ready")
    
    async def __call__(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete inference request.
        
        Args:
            request_data: Dictionary containing:
                - image_base64: Base64-encoded image
                - or image: numpy array directly
                - conf_threshold: Optional confidence threshold
                - project_id: Optional project ID for quote generation
                - location: Optional location for regional pricing
                
        Returns:
            Complete analysis result with detections, text, and optional quote
        """
        try:
            # 1. Decode image
            if 'image_base64' in request_data:
                image = self._decode_image(request_data['image_base64'])
            elif 'image' in request_data:
                image = request_data['image']
            else:
                raise ValueError("No image provided")
            
            conf_threshold = request_data.get('conf_threshold', 0.5)
            
            # 2. Object Detection
            logger.info("Running object detection...")
            detections = await self.object_detector.remote({
                'image': image,
                'conf_threshold': conf_threshold
            })
            
            # 3. Filter for text-rich classes and process
            logger.info(f"Processing {len(detections)} detections...")
            enriched_detections = await self._process_detections(
                detections,
                image,
                conf_threshold
            )
            
            # 4. Generate pricing quote if enabled
            quote = None
            if self.enable_pricing and self.pricing_engine:
                try:
                    quote = await self._generate_quote(
                        enriched_detections,
                        request_data.get('project_id', f'AUTO-{int(time.time())}'),
                        request_data.get('location', self.default_location)
                    )
                except Exception as e:
                    logger.error(f"Quote generation failed: {e}", exc_info=True)
                    # Continue without quote rather than failing the entire request
            
            # 5. Build response
            response = {
                'status': 'success',
                'total_detections': len(enriched_detections),
                'detections': enriched_detections,
                'image_shape': image.shape[:2]
            }
            
            # Add quote to response if available
            if quote:
                response['quote'] = quote
            
            return response
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _process_detections(
        self,
        detections: List[Dict[str, Any]],
        image: np.ndarray,
        conf_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Process detections: extract text for text-rich classes.
        
        Args:
            detections: List of detections from ObjectDetector
            image: Original image
            conf_threshold: Confidence threshold for text extraction
            
        Returns:
            Enriched detections with text content
        """
        enriched = []
        
        # Process text-rich detections in parallel
        text_tasks = []
        text_indices = []
        
        for i, detection in enumerate(detections):
            label = detection['label'].lower()
            
            # Check if this is a text-rich class (exact word matching to avoid false positives)
            is_text_rich = label in TEXT_RICH_CLASSES or any(
                label == text_class or 
                label.startswith(text_class + '_') or 
                label.endswith('_' + text_class) or
                ('_' + text_class + '_') in label
                for text_class in TEXT_RICH_CLASSES
            )
            
            if is_text_rich:
                # Extract and rectify the region
                if 'obb' in detection:
                    # Use OBB geometry
                    obb = OBB(
                        x_center=detection['obb']['x_center'],
                        y_center=detection['obb']['y_center'],
                        width=detection['obb']['width'],
                        height=detection['obb']['height'],
                        rotation=detection['obb']['rotation']
                    )
                    
                    # Extract and preprocess crop
                    crop, metadata = GeometryUtils.extract_and_preprocess_obb(
                        image, obb, padding=5, preprocess=True
                    )
                    
                    if 'error' not in metadata:
                        # Schedule text extraction
                        text_indices.append(i)
                        text_tasks.append(
                            self.text_extractor.remote({
                                'crop': crop,
                                'conf_threshold': conf_threshold
                            })
                        )
                elif 'bbox' in detection:
                    # Use standard bbox
                    x1, y1, x2, y2 = detection['bbox']
                    crop = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Preprocess
                    crop_processed = GeometryUtils.preprocess_for_ocr(crop)
                    
                    # Schedule text extraction
                    text_indices.append(i)
                    text_tasks.append(
                        self.text_extractor.remote({
                            'crop': crop_processed,
                            'conf_threshold': conf_threshold
                        })
                    )
        
        # Wait for all text extractions
        if text_tasks:
            text_results = await asyncio.gather(*text_tasks)
            
            # Map results back to detections
            text_map = dict(zip(text_indices, text_results))
        else:
            text_map = {}
        
        # Build enriched detections
        for i, detection in enumerate(detections):
            enriched_det = detection.copy()
            
            # Add text content if available
            if i in text_map and text_map[i]:
                enriched_det['textContent'] = text_map[i]['text']
                enriched_det['textConfidence'] = text_map[i]['confidence']
            
            enriched.append(enriched_det)
        
        return enriched
    
    def _decode_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    async def _generate_quote(
        self,
        detections: List[Dict[str, Any]],
        project_id: str,
        location: str
    ) -> Dict[str, Any]:
        """
        Generate a pricing quote from detections.
        
        Args:
            detections: List of enriched detections
            project_id: Project identifier
            location: Location for regional pricing
            
        Returns:
            Quote dictionary or None if pricing fails
        """
        # Count detections by category
        counts_by_category = {}
        for detection in detections:
            label = detection.get('label', '').lower()
            # Normalize label for catalog lookup
            normalized_label = label.replace(' ', '_').replace('-', '_')
            counts_by_category[normalized_label] = counts_by_category.get(normalized_label, 0) + 1
        
        # Create analysis data
        analysis_data = AnalysisData(
            total_objects=len(detections),
            counts_by_category=counts_by_category
        )
        
        # Create quote request
        quote_request = QuoteRequest(
            project_id=project_id,
            location=location,
            analysis_data=analysis_data,
            settings=None  # Use defaults
        )
        
        # Generate quote (run in thread to avoid blocking async loop)
        quote_response = await asyncio.to_thread(
            self.pricing_engine.generate_quote,
            quote_request
        )
        
        # Convert Pydantic model to dict
        return quote_response.model_dump()  # or .dict() for older pydantic versions


def build_inference_graph(
    model_path: str,
    conf_threshold: float = 0.5,
    enable_pricing: bool = True,
    default_location: str = "default"
) -> Application:
    """
    Build and return the complete inference graph application.
    
    Args:
        model_path: Path to YOLO model weights
        conf_threshold: Default confidence threshold
        enable_pricing: Enable pricing engine integration
        default_location: Default location for pricing
        
    Returns:
        Ray Serve Application
    """
    # Deploy ObjectDetector
    object_detector = ObjectDetectorDeployment.bind(
        model_path=model_path,
        conf_threshold=conf_threshold
    )
    
    # Deploy TextExtractor
    text_extractor = TextExtractorDeployment.bind(
        lang='en',
        use_gpu=True
    )
    
    # Deploy Ingress with handles to other services
    ingress = InferenceGraphIngress.bind(
        object_detector_handle=object_detector,
        text_extractor_handle=text_extractor,
        enable_pricing=enable_pricing,
        default_location=default_location
    )
    
    return ingress


# Entrypoint for Ray Serve CLI
def entrypoint():
    """
    Entrypoint for Ray Serve: serve run inference_graph:entrypoint
    """
    # Get model path from environment or use relative default
    # Try to find model relative to the current directory structure
    default_model = os.path.join(os.getcwd(), 'ai_model', 'best.pt')
    if not os.path.exists(default_model):
        # Try parent directory
        default_model = os.path.join(os.path.dirname(os.getcwd()), 'ai_model', 'best.pt')
    
    # Prefer MODEL_PATH env var; fall back to legacy YOLO_MODEL_PATH for compatibility
    model_path = os.getenv('MODEL_PATH') or os.getenv('YOLO_MODEL_PATH') or default_model
    
    conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.5'))
    enable_pricing = os.getenv('ENABLE_PRICING', '1').lower() not in ('0', 'false', 'no')
    default_location = os.getenv('DEFAULT_LOCATION', 'default')
    
    logger.info(f"Building inference graph with model: {model_path}")
    logger.info(f"Pricing enabled: {enable_pricing}")
    return build_inference_graph(
        model_path,
        conf_threshold,
        enable_pricing=enable_pricing,
        default_location=default_location
    )
