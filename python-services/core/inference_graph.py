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

# Local imports
from core.services.object_detector import ObjectDetector
from core.services.text_extractor import TextExtractor
from core.utils.geometry import GeometryUtils, OBB

logger = logging.getLogger(__name__)

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
        logger.info("🚀 Initializing ObjectDetectorDeployment...")
        self.detector = ObjectDetector(
            model_path=model_path,
            device='cuda',  # Will use the fractional GPU allocation
            conf_threshold=conf_threshold
        )
        logger.info("✅ ObjectDetectorDeployment ready")
    
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
        logger.info("🚀 Initializing TextExtractorDeployment...")
        self.extractor = TextExtractor(
            lang=lang,
            use_angle_cls=False,  # We handle rotation via geometry
            use_gpu=use_gpu
        )
        logger.info("✅ TextExtractorDeployment ready")
    
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
        text_extractor_handle
    ):
        """
        Initialize the ingress node with handles to downstream services.
        
        Args:
            object_detector_handle: Ray Serve handle to ObjectDetectorDeployment
            text_extractor_handle: Ray Serve handle to TextExtractorDeployment
        """
        logger.info("🚀 Initializing InferenceGraphIngress...")
        self.object_detector = object_detector_handle
        self.text_extractor = text_extractor_handle
        logger.info("✅ InferenceGraphIngress ready")
    
    async def __call__(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete inference request.
        
        Args:
            request_data: Dictionary containing:
                - image_base64: Base64-encoded image
                - or image: numpy array directly
                - conf_threshold: Optional confidence threshold
                
        Returns:
            Complete analysis result with detections and text
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
            
            # 4. Build response
            response = {
                'status': 'success',
                'total_detections': len(enriched_detections),
                'detections': enriched_detections,
                'image_shape': image.shape[:2]
            }
            
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
            
            # Check if this is a text-rich class
            if any(text_class in label for text_class in TEXT_RICH_CLASSES):
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


def build_inference_graph(
    model_path: str,
    conf_threshold: float = 0.5
) -> Application:
    """
    Build and return the complete inference graph application.
    
    Args:
        model_path: Path to YOLO model weights
        conf_threshold: Default confidence threshold
        
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
        text_extractor_handle=text_extractor
    )
    
    return ingress


# Entrypoint for Ray Serve CLI
def entrypoint():
    """
    Entrypoint for Ray Serve: serve run core.inference_graph:entrypoint
    """
    # Get model path from environment or use default
    model_path = os.getenv(
        'YOLO_MODEL_PATH',
        '/home/runner/work/hvac-ai/hvac-ai/ai_model/best.pt'
    )
    
    conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.5'))
    
    logger.info(f"Building inference graph with model: {model_path}")
    return build_inference_graph(model_path, conf_threshold)
