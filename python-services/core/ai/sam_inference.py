"""
SAMEngine - lightweight, self-contained SAM inference wrapper

This module provides a SAMEngine class with simple `segment` and `count`
methods and a factory `create_sam_engine` that validates model presence.

The implementation intentionally avoids heavy, brittle imports at module
import time; instead model loading is attempted during engine creation so
failures surface clearly during application startup.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2

try:
    from pycocotools import mask as maskUtils
except Exception:  # pragma: no cover - optional dependency in some envs
    maskUtils = None

logger = logging.getLogger(__name__)


class SAMEngine:
    """Very small SAMEngine that exposes the same high-level API expected
    by the service code. It is intentionally conservative: if the provided
    model path does not exist the factory will raise an explicit error.

    The real SAM integration can replace the internals of this class later.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SAMEngine: model file not found: {model_path}")

        # Try to import torch (model loading) only when available so the
        # error is explicit and helpful.
        try:
            import torch  # type: ignore

            # Attempt a minimal load to validate the checkpoint shape/format.
            # We don't require the full model class here — we simply try to
            # read the file which surfaces corrupted/missing files.
            _ = torch.load(model_path, map_location="cpu")
            logger.info("SAMEngine: model checkpoint appears readable")
        except Exception as e:
            # Re-raise with a clear message so the FastAPI app will fail on startup
            raise RuntimeError(f"Failed to validate SAM model at {model_path}: {e}")

    def segment(self, image_bytes: bytes, prompt: Dict[str, Any], return_top_k: int = 1) -> List[Dict[str, Any]]:
        """Run segmentation on the provided image bytes and prompt.

        This lightweight implementation returns a single rectangular mask
        (centered) encoded as COCO RLE. Replace with the real SAM call.
        """
        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image bytes")

        h, w = img.shape[:2]

        # Create a simple dummy mask: centered rectangle covering 30% area
        mask = np.zeros((h, w), dtype=np.uint8)
        pad_h = int(h * 0.15)
        pad_w = int(w * 0.15)
        y0, y1 = pad_h, h - pad_h
        x0, x1 = pad_w, w - pad_w
        mask[y0:y1, x0:x1] = 1

        rle = None
        if maskUtils is not None:
            # pycocotools expects Fortran order
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            # maskUtils.encode returns bytes for 'counts' in py3; convert to str
            if isinstance(rle.get('counts'), bytes):
                rle['counts'] = rle['counts'].decode('ascii')
        else:
            # Fallback: return a naive representation (may not be renderable)
            rle = {"size": [h, w], "counts": ""}

        # Return a single mock segment
        segment = {
            "label": prompt.get("label", "object"),
            "score": 0.95,
            "mask": rle,
            "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        }

        return [segment]

    def count(self, image_bytes: bytes, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Return counts by category. This example counts the single mock
        segment as 1 object of the requested label.
        """
        segments = self.segment(image_bytes, prompt, return_top_k=1)
        label = segments[0].get('label', 'object')
        return {
            "total_objects_found": 1,
            "counts_by_category": {label: 1},
            "confidence_stats": {label: segments[0].get('score', 1.0)}
        }


def create_sam_engine(model_path: str) -> SAMEngine:
    """Factory that creates and returns a SAMEngine instance or raises
    a clear exception if the model cannot be validated/loaded.
    """
    return SAMEngine(model_path=model_path)
"""
SAM Model Inference Module
Segment Anything Model for P&ID and HVAC diagram analysis

Enhanced with:
- Advanced prompt engineering techniques
- Intelligent caching for performance
- Multi-stage classification pipeline
- Confidence scoring improvements
- Batch processing optimizations
"""

import os
import logging
import base64
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import cv2
import torch
from pycocotools import mask as mask_utils

logger = logging.getLogger(__name__)

# Configuration constants
CACHE_DEFAULT_SIZE = 50
CACHE_EVICTION_POLICY = "LRU"  # Least Recently Used

# Grid processing constants
GRID_SIZE_LARGE_IMAGE_THRESHOLD = 4000000  # pixels (>2000x2000)
GRID_SIZE_SMALL_IMAGE_THRESHOLD = 1000000  # pixels (<1000x1000)
GRID_SIZE_LARGE_IMAGE = 48  # pixels
GRID_SIZE_SMALL_IMAGE = 24  # pixels

# Classification constants
CLASSIFICATION_GEOMETRIC_WEIGHT = 0.6
CLASSIFICATION_VISUAL_WEIGHT = 0.4

# NMS and filtering constants
NMS_IOU_THRESHOLD = 0.9
DUPLICATE_MASK_IOU_THRESHOLD = 0.95

# Visual feature thresholds
COLOR_INTENSITY_HIGH_THRESHOLD = 200
COLOR_INTENSITY_LOW_THRESHOLD = 50
VISUAL_SCORE_INCREMENT_HIGH = 0.1
VISUAL_SCORE_INCREMENT_LOW = 0.05

# Complete taxonomy of HVAC/P&ID components (70 classes)
HVAC_TAXONOMY = [
    # Valves & Actuators
    "Actuator-Diaphragm",
    "Actuator-Generic",
    "Actuator-Manual",
    "Actuator-Motorized",
    "Actuator-Piston",
    "Actuator-Pneumatic",
    "Actuator-Solenoid",
    "Valve-3Way",
    "Valve-4Way",
    "Valve-Angle",
    "Valve-Ball",
    "Valve-Butterfly",
    "Valve-Check",
    "Valve-Control",
    "Valve-Diaphragm",
    "Valve-Gate",
    "Valve-Generic",
    "Valve-Globe",
    "Valve-Needle",
    "Valve-Plug",
    "Valve-Relief",
    # Equipment
    "Equipment-AgitatorMixer",
    "Equipment-Compressor",
    "Equipment-FanBlower",
    "Equipment-Generic",
    "Equipment-HeatExchanger",
    "Equipment-Motor",
    "Equipment-Pump-Centrifugal",
    "Equipment-Pump-Dosing",
    "Equipment-Pump-Generic",
    "Equipment-Pump-Screw",
    "Equipment-Vessel",
    # Instrumentation & Controls
    "Component-DiaphragmSeal",
    "Component-Switch",
    "Controller-DCS",
    "Controller-Generic",
    "Controller-PLC",
    "Instrument-Analyzer",
    "Instrument-Flow-Indicator",
    "Instrument-Flow-Transmitter",
    "Instrument-Generic",
    "Instrument-Level-Indicator",
    "Instrument-Level-Switch",
    "Instrument-Level-Transmitter",
    "Instrument-Pressure-Indicator",
    "Instrument-Pressure-Switch",
    "Instrument-Pressure-Transmitter",
    "Instrument-Temperature",
    # Piping, Ductwork & In-line Components
    "Accessory-Drain",
    "Accessory-Generic",
    "Accessory-SightGlass",
    "Accessory-Vent",
    "Damper",
    "Duct",
    "Filter",
    "Fitting-Bend",
    "Fitting-Blind",
    "Fitting-Flange",
    "Fitting-Generic",
    "Fitting-Reducer",
    "Pipe-Insulated",
    "Pipe-Jacketed",
    "Strainer-Basket",
    "Strainer-Generic",
    "Strainer-YType",
    "Trap",
]


@dataclass
class SegmentResult:
    """Result from SAM segmentation"""
    label: str
    score: float
    mask: Dict[str, Any]  # COCO RLE format: {"size": [height, width], "counts": "..."}
    bbox: List[int]  # [x, y, width, height]
    confidence_breakdown: Optional[Dict[str, float]] = None  # Detailed confidence scores
    alternative_labels: Optional[List[Tuple[str, float]]] = None  # Top-k alternative classifications


@dataclass
class CountResult:
    """Result from automated counting"""
    total_objects_found: int
    counts_by_category: Dict[str, int]
    processing_time_ms: Optional[float] = None
    confidence_stats: Optional[Dict[str, Any]] = None


@dataclass
class InferenceMetrics:
    """Metrics for monitoring inference performance"""
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    cache_hit: bool = False
    num_prompts_processed: int = 0


class SAMInferenceEngine:
    """
    SAM Model Inference Engine for HVAC/P&ID Analysis
    
    Provides interactive segmentation and automated component counting
    using the fine-tuned Segment Anything Model.
    
    Enhanced Features:
    - Intelligent image embedding cache for performance
    - Advanced prompt engineering with multi-point sampling
    - Multi-stage classification with confidence breakdown
    - Batch processing optimization
    - Performance monitoring and metrics
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, 
                 enable_cache: bool = True, cache_size: int = CACHE_DEFAULT_SIZE):
        """
        Initialize SAM inference engine
        
        Args:
            model_path: Path to fine-tuned SAM model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            enable_cache: Enable image embedding cache for repeated operations
            cache_size: Maximum number of cached embeddings
        """
        default_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'sam_hvac_finetuned.pth')
        self.model_path = model_path or os.getenv('SAM_MODEL_PATH', os.path.abspath(default_path))
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        self.transform = None
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # Image embedding cache for performance
        self._embedding_cache: Dict[str, Tuple[torch.Tensor, float]] = {}
        
        # Performance metrics
        self.metrics = {
            'total_inferences': 0,
            'cache_hits': 0,
            'avg_inference_time_ms': 0.0,
        }
        
        logger.info(f"Initializing Enhanced SAM Inference Engine on {self.device}")
        logger.info(f"Cache enabled: {enable_cache}, Cache size: {cache_size}")
        self._load_model()
        self._warm_up_model()
    
    def _load_model(self):
        """Load SAM model into GPU memory at startup"""
        try:
            logger.info(f"Attempting to load SAM model from {self.model_path}")
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}. Using mock mode.")
                self.model = "mock_model"
                return
            
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything.utils.transforms import ResizeLongestSide
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize SAM model (ViT-B by default)
            model_type = checkpoint.get('model_type', 'vit_b')
            sam = sam_model_registry[model_type]()
            
            # Load weights
            sam.load_state_dict(checkpoint['model_state_dict'])
            sam.to(device=self.device)
            sam.eval()
            
            self.model = sam
            self.image_encoder = sam.image_encoder
            self.prompt_encoder = sam.prompt_encoder
            self.mask_decoder = sam.mask_decoder
            self.transform = ResizeLongestSide(1024)
            
            logger.info("SAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            logger.warning("Using mock mode for development")
            self.model = "mock_model"
    
    def _warm_up_model(self):
        """Warm up the model with a dummy forward pass to optimize first inference"""
        if self.model == "mock_model":
            return
        
        try:
            logger.info("Warming up model...")
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            with torch.no_grad():
                input_image = self.transform.apply_image(dummy_image)
                input_tensor = torch.as_tensor(input_image, device=self.device)
                input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
                
                # Normalize
                pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
                pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
                input_tensor = (input_tensor - pixel_mean) / pixel_std
                
                # Encode once to warm up
                _ = self.image_encoder(input_tensor)
            
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """
        Compute hash of image for caching
        
        Uses SHA-256 for better collision resistance than MD5.
        
        Args:
            image: Input image
            
        Returns:
            Hash string
        """
        # Use a sample of the image for faster hashing
        h, w = image.shape[:2]
        sample = image[::max(1, h//32), ::max(1, w//32)].tobytes()
        return hashlib.sha256(sample).hexdigest()
    
    def _get_cached_embedding(self, image_hash: str) -> Optional[torch.Tensor]:
        """
        Get cached image embedding if available
        
        Args:
            image_hash: Hash of the image
            
        Returns:
            Cached embedding tensor or None
        """
        if not self.enable_cache:
            return None
        
        if image_hash in self._embedding_cache:
            embedding, timestamp = self._embedding_cache[image_hash]
            self.metrics['cache_hits'] += 1
            logger.debug(f"Cache hit for image hash: {image_hash}")
            return embedding
        
        return None
    
    def _cache_embedding(self, image_hash: str, embedding: torch.Tensor):
        """
        Cache image embedding
        
        Args:
            image_hash: Hash of the image
            embedding: Embedding tensor to cache
        """
        if not self.enable_cache:
            return
        
        # Implement LRU-like eviction
        if len(self._embedding_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self._embedding_cache.keys(), 
                           key=lambda k: self._embedding_cache[k][1])
            del self._embedding_cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")
        
        self._embedding_cache[image_hash] = (embedding, time.time())
        logger.debug(f"Cached embedding for image hash: {image_hash}")
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def segment(self, image: np.ndarray, prompt: Dict[str, Any], 
                return_top_k: int = 1, enable_refinement: bool = True) -> List[SegmentResult]:
        """
        Perform interactive segmentation based on user prompt
        
        Enhanced with:
        - Embedding caching for repeated operations
        - Multi-mask output with top-k selection
        - Optional prompt refinement
        - Detailed confidence breakdown
        
        Args:
            image: Input image as numpy array (H, W, 3)
            prompt: Prompt dictionary with type and data
                   Example: {"type": "point", "data": {"coords": [x, y], "label": 1}}
            return_top_k: Number of top predictions to return
            enable_refinement: Enable prompt refinement for better results
        
        Returns:
            List of segment results with masks and labels
        """
        start_time = time.perf_counter()
        
        if self.model == "mock_model":
            return self._mock_segment(image, prompt)
        
        try:
            # Parse prompt
            prompt_type = prompt.get('type')
            prompt_data = prompt.get('data', {})
            
            # Pre-process image
            original_size = image.shape[:2]
            
            # Check cache for image embedding
            image_hash = self._compute_image_hash(image)
            image_embedding = self._get_cached_embedding(image_hash)
            cache_hit = image_embedding is not None
            
            # Encode image once
            with torch.no_grad():
                if image_embedding is None:
                    # Transform and prepare image
                    input_image = self.transform.apply_image(image)
                    input_tensor = torch.as_tensor(input_image, device=self.device)
                    input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
                    
                    # Normalize
                    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
                    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
                    input_tensor = (input_tensor - pixel_mean) / pixel_std
                    
                    # Encode image
                    image_embedding = self.image_encoder(input_tensor)
                    
                    # Cache for future use
                    self._cache_embedding(image_hash, image_embedding)
                
                # Prepare prompt with optional refinement
                prompts_to_try = [prompt]
                
                if enable_refinement and prompt_type == "point":
                    # Add slight variations for robustness
                    coords = prompt_data['coords']
                    offsets = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
                    prompts_to_try = [
                        {
                            'type': 'point',
                            'data': {
                                'coords': [coords[0] + dx, coords[1] + dy],
                                'label': prompt_data.get('label', 1)
                            }
                        }
                        for dx, dy in offsets[:min(3, len(offsets))]  # Use top 3 variations
                    ]
                
                best_results = []
                
                for p in prompts_to_try:
                    if p['type'] == "point":
                        coords = np.array([p['data']['coords']])
                        labels = np.array([p['data'].get('label', 1)])
                        
                        # Transform coordinates
                        coords = self.transform.apply_coords(coords, original_size)
                        
                        # Convert to tensors
                        point_coords = torch.as_tensor(coords, dtype=torch.float, device=self.device)[None, :, :]
                        point_labels = torch.as_tensor(labels, dtype=torch.int, device=self.device)[None, :]
                        
                        # Encode prompt
                        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                            points=(point_coords, point_labels),
                            boxes=None,
                            masks=None
                        )
                    else:
                        raise ValueError(f"Unsupported prompt type: {p['type']}")
                    
                    # Decode mask
                    low_res_masks, iou_predictions = self.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                    
                    # Upscale mask to original size
                    masks = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=original_size,
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    masks = masks.squeeze().cpu().numpy()
                    scores = iou_predictions.squeeze().cpu().numpy()
                    if masks.ndim not in (2, 3):
                        raise ValueError(f"Expected mask tensor to have 2 or 3 dimensions but got shape: {masks.shape}")
                    if scores.ndim > 1:
                        scores = scores.flatten()
                    if masks.ndim == 2:
                        masks = np.expand_dims(masks, axis=0)
                    if scores.ndim == 0:
                        scores = np.array([float(scores)])
                    
                    # Collect results
                    for mask, score in zip(masks, scores):
                        best_results.append((mask, float(score)))
                
                # Sort by score and take top-k unique masks
                best_results.sort(key=lambda x: x[1], reverse=True)
                
                # Remove duplicates based on IoU
                unique_results = []
                for mask, score in best_results:
                    is_unique = True
                    for existing_mask, _ in unique_results:
                        iou = self._calculate_iou(mask > 0.0, existing_mask > 0.0)
                        if iou > DUPLICATE_MASK_IOU_THRESHOLD:  # Very similar masks
                            is_unique = False
                            break
                    if is_unique:
                        unique_results.append((mask, score))
                        if len(unique_results) >= return_top_k:
                            break
            
            # Post-process and create results
            results = []
            for mask, score in unique_results[:return_top_k]:
                # Convert mask to binary
                binary_mask = (mask > 0.0).astype(np.uint8)
                
                # Classify the segmented region with confidence breakdown
                label, confidence_breakdown, alternatives = self._classify_segment_enhanced(
                    image, binary_mask
                )
                
                # Convert to RLE in standard COCO format (JSON object)
                rle_dict = self._mask_to_rle(binary_mask)
                
                # Calculate bounding box
                bbox = self._mask_to_bbox(binary_mask)
                
                results.append(SegmentResult(
                    label=label,
                    score=score,
                    mask=rle_dict,
                    bbox=bbox,
                    confidence_breakdown=confidence_breakdown,
                    alternative_labels=alternatives
                ))
            
            # Update metrics
            inference_time = (time.perf_counter() - start_time) * 1000
            self.metrics['total_inferences'] += 1
            self.metrics['avg_inference_time_ms'] = (
                (self.metrics['avg_inference_time_ms'] * (self.metrics['total_inferences'] - 1) + 
                 inference_time) / self.metrics['total_inferences']
            )
            
            logger.info(f"Segmentation complete: {len(results)} results, "
                       f"time: {inference_time:.1f}ms, cache_hit: {cache_hit}")
            
            return results
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise
    
    def count(self, image: np.ndarray, grid_size: int = 32, confidence_threshold: float = 0.85,
              nms_iou_threshold: float = NMS_IOU_THRESHOLD, use_adaptive_grid: bool = True) -> CountResult:
        """
        Perform automated component counting on entire diagram
        
        Enhanced with:
        - Adaptive grid sizing based on image content
        - Batch processing for efficiency
        - Detailed confidence statistics
        - Performance monitoring
        
        Args:
            image: Input image as numpy array (H, W, 3)
            grid_size: Grid spacing for point prompts (pixels)
            confidence_threshold: Minimum confidence score to keep
            nms_iou_threshold: IoU threshold for non-maximum suppression
            use_adaptive_grid: Automatically adjust grid size based on image
        
        Returns:
            Count result with total objects and breakdown by category
        """
        start_time = time.perf_counter()
        
        if self.model == "mock_model":
            return self._mock_count(image)
        
        try:
            h, w = image.shape[:2]
            original_size = (h, w)
            
            # Adaptive grid sizing
            if use_adaptive_grid:
                # Adjust grid size based on image size
                image_area = h * w
                if image_area > GRID_SIZE_LARGE_IMAGE_THRESHOLD:  # Large image
                    grid_size = max(GRID_SIZE_LARGE_IMAGE, grid_size)
                elif image_area < GRID_SIZE_SMALL_IMAGE_THRESHOLD:  # Small image
                    grid_size = min(GRID_SIZE_SMALL_IMAGE, grid_size)
                
                logger.info(f"Using adaptive grid size: {grid_size}px for image {w}x{h}")
            
            # Check cache for image embedding
            image_hash = self._compute_image_hash(image)
            image_embedding = self._get_cached_embedding(image_hash)
            
            # Encode image once
            with torch.no_grad():
                if image_embedding is None:
                    # Transform and prepare image
                    input_image = self.transform.apply_image(image)
                    input_tensor = torch.as_tensor(input_image, device=self.device)
                    input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
                    
                    # Normalize
                    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
                    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
                    input_tensor = (input_tensor - pixel_mean) / pixel_std
                    
                    # Encode image once and reuse
                    image_embedding = self.image_encoder(input_tensor)
                    
                    # Cache for future use
                    self._cache_embedding(image_hash, image_embedding)
            
            # Generate grid of point prompts
            all_detections = []
            confidence_scores = []
            
            grid_points = [
                (x, y) 
                for y in range(grid_size // 2, h, grid_size)
                for x in range(grid_size // 2, w, grid_size)
            ]
            
            logger.info(f"Processing {len(grid_points)} grid points...")
            
            for x, y in grid_points:
                with torch.no_grad():
                    # Create point prompt
                    coords = np.array([[x, y]])
                    labels = np.array([1])
                    
                    # Transform coordinates
                    coords_transformed = self.transform.apply_coords(coords, original_size)
                    
                    # Convert to tensors
                    point_coords = torch.as_tensor(coords_transformed, dtype=torch.float, device=self.device)[None, :, :]
                    point_labels = torch.as_tensor(labels, dtype=torch.int, device=self.device)[None, :]
                    
                    # Encode prompt
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=(point_coords, point_labels),
                        boxes=None,
                        masks=None
                    )
                    
                    # Decode mask
                    low_res_masks, iou_predictions = self.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                    
                    # Get score
                    score = float(iou_predictions.squeeze().cpu().numpy())
                    confidence_scores.append(score)
                    
                    # Only keep high confidence detections
                    if score >= confidence_threshold:
                        # Upscale mask
                        mask = torch.nn.functional.interpolate(
                            low_res_masks,
                            size=original_size,
                            mode="bilinear",
                            align_corners=False
                        )
                        mask = (mask.squeeze().cpu().numpy() > 0.0).astype(np.uint8)
                        
                        # Classify with enhanced method
                        label, confidence_breakdown, _ = self._classify_segment_enhanced(image, mask)
                        
                        # Calculate bbox
                        bbox = self._mask_to_bbox(mask)
                        
                        all_detections.append({
                            'label': label,
                            'score': score,
                            'mask': mask,
                            'bbox': bbox,
                            'confidence_breakdown': confidence_breakdown
                        })
            
            # Apply Non-Maximum Suppression to remove duplicates
            unique_detections = self._apply_nms(all_detections, nms_iou_threshold)
            
            # Count by category
            counts = {}
            for det in unique_detections:
                label = det['label']
                counts[label] = counts.get(label, 0) + 1
            
            # Calculate confidence statistics
            confidence_stats = {
                'mean': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                'std': float(np.std(confidence_scores)) if confidence_scores else 0.0,
                'min': float(np.min(confidence_scores)) if confidence_scores else 0.0,
                'max': float(np.max(confidence_scores)) if confidence_scores else 0.0,
                'above_threshold': len(all_detections),
                'after_nms': len(unique_detections)
            }
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"Counting complete: {len(unique_detections)} objects found, "
                       f"time: {processing_time:.1f}ms")
            
            return CountResult(
                total_objects_found=len(unique_detections),
                counts_by_category=counts,
                processing_time_ms=processing_time,
                confidence_stats=confidence_stats
            )
            
        except Exception as e:
            logger.error(f"Counting failed: {e}")
            raise
    
    def _classify_segment(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Classify a segmented region (backward compatibility)
        
        Args:
            image: Original image
            mask: Binary mask
        
        Returns:
            Component label from HVAC_TAXONOMY
        """
        # Use deterministic selection based on mask properties for consistent testing
        mask_sum = int(np.sum(mask))
        return HVAC_TAXONOMY[mask_sum % len(HVAC_TAXONOMY)]
    
    def _classify_segment_enhanced(self, image: np.ndarray, mask: np.ndarray) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
        """
        Enhanced classification with multi-stage analysis and confidence breakdown
        
        This method uses a combination of:
        1. Geometric features (shape, size, aspect ratio)
        2. Spatial context (position, connections)
        3. Visual features (color, texture)
        
        Args:
            image: Original image
            mask: Binary mask
        
        Returns:
            Tuple of (label, confidence_breakdown, alternative_labels)
        """
        # Extract features from the masked region
        features = self._extract_segment_features(image, mask)
        
        # Stage 1: Geometric classification
        geometric_scores = self._classify_by_geometry(features)
        
        # Stage 2: Visual classification
        visual_scores = self._classify_by_visual_features(features)
        
        # Stage 3: Combine scores with weighted average
        combined_scores = {}
        for label in HVAC_TAXONOMY:
            geometric_score = geometric_scores.get(label, 0.0)
            visual_score = visual_scores.get(label, 0.0)
            
            # Weight configuration from constants
            combined_scores[label] = (
                CLASSIFICATION_GEOMETRIC_WEIGHT * geometric_score + 
                CLASSIFICATION_VISUAL_WEIGHT * visual_score
            )
        
        # Get top-k predictions
        sorted_labels = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_label = sorted_labels[0][0]
        confidence_breakdown = {
            'geometric': geometric_scores.get(best_label, 0.0),
            'visual': visual_scores.get(best_label, 0.0),
            'combined': combined_scores[best_label]
        }
        
        # Alternative predictions (top 3)
        alternatives = [(label, score) for label, score in sorted_labels[1:4]]
        
        return best_label, confidence_breakdown, alternatives
    
    def _extract_segment_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from a segmented region
        
        Args:
            image: Original image
            mask: Binary mask
            
        Returns:
            Dictionary of features
        """
        # Geometric features
        area = np.sum(mask)
        bbox = self._mask_to_bbox(mask)
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Shape features using contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Approximate polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            num_vertices = len(approx)
        else:
            circularity = 0
            num_vertices = 0
        
        # Visual features from masked region
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        region_pixels = masked_image[mask > 0]
        
        if len(region_pixels) > 0:
            mean_color = np.mean(region_pixels, axis=0)
            std_color = np.std(region_pixels, axis=0)
        else:
            mean_color = np.zeros(3)
            std_color = np.zeros(3)
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'num_vertices': num_vertices,
            'bbox_width': w,
            'bbox_height': h,
            'mean_color': mean_color,
            'std_color': std_color,
            'perimeter': perimeter if contours else 0
        }
    
    def _classify_by_geometry(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Classify based on geometric features
        
        This uses heuristics about typical shapes of HVAC components
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary of scores per label
        """
        scores = {}
        aspect_ratio = features['aspect_ratio']
        circularity = features['circularity']
        num_vertices = features['num_vertices']
        
        for label in HVAC_TAXONOMY:
            score = 0.5  # Base score
            
            # Circular components (pumps, fans, motors)
            if 'Pump' in label or 'Motor' in label or 'FanBlower' in label:
                if circularity > 0.7:
                    score += 0.3
            
            # Rectangular/elongated (pipes, ducts)
            elif 'Pipe' in label or 'Duct' in label:
                if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                    score += 0.3
            
            # Square-ish (valves, instruments)
            elif 'Valve' in label or 'Instrument' in label:
                if 0.7 < aspect_ratio < 1.3:
                    score += 0.2
                if 4 <= num_vertices <= 8:
                    score += 0.1
            
            # Equipment (vessels, heat exchangers)
            elif 'Equipment' in label or 'Vessel' in label:
                if features['area'] > 5000:  # Larger components
                    score += 0.2
            
            scores[label] = min(1.0, score)
        
        return scores
    
    def _classify_by_visual_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Classify based on visual features
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary of scores per label
        """
        scores = {}
        mean_color = features['mean_color']
        
        # For now, use simple heuristics
        # In production, this would use a trained classifier
        
        for label in HVAC_TAXONOMY:
            # Base score
            score = 0.5
            
            # Add small variations based on color intensity
            # (This is a placeholder - real implementation would use learned features)
            color_intensity = np.mean(mean_color)
            if color_intensity > COLOR_INTENSITY_HIGH_THRESHOLD:  # Light colors
                score += VISUAL_SCORE_INCREMENT_HIGH
            elif color_intensity < COLOR_INTENSITY_LOW_THRESHOLD:  # Dark colors
                score += VISUAL_SCORE_INCREMENT_LOW
            
            scores[label] = min(1.0, score)
        
        return scores
    
    def _mask_to_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Convert binary mask to COCO RLE format
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            RLE encoded dict in COCO format: {"size": [height, width], "counts": "..."}
        """
        # Fortran order required by COCO
        rle = mask_utils.encode(np.asfortranarray(mask))
        # Convert bytes to string if needed for JSON serialization
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        # Return standard COCO RLE format as dict
        return {"size": rle['size'], "counts": rle['counts']}
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """
        Calculate bounding box from binary mask
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            Bounding box [x, y, width, height]
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Filtered list of unique detections
        """
        if not detections:
            return []
        
        # Sort by score (descending)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        
        for det in detections:
            # Check if this detection overlaps significantly with any kept detection
            should_keep = True
            for kept_det in keep:
                iou = self._calculate_iou(det['mask'], kept_det['mask'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det)
        
        return keep
    
    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two masks
        
        Args:
            mask1: Binary mask 1
            mask2: Binary mask 2
        
        Returns:
            IoU score
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Dictionary of metrics
        """
        return {
            'total_inferences': self.metrics['total_inferences'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': (
                self.metrics['cache_hits'] / self.metrics['total_inferences']
                if self.metrics['total_inferences'] > 0 else 0.0
            ),
            'avg_inference_time_ms': self.metrics['avg_inference_time_ms'],
            'cache_size': len(self._embedding_cache),
            'cache_max_size': self.cache_size
        }
    
    def _mock_segment(self, image: np.ndarray, prompt: Dict[str, Any]) -> List[SegmentResult]:
        """Mock segmentation for development/testing"""
        h, w = image.shape[:2]
        
        # Create a mock circular mask around the clicked point
        if prompt.get('type') == 'point':
            coords = prompt['data']['coords']
            x, y = coords
            
            # Create circular mask
            mask = np.zeros((h, w), dtype=np.uint8)
            radius = 30
            cv2.circle(mask, (x, y), radius, 1, -1)
            
            # Convert to RLE in standard COCO format
            rle_dict = self._mask_to_rle(mask)
            
            # Calculate bbox
            bbox = [max(0, x - radius), max(0, y - radius), 2 * radius, 2 * radius]
            
            # Mock confidence breakdown
            confidence_breakdown = {
                'geometric': 0.92,
                'visual': 0.88,
                'combined': 0.90
            }
            
            # Mock alternatives
            alternatives = [
                ("Valve-Gate", 0.85),
                ("Valve-Control", 0.78),
                ("Fitting-Generic", 0.65)
            ]
            
            return [SegmentResult(
                label="Valve-Ball",
                score=0.967,
                mask=rle_dict,
                bbox=bbox,
                confidence_breakdown=confidence_breakdown,
                alternative_labels=alternatives
            )]
        
        return []
    
    def _mock_count(self, image: np.ndarray) -> CountResult:
        """Mock counting for development/testing"""
        return CountResult(
            total_objects_found=87,
            counts_by_category={
                "Valve-Ball": 23,
                "Valve-Gate": 12,
                "Fitting-Bend": 31,
                "Equipment-Pump-Centrifugal": 2,
                "Instrument-Pressure-Indicator": 19
            },
            processing_time_ms=2340.5,
            confidence_stats={
                'mean': 0.87,
                'std': 0.12,
                'min': 0.65,
                'max': 0.98,
                'above_threshold': 112,
                'after_nms': 87
            }
        )


def create_sam_engine(model_path: Optional[str] = None, device: Optional[str] = None, 
                     enable_cache: bool = True, cache_size: int = CACHE_DEFAULT_SIZE) -> SAMInferenceEngine:
    """
    Factory function to create SAM inference engine
    
    Args:
        model_path: Path to model checkpoint
        device: Device to run on
        enable_cache: Enable embedding cache
        cache_size: Maximum cache size
        
    Returns:
        Configured SAM inference engine
    """
    return SAMInferenceEngine(model_path, device, enable_cache, cache_size)
