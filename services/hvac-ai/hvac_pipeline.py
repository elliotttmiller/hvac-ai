"""
HVAC Drawing Analyzer - End-to-End Pipeline
Integrates YOLOv11-obb detection, EasyOCR text recognition, and HVAC semantic interpretation.
"""

import logging
import time
import re
import numpy as np
import cv2
import torch
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import uuid

try:
    import easyocr
except ImportError:
    easyocr = None

from .pipeline_models import (
    DetectionBox, DetectionResult, TextRecognitionResult,
    HVACInterpretation, HVACInterpretationResult, HVACResult,
    PipelineConfig, PipelineStage, PipelineError, ErrorSeverity,
    HVACEquipmentType
)
from .yolo_inference import YOLOInferenceEngine

logger = logging.getLogger(__name__)


class HVACDrawingAnalyzer:
    """
    Production-grade HVAC Drawing Analysis Pipeline.
    
    Integrates three stages:
    1. YOLOv11-obb for component and text region detection
    2. EasyOCR for text recognition within detected regions  
    3. HVAC-specific semantic interpretation and validation
    
    Thread-safe with parallel processing capabilities.
    """
    
    # HVAC pattern definitions
    HVAC_PATTERNS = {
        'vav': (r'VAV-?\d+', HVACEquipmentType.VAV),
        'ahu': (r'AHU-?\d+', HVACEquipmentType.AHU),
        'fcu': (r'FCU-?\d+', HVACEquipmentType.FCU),
        'pic': (r'PIC-?\d+', HVACEquipmentType.PIC),
        'te': (r'TE-?\d+', HVACEquipmentType.TE),
        'fit': (r'FIT-?\d+', HVACEquipmentType.FIT),
        'id_pattern': (r'[A-Z]{1,2}\d{1,2}', HVACEquipmentType.UNKNOWN),
        'id_pattern_dash': (r'[A-Z]{2}-\d+', HVACEquipmentType.UNKNOWN),
    }
    
    # Text classes that should trigger OCR
    TEXT_CLASSES = {'id_letters', 'tag_number', 'text_label', 'label', 'text', 'tag'}
    
    def __init__(
        self,
        yolo_model_path: str,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the HVAC Drawing Analyzer.
        
        Args:
            yolo_model_path: Path to YOLOv11-obb model file
            config: Pipeline configuration (uses defaults if None)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.config = config or PipelineConfig()
        self.yolo_model_path = yolo_model_path
        self.device = device or ('cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu')
        
        # Model components
        self.yolo_engine: Optional[YOLOInferenceEngine] = None
        self.ocr_reader = None
        
        # Thread safety
        self._init_lock = Lock()
        self._ocr_lock = Lock()
        
        # Thread pool for concurrent processing
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        
        # Performance tracking
        self._total_requests = 0
        self._total_processing_time = 0.0
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"✅ HVACDrawingAnalyzer initialized on {self.device}")
        logger.info(f"   Confidence threshold: {self.config.confidence_threshold}")
        logger.info(f"   Max processing time: {self.config.max_processing_time_ms}ms")
    
    def _initialize_models(self):
        """Initialize YOLO and EasyOCR models with health checks."""
        with self._init_lock:
            # Initialize YOLO
            try:
                logger.info("🔄 Initializing YOLOv11-obb detection model...")
                self.yolo_engine = YOLOInferenceEngine(
                    model_path=self.yolo_model_path,
                    device=self.device
                )
                logger.info("✅ YOLO model initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize YOLO model: {e}", exc_info=True)
                raise RuntimeError(f"YOLO initialization failed: {e}")
            
            # Initialize EasyOCR
            if easyocr is None:
                logger.warning("⚠️  EasyOCR not available - text recognition will be skipped")
                logger.warning("   Install with: pip install easyocr")
            else:
                try:
                    logger.info("🔄 Initializing EasyOCR text recognition...")
                    gpu_enabled = self.device == 'cuda'
                    self.ocr_reader = easyocr.Reader(
                        ['en'],
                        gpu=gpu_enabled,
                        verbose=False
                    )
                    logger.info(f"✅ EasyOCR initialized (GPU: {gpu_enabled})")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize EasyOCR: {e}", exc_info=True)
                    logger.warning("⚠️  Continuing without OCR capabilities")
                    self.ocr_reader = None
    
    def analyze_drawing(
        self,
        image_path: str,
        request_id: Optional[str] = None
    ) -> HVACResult:
        """
        Analyze an HVAC drawing end-to-end.
        
        Args:
            image_path: Path to the input image
            request_id: Optional unique request identifier
            
        Returns:
            HVACResult containing all pipeline results and metadata
        """
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        start_time = time.perf_counter()
        stage_timings = {}
        errors = []
        warnings = []
        
        logger.info(f"🚀 Starting analysis for request {request_id}")
        logger.info(f"   Image: {image_path}")
        
        # Initialize result
        result = HVACResult(
            request_id=request_id,
            stage=PipelineStage.DETECTION,
            total_processing_time_ms=0.0,
            stage_timings=stage_timings,
            errors=errors,
            warnings=warnings,
            image_path=image_path
        )
        
        try:
            # Load image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
            
            # Stage 1: Detection
            stage_start = time.perf_counter()
            detection_result = self._stage1_detection(image, request_id)
            stage_timings['detection'] = (time.perf_counter() - stage_start) * 1000
            result.detection_result = detection_result
            result.stage = PipelineStage.TEXT_RECOGNITION
            
            # Check timeout
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.config.max_processing_time_ms:
                warnings.append(f"Detection took {elapsed_ms:.1f}ms, exceeding timeout")
            
            # Stage 2: Text Recognition
            if detection_result.text_regions and self.ocr_reader is not None:
                stage_start = time.perf_counter()
                text_results = self._stage2_text_recognition(
                    image,
                    detection_result.text_regions,
                    request_id
                )
                stage_timings['text_recognition'] = (time.perf_counter() - stage_start) * 1000
                result.text_results = text_results
            else:
                if not detection_result.text_regions:
                    warnings.append("No text regions detected")
                if self.ocr_reader is None:
                    warnings.append("OCR not available - skipping text recognition")
                stage_timings['text_recognition'] = 0.0
            
            result.stage = PipelineStage.INTERPRETATION
            
            # Stage 3: Semantic Interpretation
            if result.text_results:
                stage_start = time.perf_counter()
                interpretation_result = self._stage3_interpretation(
                    result.text_results,
                    detection_result.detections,
                    image.shape[1],  # width
                    image.shape[0],  # height
                    request_id
                )
                stage_timings['interpretation'] = (time.perf_counter() - stage_start) * 1000
                result.interpretation_result = interpretation_result
            else:
                warnings.append("No text recognized - skipping interpretation")
                stage_timings['interpretation'] = 0.0
            
            # Success!
            result.stage = PipelineStage.COMPLETE
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed for {request_id}: {e}", exc_info=True)
            result.stage = PipelineStage.FAILED
            errors.append(PipelineError(
                stage=result.stage,
                severity=ErrorSeverity.CRITICAL,
                message=str(e),
                details={'traceback': str(e)}
            ))
        
        # Calculate total time
        result.total_processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Update statistics
        self._total_requests += 1
        self._total_processing_time += result.total_processing_time_ms
        
        logger.info(f"✅ Analysis complete for {request_id}")
        logger.info(f"   Total time: {result.total_processing_time_ms:.2f}ms")
        logger.info(f"   Stage timings: {stage_timings}")
        logger.info(f"   Success: {result.success}")
        
        return result
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _stage1_detection(
        self,
        image: np.ndarray,
        request_id: str
    ) -> DetectionResult:
        """
        Stage 1: Component & Text Region Detection using YOLOv11-obb.
        
        Args:
            image: Input image (H, W, 3)
            request_id: Request identifier for logging
            
        Returns:
            DetectionResult containing all detections and text regions
        """
        logger.info(f"[{request_id}] Stage 1: Running YOLO detection...")
        
        start_time = time.perf_counter()
        
        # Run YOLO inference
        yolo_results = self.yolo_engine.predict(
            image,
            conf_threshold=self.config.confidence_threshold
        )
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse YOLO results into DetectionBox objects
        all_detections = []
        text_regions = []
        
        for seg in yolo_results.get('segments', []):
            bbox_coords = seg['bbox']  # [x1, y1, x2, y2]
            
            detection = DetectionBox(
                x1=float(bbox_coords[0]),
                y1=float(bbox_coords[1]),
                x2=float(bbox_coords[2]),
                y2=float(bbox_coords[3]),
                confidence=float(seg['score']),
                class_id=seg.get('class_id', 0),
                class_name=seg['label']
            )
            
            all_detections.append(detection)
            
            # Check if this is a text class
            if seg['label'].lower() in self.TEXT_CLASSES:
                text_regions.append(detection)
        
        logger.info(f"[{request_id}] Stage 1 complete: {len(all_detections)} detections, "
                   f"{len(text_regions)} text regions in {processing_time_ms:.2f}ms")
        
        return DetectionResult(
            detections=all_detections,
            text_regions=text_regions,
            processing_time_ms=processing_time_ms,
            image_width=image.shape[1],
            image_height=image.shape[0],
            model_version="yolo11m-obb"
        )
    
    def _stage2_text_recognition(
        self,
        image: np.ndarray,
        text_regions: List[DetectionBox],
        request_id: str
    ) -> List[TextRecognitionResult]:
        """
        Stage 2: Text Recognition using EasyOCR with parallel processing.
        
        Args:
            image: Input image (H, W, 3)
            text_regions: List of detected text regions
            request_id: Request identifier for logging
            
        Returns:
            List of TextRecognitionResult objects
        """
        logger.info(f"[{request_id}] Stage 2: Running OCR on {len(text_regions)} text regions...")
        
        if not self.ocr_reader:
            logger.warning(f"[{request_id}] OCR reader not available")
            return []
        
        # Process text regions in parallel
        results = []
        futures = []
        
        for i, region in enumerate(text_regions):
            future = self._executor.submit(
                self._recognize_text_region,
                image,
                region,
                i,
                request_id
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result(timeout=5.0)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"[{request_id}] Text recognition failed: {e}")
        
        logger.info(f"[{request_id}] Stage 2 complete: {len(results)} texts recognized")
        
        return results
    
    def _recognize_text_region(
        self,
        image: np.ndarray,
        region: DetectionBox,
        region_idx: int,
        request_id: str
    ) -> Optional[TextRecognitionResult]:
        """
        Recognize text in a single region with adaptive padding.
        
        Args:
            image: Full input image
            region: Text region to process
            region_idx: Region index for logging
            request_id: Request identifier
            
        Returns:
            TextRecognitionResult or None if recognition failed
        """
        try:
            # Calculate adaptive padding based on region size
            region_width = region.width
            region_height = region.height
            avg_dimension = (region_width + region_height) / 2
            
            if avg_dimension < 20:
                padding = self.config.padding_max  # 10 pixels for very small regions
            elif avg_dimension < 50:
                padding = (self.config.padding_min + self.config.padding_max) // 2  # 7-8 pixels
            else:
                padding = self.config.padding_min  # 5 pixels for larger regions
            
            # Crop region with padding
            x1 = max(0, int(region.x1 - padding))
            y1 = max(0, int(region.y1 - padding))
            x2 = min(image.shape[1], int(region.x2 + padding))
            y2 = min(image.shape[0], int(region.y2 + padding))
            
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                logger.warning(f"[{request_id}] Region {region_idx} is empty after cropping")
                return None
            
            # Run OCR with HVAC-optimized parameters
            with self._ocr_lock:  # Thread-safe OCR calls
                ocr_results = self.ocr_reader.readtext(
                    cropped,
                    detail=1,
                    paragraph=False,
                    min_size=self.config.ocr_min_size,
                    text_threshold=self.config.ocr_text_threshold,
                    low_text=self.config.ocr_low_text,
                    canvas_size=self.config.ocr_canvas_size
                )
            
            # Extract best result
            if ocr_results:
                # EasyOCR returns list of ([bbox], text, confidence)
                best_result = max(ocr_results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = float(best_result[2])
                
                if text:
                    return TextRecognitionResult(
                        region=region,
                        text=text,
                        confidence=confidence,
                        preprocessing_metadata={
                            'padding_applied': padding,
                            'region_width': int(region_width),
                            'region_height': int(region_height),
                            'cropped_width': x2 - x1,
                            'cropped_height': y2 - y1
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"[{request_id}] Failed to recognize text in region {region_idx}: {e}")
            return None
    
    def _stage3_interpretation(
        self,
        text_results: List[TextRecognitionResult],
        all_detections: List[DetectionBox],
        image_width: int,
        image_height: int,
        request_id: str
    ) -> HVACInterpretationResult:
        """
        Stage 3: HVAC Semantic Interpretation with parallel processing.
        
        Args:
            text_results: Recognized text results
            all_detections: All detected components
            image_width: Image width for spatial calculations
            image_height: Image height for spatial calculations
            request_id: Request identifier
            
        Returns:
            HVACInterpretationResult with all interpretations
        """
        logger.info(f"[{request_id}] Stage 3: Interpreting {len(text_results)} text results...")
        
        start_time = time.perf_counter()
        
        interpretations = []
        validated = 0
        failed = 0
        
        # Process interpretations in parallel
        futures = []
        for text_result in text_results:
            future = self._executor.submit(
                self._interpret_text,
                text_result,
                all_detections,
                image_width,
                image_height
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                interpretation = future.result(timeout=1.0)
                if interpretation:
                    interpretations.append(interpretation)
                    if interpretation.equipment_type != HVACEquipmentType.UNKNOWN:
                        validated += 1
                    else:
                        failed += 1
            except Exception as e:
                logger.error(f"[{request_id}] Interpretation failed: {e}")
                failed += 1
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"[{request_id}] Stage 3 complete: {validated} validated, "
                   f"{failed} failed in {processing_time_ms:.2f}ms")
        
        return HVACInterpretationResult(
            interpretations=interpretations,
            processing_time_ms=processing_time_ms,
            total_validated=validated,
            total_failed=failed
        )
    
    def _interpret_text(
        self,
        text_result: TextRecognitionResult,
        all_detections: List[DetectionBox],
        image_width: int,
        image_height: int
    ) -> Optional[HVACInterpretation]:
        """
        Interpret a single text result using HVAC patterns.
        
        Args:
            text_result: Text recognition result
            all_detections: All detected components for spatial association
            image_width: Image width
            image_height: Image height
            
        Returns:
            HVACInterpretation or None if interpretation failed
        """
        text = text_result.text.upper().strip()
        
        # Try to match against HVAC patterns
        equipment_type = HVACEquipmentType.UNKNOWN
        pattern_matched = "none"
        zone_number = None
        system_id = None
        confidence = text_result.confidence
        
        for pattern_name, (pattern, equip_type) in self.HVAC_PATTERNS.items():
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                equipment_type = equip_type
                pattern_matched = pattern
                
                # Extract components
                parts = text.replace('-', ' ').split()
                if len(parts) >= 2:
                    system_id = parts[0]
                    zone_number = parts[1]
                elif len(parts) == 1:
                    # Try to split alphanumeric
                    alpha_match = re.match(r'([A-Z]+)(\d+)', text)
                    if alpha_match:
                        system_id = alpha_match.group(1)
                        zone_number = alpha_match.group(2)
                
                break
        
        # Find spatially associated component
        associated_component = self._find_associated_component(
            text_result.region,
            all_detections,
            image_width,
            image_height
        )
        
        return HVACInterpretation(
            text=text_result.text,
            equipment_type=equipment_type,
            zone_number=zone_number,
            system_id=system_id,
            confidence=confidence,
            pattern_matched=pattern_matched,
            associated_component=associated_component
        )
    
    def _find_associated_component(
        self,
        text_region: DetectionBox,
        all_detections: List[DetectionBox],
        image_width: int,
        image_height: int
    ) -> Optional[DetectionBox]:
        """
        Find the component most likely associated with a text region.
        
        Uses spatial proximity with a maximum distance threshold.
        
        Args:
            text_region: Text region box
            all_detections: All detected components
            image_width: Image width
            image_height: Image height
            
        Returns:
            Associated DetectionBox or None
        """
        text_center = text_region.center
        min_distance = float('inf')
        associated = None
        
        for detection in all_detections:
            # Skip text regions
            if detection.class_name.lower() in self.TEXT_CLASSES:
                continue
            
            # Calculate distance between centers
            det_center = detection.center
            distance = np.sqrt(
                (text_center[0] - det_center[0])**2 +
                (text_center[1] - det_center[1])**2
            )
            
            # Check if within association threshold
            max_distance = max(detection.width, detection.height) * self.config.max_association_distance_multiplier
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                associated = detection
        
        return associated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        avg_time = (
            self._total_processing_time / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        
        return {
            'total_requests': self._total_requests,
            'total_processing_time_ms': self._total_processing_time,
            'average_processing_time_ms': avg_time,
            'device': self.device,
            'models_loaded': {
                'yolo': self.yolo_engine is not None,
                'ocr': self.ocr_reader is not None
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pipeline components."""
        return {
            'status': 'healthy',
            'yolo_loaded': self.yolo_engine is not None,
            'ocr_loaded': self.ocr_reader is not None,
            'device': self.device,
            'config': {
                'confidence_threshold': self.config.confidence_threshold,
                'max_processing_time_ms': self.config.max_processing_time_ms,
                'enable_gpu': self.config.enable_gpu
            }
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


def create_hvac_analyzer(
    model_path: str,
    config: Optional[PipelineConfig] = None,
    device: Optional[str] = None
) -> HVACDrawingAnalyzer:
    """
    Factory function to create an HVAC Drawing Analyzer.
    
    Args:
        model_path: Path to YOLOv11-obb model
        config: Optional pipeline configuration
        device: Optional device specification
        
    Returns:
        Initialized HVACDrawingAnalyzer instance
    """
    return HVACDrawingAnalyzer(
        yolo_model_path=model_path,
        config=config,
        device=device
    )
