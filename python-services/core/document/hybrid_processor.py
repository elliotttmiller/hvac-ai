"""
Hybrid OCR + Vision-Language Model Processor
Implements advanced document understanding by combining traditional OCR with VLM

Based on research:
- Complex Document Recognition (HackerNoon)
- Commercial Proposals Parsing
- Vision-Language Model integration patterns
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OCREngine(Enum):
    """Supported OCR engines"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


@dataclass
class OCRResult:
    """OCR extraction result"""
    text: str
    bbox: tuple
    confidence: float
    source: str = "ocr"


@dataclass
class VLMResult:
    """VLM understanding result"""
    text: str
    context: str
    entities: List[Dict[str, Any]]
    confidence: float
    source: str = "vlm"


@dataclass
class HybridResult:
    """Combined OCR + VLM result"""
    text: str
    bbox: Optional[tuple]
    confidence: float
    validated: bool
    entities: List[Dict[str, Any]]
    context: str
    sources: List[str]


class TraditionalOCR:
    """
    Traditional OCR wrapper supporting multiple engines
    
    Provides unified interface for different OCR backends
    """
    
    def __init__(self, engine: OCREngine = OCREngine.TESSERACT):
        self.engine = engine
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize selected OCR engine"""
        if self.engine == OCREngine.TESSERACT:
            try:
                import pytesseract
                self.ocr_fn = self._tesseract_ocr
                logger.info("Initialized Tesseract OCR")
            except ImportError:
                logger.error("Tesseract not available")
                self.ocr_fn = self._fallback_ocr
        
        elif self.engine == OCREngine.EASYOCR:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'])
                self.ocr_fn = self._easyocr_ocr
                logger.info("Initialized EasyOCR")
            except ImportError:
                logger.error("EasyOCR not available")
                self.ocr_fn = self._fallback_ocr
        
        elif self.engine == OCREngine.PADDLEOCR:
            logger.warning("PaddleOCR not yet implemented, using fallback")
            self.ocr_fn = self._fallback_ocr
    
    def extract(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text from image
        
        Args:
            image: Input image
            
        Returns:
            List of OCR results
        """
        return self.ocr_fn(image)
    
    def _tesseract_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using Tesseract"""
        import pytesseract
        
        # Get detailed results with bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
            
            conf = float(data['conf'][i])
            if conf < 0:  # Invalid confidence
                continue
            
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            results.append(OCRResult(
                text=text,
                bbox=(x, y, w, h),
                confidence=conf / 100.0,  # Normalize to 0-1
                source="tesseract"
            ))
        
        return results
    
    def _easyocr_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using EasyOCR"""
        # EasyOCR returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
        detections = self.reader.readtext(image)
        
        results = []
        for detection in detections:
            bbox_points, text, conf = detection
            
            # Convert polygon to bounding box
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            x, y = min(x_coords), min(y_coords)
            w, h = max(x_coords) - x, max(y_coords) - y
            
            results.append(OCRResult(
                text=text,
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=float(conf),
                source="easyocr"
            ))
        
        return results
    
    def _fallback_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """Fallback when no OCR engine available"""
        logger.warning("No OCR engine available, returning empty results")
        return []


class VisionLanguageModel:
    """
    Vision-Language Model for contextual understanding
    
    Integrates with the VLM system in core/vlm/ for enhanced document understanding
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "qwen2-vl"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize VLM model"""
        try:
            # Import VLM interface from core
            from ..vlm.model_interface import VLMInterface
            self.model = VLMInterface(model_name=self.model_name)
            logger.info(f"Initialized VLM: {self.model_name}")
        except Exception as e:
            logger.warning(f"VLM not available: {e}")
            self.model = None
    
    def analyze(self, image: np.ndarray, ocr_results: Optional[List[OCRResult]] = None) -> VLMResult:
        """
        Analyze image with contextual understanding
        
        Args:
            image: Input image
            ocr_results: Optional OCR results to guide analysis
            
        Returns:
            VLM analysis result
        """
        if self.model is None:
            return self._fallback_analyze(image, ocr_results)
        
        # Construct prompt based on OCR results
        prompt = self._construct_prompt(ocr_results)
        
        try:
            # Use VLM to analyze image
            response = self.model.analyze_image(image, prompt)
            
            # Parse VLM response
            return self._parse_vlm_response(response)
        
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return self._fallback_analyze(image, ocr_results)
    
    def _construct_prompt(self, ocr_results: Optional[List[OCRResult]]) -> str:
        """Construct VLM prompt based on OCR results"""
        if not ocr_results:
            return """Analyze this HVAC blueprint and extract:
1. Project information (title, date, project number)
2. Equipment and components visible
3. Technical specifications and measurements
4. Notes and annotations
Provide structured output in JSON format."""
        
        # Include OCR hints in prompt
        ocr_texts = [r.text for r in ocr_results[:10]]  # First 10 for context
        return f"""Analyze this HVAC blueprint. OCR detected these texts: {', '.join(ocr_texts)}.

Extract and validate:
1. Project information
2. Equipment specifications
3. Technical measurements
4. Relationships between components

Provide structured output in JSON format."""
    
    def _parse_vlm_response(self, response: Any) -> VLMResult:
        """Parse VLM model response"""
        # Extract structured information from VLM response
        text = str(response.get('text', ''))
        entities = response.get('entities', [])
        context = response.get('context', '')
        confidence = response.get('confidence', 0.7)
        
        return VLMResult(
            text=text,
            context=context,
            entities=entities,
            confidence=confidence,
            source="vlm"
        )
    
    def _fallback_analyze(self, image: np.ndarray, ocr_results: Optional[List[OCRResult]]) -> VLMResult:
        """Fallback when VLM not available"""
        # Use OCR results as fallback
        if ocr_results:
            text = ' '.join([r.text for r in ocr_results])
            entities = [{'text': r.text, 'type': 'text'} for r in ocr_results]
        else:
            text = ""
            entities = []
        
        return VLMResult(
            text=text,
            context="Fallback analysis - VLM not available",
            entities=entities,
            confidence=0.5,
            source="fallback"
        )


class SemanticValidator:
    """
    Validates and merges OCR and VLM results
    
    Uses semantic understanding to resolve conflicts and improve accuracy
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
    
    def merge(self, ocr_results: List[OCRResult], vlm_result: VLMResult) -> List[HybridResult]:
        """
        Merge and validate OCR and VLM results
        
        Args:
            ocr_results: Results from OCR engine
            vlm_result: Result from VLM analysis
            
        Returns:
            Validated and merged results
        """
        merged_results = []
        
        # Strategy 1: Use VLM to validate OCR results
        for ocr in ocr_results:
            validated = self._validate_ocr_with_vlm(ocr, vlm_result)
            merged_results.append(validated)
        
        # Strategy 2: Add VLM-only entities not captured by OCR
        vlm_entities = self._extract_vlm_only_entities(vlm_result, ocr_results)
        merged_results.extend(vlm_entities)
        
        # Strategy 3: Filter low-confidence results
        merged_results = [r for r in merged_results if r.confidence >= self.confidence_threshold]
        
        return merged_results
    
    def _validate_ocr_with_vlm(self, ocr: OCRResult, vlm: VLMResult) -> HybridResult:
        """Validate OCR result using VLM context"""
        # Check if OCR text appears in VLM result
        text_in_vlm = ocr.text.lower() in vlm.text.lower()
        
        # Find matching entities in VLM
        matching_entities = [
            e for e in vlm.entities 
            if ocr.text.lower() in str(e.get('text', '')).lower()
        ]
        
        # Compute validation score
        validation_score = 0.5  # Base score
        if text_in_vlm:
            validation_score += 0.3
        if matching_entities:
            validation_score += 0.2
        
        # Combine confidences
        combined_confidence = (ocr.confidence + vlm.confidence * validation_score) / 2
        
        return HybridResult(
            text=ocr.text,
            bbox=ocr.bbox,
            confidence=combined_confidence,
            validated=validation_score > 0.6,
            entities=matching_entities,
            context=vlm.context,
            sources=["ocr", "vlm"]
        )
    
    def _extract_vlm_only_entities(self, vlm: VLMResult, ocr_results: List[OCRResult]) -> List[HybridResult]:
        """Extract entities found by VLM but not by OCR"""
        vlm_only = []
        
        ocr_texts = set(r.text.lower() for r in ocr_results)
        
        for entity in vlm.entities:
            entity_text = str(entity.get('text', '')).lower()
            
            # Skip if already captured by OCR
            if any(entity_text in ocr_text for ocr_text in ocr_texts):
                continue
            
            vlm_only.append(HybridResult(
                text=entity.get('text', ''),
                bbox=None,  # No spatial location from VLM
                confidence=vlm.confidence,
                validated=True,
                entities=[entity],
                context=vlm.context,
                sources=["vlm"]
            ))
        
        return vlm_only


class HybridProcessor:
    """
    Main hybrid OCR + VLM processor
    
    Implements the complete hybrid pipeline:
    1. Traditional OCR for raw text extraction
    2. VLM for contextual understanding
    3. Semantic validation and merging
    """
    
    def __init__(self, 
                 ocr_engine: OCREngine = OCREngine.EASYOCR,
                 vlm_model: Optional[str] = None,
                 confidence_threshold: float = 0.6):
        """
        Initialize hybrid processor
        
        Args:
            ocr_engine: OCR engine to use
            vlm_model: VLM model name (None for default)
            confidence_threshold: Minimum confidence for results
        """
        self.ocr = TraditionalOCR(engine=ocr_engine)
        self.vlm = VisionLanguageModel(model_name=vlm_model)
        self.validator = SemanticValidator(confidence_threshold=confidence_threshold)
        
        logger.info("Hybrid Processor initialized")
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image through hybrid pipeline
        
        Args:
            image: Input blueprint image
            
        Returns:
            Structured results with high accuracy
        """
        # Stage 1: OCR extraction
        logger.info("Stage 1: OCR extraction")
        ocr_results = self.ocr.extract(image)
        logger.info(f"OCR found {len(ocr_results)} text blocks")
        
        # Stage 2: VLM analysis
        logger.info("Stage 2: VLM contextual analysis")
        vlm_result = self.vlm.analyze(image, ocr_results)
        logger.info(f"VLM found {len(vlm_result.entities)} entities")
        
        # Stage 3: Validation and merging
        logger.info("Stage 3: Semantic validation and merging")
        merged_results = self.validator.merge(ocr_results, vlm_result)
        logger.info(f"Final: {len(merged_results)} validated results")
        
        # Structure output
        return {
            'results': merged_results,
            'metadata': {
                'ocr_count': len(ocr_results),
                'vlm_entities': len(vlm_result.entities),
                'validated_count': len(merged_results),
                'vlm_confidence': vlm_result.confidence,
                'sources': ['ocr', 'vlm']
            },
            'context': vlm_result.context
        }
    
    def process_with_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process specific regions with hybrid approach
        
        Args:
            image: Full document image
            regions: List of regions to process
            
        Returns:
            Region-wise processing results
        """
        region_results = []
        
        for region in regions:
            bbox = region['bbox']
            x, y, w, h = bbox
            
            # Extract region
            region_image = image[y:y+h, x:x+w]
            
            # Process region
            result = self.process(region_image)
            result['bbox'] = bbox
            result['region_type'] = region.get('type', 'unknown')
            
            region_results.append(result)
        
        return {
            'regions': region_results,
            'total_regions': len(region_results)
        }


def create_hybrid_processor(ocr_engine: str = "easyocr",
                           vlm_model: Optional[str] = None,
                           confidence_threshold: float = 0.6) -> HybridProcessor:
    """
    Factory function to create hybrid processor
    
    Args:
        ocr_engine: OCR engine name ("tesseract", "easyocr", "paddleocr")
        vlm_model: VLM model name (None for default)
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Configured hybrid processor
    """
    # Convert string to enum
    engine_map = {
        "tesseract": OCREngine.TESSERACT,
        "easyocr": OCREngine.EASYOCR,
        "paddleocr": OCREngine.PADDLEOCR
    }
    
    engine = engine_map.get(ocr_engine.lower(), OCREngine.EASYOCR)
    
    return HybridProcessor(
        ocr_engine=engine,
        vlm_model=vlm_model,
        confidence_threshold=confidence_threshold
    )
