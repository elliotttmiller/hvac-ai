"""
TextExtractor Service - Universal Text Recognition Interface
Tool-agnostic wrapper for OCR engines (currently PaddleOCR).

Following Domain-Driven Design: Uses universal terminology (TextExtractor)
rather than implementation-specific names (PaddleOCRWrapper).
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Universal text extraction service.
    Currently implements PaddleOCR for high-accuracy text recognition.
    
    Design Philosophy:
    - Generic interface that abstracts the underlying OCR engine
    - Easy to swap PaddleOCR for EasyOCR, Tesseract, or GPT-4V without changing API
    - Returns standardized text extraction format
    - Supports batch processing for efficiency
    """
    
    def __init__(
        self,
        lang: str = 'en',
        use_angle_cls: bool = False,
        use_gpu: bool = True,
        enable_mkldnn: bool = False
    ):
        """
        Initialize the text extractor.
        
        Args:
            lang: Language code (default: 'en' for English)
            use_angle_cls: Enable angle classification (we handle rotation via geometry)
            use_gpu: Use GPU acceleration if available
            enable_mkldnn: Enable MKLDNN for CPU optimization
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        
        self.ocr_engine = None
        
        # Load OCR engine
        self._load_engine()
        
        logger.info(f"[OK] TextExtractor initialized")
        logger.info(f"   Language: {self.lang}")
        logger.info(f"   Use GPU: {self.use_gpu}")
        logger.info(f"   Angle classification: {self.use_angle_cls}")
    
    def _load_engine(self):
        """Load the OCR engine (internal implementation detail)."""
        try:
            logger.info("[LOAD] Loading OCR engine (PaddleOCR)...")
            
            # Import PaddleOCR
            try:
                from paddleocr import PaddleOCR  # type: ignore
            except ImportError:
                logger.error("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")
                raise RuntimeError("PaddleOCR not available")
            
            # Initialize PaddleOCR
            # Note: Newer versions of PaddleOCR auto-detect GPU and removed explicit flags
            # We pass only the essential parameters to be safe across different PaddleOCR versions
            self.ocr_engine = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang
            )
            
            logger.info("[OK] OCR engine loaded successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load OCR engine: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize TextExtractor: {e}")
    
    def extract_text(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a single image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR or RGB format
            conf_threshold: Minimum confidence threshold for text detections
            
        Returns:
            List of text extractions, each containing:
            - text: Extracted text string
            - confidence: Confidence score (0-1)
            - bbox: Bounding box coordinates (optional)
        """
        if self.ocr_engine is None:
            raise RuntimeError("OCR engine not loaded")
        
        try:
            # Run OCR
            results = self.ocr_engine.ocr(image, cls=self.use_angle_cls)
            
            extractions = []
            
            # Process results
            if results and results[0]:
                for line in results[0]:
                    if line:
                        # PaddleOCR returns: [bbox, (text, confidence)]
                        bbox_points = line[0]  # 4 corner points
                        text_info = line[1]
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        # Filter by confidence
                        if confidence >= conf_threshold:
                            extraction = {
                                'text': text,
                                'confidence': float(confidence),
                                'bbox': bbox_points
                            }
                            extractions.append(extraction)
            
            logger.debug(f"Extracted {len(extractions)} text regions")
            return extractions
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}", exc_info=True)
            return []
    
    def extract_text_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract text from multiple images in batch.
        
        Args:
            images: List of input images as numpy arrays
            conf_threshold: Minimum confidence threshold
            
        Returns:
            List of extraction results, one per input image
        """
        batch_results = []
        
        for image in images:
            results = self.extract_text(image, conf_threshold)
            batch_results.append(results)
        
        return batch_results
    
    def extract_text_from_crops(
        self,
        crops: List[np.ndarray],
        conf_threshold: float = 0.5
    ) -> List[Optional[str]]:
        """
        Extract text from cropped regions, returning just the text string.
        Optimized for the common case of extracting text from pre-cropped regions.
        
        Args:
            crops: List of cropped images
            conf_threshold: Minimum confidence threshold
            
        Returns:
            List of extracted text strings (or None if no text found)
        """
        results = []
        
        for crop in crops:
            extractions = self.extract_text(crop, conf_threshold)
            
            if extractions:
                # Return the text with highest confidence
                best = max(extractions, key=lambda x: x['confidence'])
                results.append(best['text'])
            else:
                results.append(None)
        
        return results
    
    def extract_single_text(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5
    ) -> Optional[Tuple[str, float]]:
        """
        Extract a single text string from an image (e.g., a pre-cropped tag).
        Returns the most confident text detection.
        
        Args:
            image: Input image
            conf_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (text, confidence) or None if no text found
        """
        extractions = self.extract_text(image, conf_threshold)
        
        if not extractions:
            return None
        
        # Return the most confident detection
        best = max(extractions, key=lambda x: x['confidence'])
        return best['text'], best['confidence']
