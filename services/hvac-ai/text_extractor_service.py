"""
Text Extractor Service
Wraps PaddleOCR for text recognition on blueprint crops.
"""

# --- CRITICAL: Disable Phone-Home Checks BEFORE any PaddleOCR imports ---
import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import logging
import numpy as np
from typing import Optional, Tuple, List

logger = logging.getLogger("HVAC-AI")

class TextExtractor:
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.ocr_engine = None
        self._load_engine()

    def _load_engine(self):
        try:
            from paddleocr import PaddleOCR
            # Initialize PaddleOCR with minimal parameters
            # GPU is controlled via CUDA_VISIBLE_DEVICES environment variable
            # use_angle_cls is deprecated, using use_textline_orientation instead
            self.ocr_engine = PaddleOCR(
                lang=self.lang,
                use_textline_orientation=False,  # Disable text line orientation (replaces use_angle_cls)
                use_doc_orientation_classify=False,  # Disable document orientation
                use_doc_unwarping=False,  # Disable document unwarping
            )
            logger.info(f"PaddleOCR engine loaded (lang={self.lang}, gpu disabled via env)")

            # Warmup to force model download/load
            dummy = np.zeros((50, 50, 3), dtype=np.uint8)
            # Use predict() method instead of deprecated ocr() with parameters
            self.ocr_engine.predict(dummy)

        except Exception as e:
            logger.error(f"Failed to load OCR engine: {e}")
            # Don't crash app, just log. Inference will fail gracefully later.

    def extract_single_text(self, image: np.ndarray, conf_threshold: float = 0.5) -> Optional[Tuple[str, float]]:
        """
        Runs OCR on a single image crop.
        Returns (text, confidence) or None.
        """
        if self.ocr_engine is None:
            return None

        try:
            # Use predict() method for OCR inference
            # det=False because YOLO already found the text box
            # cls=False because we handled rotation via GeometryUtils
            result = self.ocr_engine.predict(image)

            # PaddleOCR predict() result parsing
            if not result:
                return None

            # Handle various return formats (list of lists, or list of tuples)
            text_info = None
            if isinstance(result, list) and len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, list) and len(first_item) > 0:
                    text_info = first_item[0]
                elif isinstance(first_item, tuple):
                    text_info = first_item

            if text_info and isinstance(text_info, tuple) and len(text_info) >= 2:
                text = str(text_info[0])
                conf = float(text_info[1])

                if conf >= conf_threshold:
                    return text, conf

        except Exception as e:
            logger.warning(f"OCR Inference failed: {e}")

        return None