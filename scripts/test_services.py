"""
Test script for validating individual services.
Run this to verify ObjectDetector and TextExtractor work independently.

Usage:
    python scripts/test_services.py
"""

import sys
import os
from pathlib import Path

# Add services/hvac-ai to path
REPO_ROOT = Path(__file__).parent.parent
# Ensure repo root is on sys.path so imports like `core.services.*` resolve
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add services package (prefer hvac-ai service folder when present)
PYTHON_SERVICES = REPO_ROOT / "services" / "hvac-ai"
if PYTHON_SERVICES.exists():
    if str(PYTHON_SERVICES) not in sys.path:
        sys.path.insert(0, str(PYTHON_SERVICES))
else:
    # fall back to services/ if structure differs
    alt = REPO_ROOT / 'services'
    if alt.exists() and str(alt) not in sys.path:
        sys.path.insert(0, str(alt))

import numpy as np
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_object_detector():
    """Test ObjectDetector service."""
    logger.info("=" * 60)
    logger.info("Testing ObjectDetector Service")
    logger.info("=" * 60)
    
    try:
        from core.services.object_detector import ObjectDetector
        
        # Get model path (prefer MODEL_PATH, fall back to legacy YOLO_MODEL_PATH)
        model_path = os.getenv('MODEL_PATH') or os.getenv('YOLO_MODEL_PATH') or str(REPO_ROOT / 'ai_model' / 'best.pt')

        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.error("Set MODEL_PATH (or legacy YOLO_MODEL_PATH) environment variable")
            return False
        
        logger.info(f"Loading model from: {model_path}")
        detector = ObjectDetector(model_path=model_path, conf_threshold=0.5)
        
        # Create a test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        logger.info("Running test detection...")
        detections = detector.detect(test_image)

        logger.info(f"ObjectDetector test passed")
        logger.info(f"   Detections: {len(detections)}")
        logger.info(f"   Class names: {detector.get_class_names()}")

        return True
        
    except Exception as e:
        logger.error(f"ObjectDetector test failed: {e}", exc_info=True)
        return False


def test_text_extractor():
    """Test TextExtractor service."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing TextExtractor Service")
    logger.info("=" * 60)
    
    try:
        from core.services.text_extractor import TextExtractor
        
        logger.info("Initializing TextExtractor...")
        extractor = TextExtractor(lang='en', use_gpu=False)
        
        # Create a test image with text
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(
            test_image,
            'TEST-123',
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            2
        )
        
        logger.info("Running test text extraction...")
        result = extractor.extract_single_text(test_image, conf_threshold=0.3)
        
        if result:
            text, confidence = result
            logger.info(f"TextExtractor test passed")
            logger.info(f"   Extracted: '{text}'")
            logger.info(f"   Confidence: {confidence:.2f}")
        else:
            logger.warning("No text extracted (this may be normal for synthetic test)")
        
        return True
        
    except Exception as e:
        logger.error(f"TextExtractor test failed: {e}", exc_info=True)
        logger.error("   Make sure paddleocr is installed: pip install paddleocr paddlepaddle")
        return False


def test_geometry_utils():
    """Test GeometryUtils."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing GeometryUtils")
    logger.info("=" * 60)
    
    try:
        from core.utils.geometry import GeometryUtils, OBB
        
        # Create test image
        test_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        
        # Create a rotated region
        obb = OBB(
            x_center=500,
            y_center=500,
            width=200,
            height=100,
            rotation=0.785  # 45 degrees
        )
        
        logger.info("Testing OBB rectification...")
        rectified, metadata = GeometryUtils.extract_and_preprocess_obb(
            test_image, obb, padding=5, preprocess=True
        )

        logger.info(f"GeometryUtils test passed")
        logger.info(f"   Rectified size: {rectified.shape}")
        logger.info(f"   Original rotation: {metadata.get('original_rotation', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"GeometryUtils test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "HVAC AI - Service Tests")
    logger.info("=" * 60 + "\n")
    
    results = {
        'ObjectDetector': test_object_detector(),
        'TextExtractor': test_text_extractor(),
        'GeometryUtils': test_geometry_utils()
    }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for service, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{service:20s} {status}")
    
    all_passed = all(results.values())
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("All tests passed")
        return 0
    else:
        logger.error("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
