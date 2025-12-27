"""
Unit tests for hybrid OCR + VLM processor
Tests OCR engines, VLM integration, and semantic validation
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python_services.core.document.hybrid_processor import (
    OCREngine,
    OCRResult,
    VLMResult,
    HybridResult,
    TraditionalOCR,
    VisionLanguageModel,
    SemanticValidator,
    HybridProcessor,
    create_hybrid_processor
)


class TestOCRResult(unittest.TestCase):
    """Test OCR result dataclass"""
    
    def test_create_ocr_result(self):
        """Test creating OCR result"""
        result = OCRResult(
            text="Test text",
            bbox=(10, 20, 100, 50),
            confidence=0.95,
            source="tesseract"
        )
        
        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.bbox, (10, 20, 100, 50))
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.source, "tesseract")


class TestVLMResult(unittest.TestCase):
    """Test VLM result dataclass"""
    
    def test_create_vlm_result(self):
        """Test creating VLM result"""
        entities = [
            {'text': 'Entity 1', 'type': 'equipment'},
            {'text': 'Entity 2', 'type': 'spec'}
        ]
        
        result = VLMResult(
            text="Analyzed text",
            context="Context information",
            entities=entities,
            confidence=0.85,
            source="vlm"
        )
        
        self.assertEqual(result.text, "Analyzed text")
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.confidence, 0.85)


class TestHybridResult(unittest.TestCase):
    """Test hybrid result dataclass"""
    
    def test_create_hybrid_result(self):
        """Test creating hybrid result"""
        result = HybridResult(
            text="Validated text",
            bbox=(10, 20, 100, 50),
            confidence=0.92,
            validated=True,
            entities=[{'text': 'Entity', 'type': 'equipment'}],
            context="Context",
            sources=["ocr", "vlm"]
        )
        
        self.assertEqual(result.text, "Validated text")
        self.assertTrue(result.validated)
        self.assertEqual(len(result.sources), 2)


class TestTraditionalOCR(unittest.TestCase):
    """Test traditional OCR wrapper"""
    
    def setUp(self):
        self.ocr = TraditionalOCR(engine=OCREngine.TESSERACT)
        
    def create_text_image(self, text="TEST"):
        """Create a simple image with text"""
        image = np.ones((100, 400), dtype=np.uint8) * 255
        cv2.putText(image, text, (50, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        return image
    
    def test_initialization(self):
        """Test OCR initialization"""
        self.assertIsNotNone(self.ocr)
        self.assertEqual(self.ocr.engine, OCREngine.TESSERACT)
    
    def test_extract_returns_list(self):
        """Test that extract returns a list"""
        image = self.create_text_image()
        results = self.ocr.extract(image)
        
        self.assertIsInstance(results, list)
    
    def test_extract_with_text(self):
        """Test extraction from image with text"""
        image = self.create_text_image("HELLO")
        results = self.ocr.extract(image)
        
        # Should return OCRResult objects
        for result in results:
            self.assertIsInstance(result, OCRResult)
            self.assertIsInstance(result.text, str)
            self.assertIsInstance(result.confidence, float)
    
    def test_fallback_ocr(self):
        """Test fallback when OCR not available"""
        ocr = TraditionalOCR(engine=OCREngine.PADDLEOCR)  # Not implemented
        image = self.create_text_image()
        results = ocr.extract(image)
        
        self.assertIsInstance(results, list)


class TestVisionLanguageModel(unittest.TestCase):
    """Test Vision-Language Model wrapper"""
    
    def setUp(self):
        self.vlm = VisionLanguageModel()
        
    def create_test_image(self):
        """Create a test image"""
        return np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """Test VLM initialization"""
        self.assertIsNotNone(self.vlm)
    
    def test_analyze_returns_vlm_result(self):
        """Test that analyze returns VLMResult"""
        image = self.create_test_image()
        result = self.vlm.analyze(image)
        
        self.assertIsInstance(result, VLMResult)
        self.assertIsInstance(result.text, str)
        self.assertIsInstance(result.entities, list)
        self.assertIsInstance(result.confidence, float)
    
    def test_analyze_with_ocr_results(self):
        """Test analysis with OCR hints"""
        image = self.create_test_image()
        ocr_results = [
            OCRResult("Text 1", (0, 0, 100, 50), 0.9, "ocr"),
            OCRResult("Text 2", (100, 0, 200, 50), 0.85, "ocr")
        ]
        
        result = self.vlm.analyze(image, ocr_results)
        
        self.assertIsInstance(result, VLMResult)
    
    def test_fallback_analyze(self):
        """Test fallback when VLM not available"""
        image = self.create_test_image()
        result = self.vlm.analyze(image)
        
        # Should still return a result (fallback)
        self.assertIsInstance(result, VLMResult)
        self.assertEqual(result.source, "fallback")


class TestSemanticValidator(unittest.TestCase):
    """Test semantic validation and merging"""
    
    def setUp(self):
        self.validator = SemanticValidator(confidence_threshold=0.6)
        
    def test_merge_empty_results(self):
        """Test merging with no results"""
        ocr_results = []
        vlm_result = VLMResult(
            text="",
            context="",
            entities=[],
            confidence=0.7,
            source="vlm"
        )
        
        merged = self.validator.merge(ocr_results, vlm_result)
        
        self.assertIsInstance(merged, list)
    
    def test_merge_with_results(self):
        """Test merging OCR and VLM results"""
        ocr_results = [
            OCRResult("Equipment A", (0, 0, 100, 50), 0.9, "ocr"),
            OCRResult("Spec B", (100, 0, 200, 50), 0.85, "ocr")
        ]
        
        vlm_result = VLMResult(
            text="Equipment A and Spec B detected",
            context="HVAC equipment context",
            entities=[
                {'text': 'Equipment A', 'type': 'equipment'},
                {'text': 'Spec B', 'type': 'specification'}
            ],
            confidence=0.88,
            source="vlm"
        )
        
        merged = self.validator.merge(ocr_results, vlm_result)
        
        self.assertIsInstance(merged, list)
        
        # Should have hybrid results
        for result in merged:
            self.assertIsInstance(result, HybridResult)
            self.assertGreaterEqual(result.confidence, self.validator.confidence_threshold)
    
    def test_validate_ocr_with_vlm(self):
        """Test OCR validation using VLM"""
        ocr = OCRResult("Test text", (0, 0, 100, 50), 0.8, "ocr")
        vlm = VLMResult(
            text="Test text found in document",
            context="Document context",
            entities=[{'text': 'Test text', 'type': 'text'}],
            confidence=0.85,
            source="vlm"
        )
        
        result = self.validator._validate_ocr_with_vlm(ocr, vlm)
        
        self.assertIsInstance(result, HybridResult)
        self.assertTrue(result.validated)
        self.assertIn("ocr", result.sources)
        self.assertIn("vlm", result.sources)
    
    def test_filter_low_confidence(self):
        """Test that low confidence results are filtered"""
        ocr_results = [
            OCRResult("High conf", (0, 0, 100, 50), 0.9, "ocr"),
            OCRResult("Low conf", (100, 0, 200, 50), 0.3, "ocr")
        ]
        
        vlm_result = VLMResult(
            text="Only high confidence text",
            context="",
            entities=[],
            confidence=0.5,
            source="vlm"
        )
        
        merged = self.validator.merge(ocr_results, vlm_result)
        
        # Low confidence results should be filtered
        for result in merged:
            self.assertGreaterEqual(result.confidence, self.validator.confidence_threshold)


class TestHybridProcessor(unittest.TestCase):
    """Test complete hybrid processor"""
    
    def setUp(self):
        self.processor = create_hybrid_processor(
            ocr_engine="tesseract",
            confidence_threshold=0.6
        )
        
    def create_test_blueprint(self):
        """Create a test blueprint with text"""
        image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
        
        # Add some text
        cv2.putText(image, "HVAC SYSTEM", (100, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(image, "EQUIPMENT", (100, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        return image
    
    def test_initialization(self):
        """Test processor initialization"""
        self.assertIsNotNone(self.processor)
        self.assertIsNotNone(self.processor.ocr)
        self.assertIsNotNone(self.processor.vlm)
        self.assertIsNotNone(self.processor.validator)
    
    def test_full_pipeline(self):
        """Test complete hybrid processing pipeline"""
        image = self.create_test_blueprint()
        results = self.processor.process(image)
        
        # Check structure
        self.assertIn('results', results)
        self.assertIn('metadata', results)
        self.assertIn('context', results)
        
        # Check metadata
        metadata = results['metadata']
        self.assertIn('ocr_count', metadata)
        self.assertIn('vlm_entities', metadata)
        self.assertIn('validated_count', metadata)
        self.assertIn('sources', metadata)
    
    def test_results_are_hybrid(self):
        """Test that results are HybridResult objects"""
        image = self.create_test_blueprint()
        results = self.processor.process(image)
        
        for result in results['results']:
            self.assertIsInstance(result, HybridResult)
    
    def test_process_with_regions(self):
        """Test processing specific regions"""
        image = self.create_test_blueprint()
        
        regions = [
            {
                'bbox': (0, 0, 500, 400),
                'type': 'title_block'
            },
            {
                'bbox': (500, 0, 1000, 400),
                'type': 'schedule'
            }
        ]
        
        results = self.processor.process_with_regions(image, regions)
        
        self.assertIn('regions', results)
        self.assertIn('total_regions', results)
        self.assertEqual(results['total_regions'], len(regions))


class TestFactoryFunction(unittest.TestCase):
    """Test factory function"""
    
    def test_create_default(self):
        """Test creating processor with defaults"""
        processor = create_hybrid_processor()
        
        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor.ocr)
        self.assertIsNotNone(processor.vlm)
    
    def test_create_with_tesseract(self):
        """Test creating with Tesseract"""
        processor = create_hybrid_processor(ocr_engine="tesseract")
        
        self.assertEqual(processor.ocr.engine, OCREngine.TESSERACT)
    
    def test_create_with_easyocr(self):
        """Test creating with EasyOCR"""
        processor = create_hybrid_processor(ocr_engine="easyocr")
        
        self.assertEqual(processor.ocr.engine, OCREngine.EASYOCR)
    
    def test_create_with_custom_threshold(self):
        """Test creating with custom confidence threshold"""
        processor = create_hybrid_processor(confidence_threshold=0.8)
        
        self.assertEqual(processor.validator.confidence_threshold, 0.8)


if __name__ == '__main__':
    unittest.main()
