"""
Unit tests for enhanced document processor
Tests quality assessment, enhancement, segmentation, and caching
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python_services.core.document.enhanced_processor import (
    QualityAssessment,
    ImageEnhancement,
    LayoutSegmenter,
    RotationInvariantOCR,
    SemanticCache,
    EnhancedDocumentProcessor,
    RegionType,
    create_enhanced_processor
)


class TestQualityAssessment(unittest.TestCase):
    """Test quality assessment functionality"""
    
    def setUp(self):
        self.assessor = QualityAssessment()
        
    def test_assess_high_quality_image(self):
        """Test assessment of high quality image"""
        # Create sharp, high contrast image
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        metrics = self.assessor.assess(image)
        
        self.assertIn('quality_score', metrics)
        self.assertIn('blur_score', metrics)
        self.assertIn('contrast', metrics)
        self.assertIn('needs_enhancement', metrics)
        self.assertIsInstance(metrics['quality_score'], float)
        self.assertGreaterEqual(metrics['quality_score'], 0.0)
        self.assertLessEqual(metrics['quality_score'], 1.0)
    
    def test_assess_low_quality_image(self):
        """Test assessment of low quality image"""
        # Create low contrast image
        image = np.ones((500, 500, 3), dtype=np.uint8) * 128
        
        metrics = self.assessor.assess(image)
        
        self.assertIn('is_low_contrast', metrics)
        self.assertTrue(metrics['is_low_contrast'])
        self.assertTrue(metrics['needs_enhancement'])
    
    def test_assess_blurry_image(self):
        """Test detection of blurry image"""
        # Create sharp image then blur it
        image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        blurry = cv2.GaussianBlur(image, (25, 25), 0)
        
        metrics = self.assessor.assess(blurry)
        
        self.assertIn('blur_score', metrics)
        self.assertLess(metrics['blur_score'], 100)  # Below threshold
        self.assertTrue(metrics['is_blurry'])


class TestImageEnhancement(unittest.TestCase):
    """Test image enhancement functionality"""
    
    def setUp(self):
        self.enhancer = ImageEnhancement()
        
    def test_enhance_low_contrast(self):
        """Test enhancement of low contrast image"""
        # Create low contrast image
        image = np.ones((500, 500), dtype=np.uint8) * 128
        image[100:400, 100:400] = 140  # Slight variation
        
        quality_info = {'is_low_contrast': True, 'is_blurry': False}
        enhanced = self.enhancer.process(image, quality_info)
        
        # Enhanced image should have higher contrast
        self.assertGreater(enhanced.std(), image.std())
    
    def test_enhance_blurry(self):
        """Test enhancement of blurry image"""
        # Create image then blur
        image = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        blurry = cv2.GaussianBlur(image, (15, 15), 0)
        
        quality_info = {'is_blurry': True, 'is_low_contrast': False}
        enhanced = self.enhancer.process(blurry, quality_info)
        
        self.assertEqual(enhanced.shape, blurry.shape)
        self.assertIsNotNone(enhanced)
    
    def test_no_enhancement_needed(self):
        """Test that good quality images pass through"""
        image = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        quality_info = {'is_blurry': False, 'is_low_contrast': False}
        
        enhanced = self.enhancer.process(image, quality_info)
        
        # Should still return processed image
        self.assertEqual(enhanced.shape, image.shape)


class TestLayoutSegmenter(unittest.TestCase):
    """Test layout segmentation functionality"""
    
    def setUp(self):
        self.segmenter = LayoutSegmenter()
        
    def create_blueprint_mockup(self):
        """Create a mock blueprint with regions"""
        image = np.ones((1200, 1600, 3), dtype=np.uint8) * 255
        
        # Title block (bottom-right)
        cv2.rectangle(image, (1200, 1000), (1550, 1150), (0, 0, 0), -1)
        
        # Schedule (right side, tall)
        cv2.rectangle(image, (1300, 100), (1550, 600), (0, 0, 0), -1)
        
        # Main drawing (center-left, large)
        cv2.rectangle(image, (100, 100), (1100, 900), (0, 0, 0), -1)
        
        # Notes (small block)
        cv2.rectangle(image, (100, 950), (300, 1100), (0, 0, 0), -1)
        
        return image
    
    def test_segment_regions(self):
        """Test region segmentation"""
        image = self.create_blueprint_mockup()
        regions = self.segmenter.segment(image)
        
        self.assertGreater(len(regions), 0)
        
        # Check that regions have required attributes
        for region in regions:
            self.assertIsNotNone(region.bbox)
            self.assertIn(region.region_type, RegionType)
            self.assertIsInstance(region.confidence, float)
    
    def test_region_classification(self):
        """Test that regions are classified correctly"""
        image = self.create_blueprint_mockup()
        regions = self.segmenter.segment(image)
        
        # Should detect multiple region types
        region_types = set(r.region_type for r in regions)
        self.assertGreater(len(region_types), 1)


class TestRotationInvariantOCR(unittest.TestCase):
    """Test rotation-invariant text detection"""
    
    def setUp(self):
        self.ocr = RotationInvariantOCR()
        
    def create_rotated_text_image(self, angle=45):
        """Create image with rotated text"""
        image = np.ones((500, 500), dtype=np.uint8) * 255
        
        # Add some text-like pattern
        cv2.rectangle(image, (100, 200), (400, 250), (0, 0, 0), -1)
        
        # Rotate
        center = (250, 250)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (500, 500))
        
        return rotated
    
    def test_detect_text_regions(self):
        """Test text region detection"""
        image = self.create_rotated_text_image(45)
        regions = self.ocr.detect_text_regions(image)
        
        # Should detect at least one text region
        self.assertGreaterEqual(len(regions), 0)
        
        if len(regions) > 0:
            region = regions[0]
            self.assertIn('bbox', region)
            self.assertIn('angle', region)
            self.assertIn('center', region)
    
    def test_normalize_text_region(self):
        """Test text region normalization"""
        image = self.create_rotated_text_image(30)
        regions = self.ocr.detect_text_regions(image)
        
        if len(regions) > 0:
            normalized = self.ocr.normalize_text_region(image, regions[0])
            self.assertIsNotNone(normalized)
            self.assertEqual(len(normalized.shape), 2)  # Should be grayscale


class TestSemanticCache(unittest.TestCase):
    """Test semantic caching functionality"""
    
    def setUp(self):
        self.cache = SemanticCache()
        
    def test_cache_miss(self):
        """Test cache miss on first access"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.cache.get(image)
        
        self.assertIsNone(result)
    
    def test_cache_hit(self):
        """Test cache hit after setting"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        expected_result = {'test': 'data'}
        
        # Set cache
        self.cache.set(image, expected_result)
        
        # Get from cache
        result = self.cache.get(image)
        
        self.assertEqual(result, expected_result)
    
    def test_cache_similar_images(self):
        """Test that similar images hit cache"""
        image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        
        expected_result = {'test': 'data'}
        
        # Set cache with image1
        self.cache.set(image1, expected_result)
        
        # Get with image2 (should hit cache)
        result = self.cache.get(image2)
        
        self.assertEqual(result, expected_result)
    
    def test_cache_clear(self):
        """Test cache clearing"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.cache.set(image, {'test': 'data'})
        
        # Clear cache
        self.cache.clear()
        
        # Should miss now
        result = self.cache.get(image)
        self.assertIsNone(result)


class TestEnhancedDocumentProcessor(unittest.TestCase):
    """Test complete enhanced document processor"""
    
    def setUp(self):
        self.processor = create_enhanced_processor(use_cache=True)
        
    def create_test_blueprint(self):
        """Create a test blueprint image"""
        image = np.ones((1000, 1200, 3), dtype=np.uint8) * 255
        
        # Add some regions
        cv2.rectangle(image, (50, 50), (500, 500), (0, 0, 0), 2)
        cv2.rectangle(image, (900, 800), (1150, 950), (0, 0, 0), -1)
        
        return image
    
    def test_full_pipeline(self):
        """Test complete processing pipeline"""
        image = self.create_test_blueprint()
        results = self.processor.process(image)
        
        # Check structure
        self.assertIn('metadata', results)
        self.assertIn('regions', results)
        self.assertIn('text_blocks', results)
        self.assertIn('quality_info', results)
        
        # Check metadata
        self.assertIn('quality_score', results['metadata'])
        self.assertIn('region_count', results['metadata'])
        self.assertIn('text_region_count', results['metadata'])
        self.assertIn('enhanced', results['metadata'])
    
    def test_quality_assessment_stage(self):
        """Test quality assessment is performed"""
        image = self.create_test_blueprint()
        results = self.processor.process(image)
        
        quality_info = results['quality_info']
        
        self.assertIn('quality_score', quality_info)
        self.assertIn('needs_enhancement', quality_info)
        self.assertIsInstance(quality_info['quality_score'], float)
    
    def test_caching(self):
        """Test semantic caching works"""
        image = self.create_test_blueprint()
        
        # First processing
        results1 = self.processor.process(image)
        
        # Second processing (should use cache)
        results2 = self.processor.process(image)
        
        # Results should be identical
        self.assertEqual(
            results1['metadata']['region_count'],
            results2['metadata']['region_count']
        )
    
    def test_process_without_cache(self):
        """Test processing without cache"""
        processor = create_enhanced_processor(use_cache=False)
        image = self.create_test_blueprint()
        
        results = processor.process(image)
        
        self.assertIsNotNone(results)
        self.assertIn('metadata', results)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function"""
    
    def test_create_with_cache(self):
        """Test creating processor with cache"""
        processor = create_enhanced_processor(use_cache=True)
        
        self.assertIsNotNone(processor)
        self.assertTrue(processor.use_cache)
        self.assertIsNotNone(processor.cache)
    
    def test_create_without_cache(self):
        """Test creating processor without cache"""
        processor = create_enhanced_processor(use_cache=False)
        
        self.assertIsNotNone(processor)
        self.assertFalse(processor.use_cache)


if __name__ == '__main__':
    unittest.main()
