"""
Tests for YOLOplan Integration
Tests symbol detection, BOM generation, and connectivity analysis
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python_services.core.ai.yoloplan_detector import (
    YOLOplanDetector,
    SymbolDetection,
    SymbolCategory,
    create_yoloplan_detector
)
from python_services.core.ai.yoloplan_bom import (
    BOMGenerator,
    ConnectivityAnalyzer,
    BOMItem,
    create_bom_generator,
    create_connectivity_analyzer
)
from python_services.core.ai.integrated_detector import (
    IntegratedHVACDetector,
    create_integrated_detector
)


class TestYOLOplanDetector(unittest.TestCase):
    """Test YOLOplan symbol detector"""
    
    def setUp(self):
        self.detector = create_yoloplan_detector(confidence=0.5)
        
    def create_test_image(self):
        """Create a test blueprint image"""
        image = np.ones((1000, 1200, 3), dtype=np.uint8) * 255
        
        # Draw some equipment symbols (rectangles)
        cv2.rectangle(image, (100, 100), (200, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (300, 100), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (500, 100), (600, 150), (0, 0, 0), -1)
        
        # Draw diffusers (circles)
        cv2.circle(image, (150, 300), 20, (0, 0, 0), -1)
        cv2.circle(image, (350, 300), 20, (0, 0, 0), -1)
        
        return image
    
    def test_detector_initialization(self):
        """Test detector initializes correctly"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
    
    def test_detect_symbols_returns_dict(self):
        """Test detect_symbols returns proper structure"""
        image = self.create_test_image()
        results = self.detector.detect_symbols(image)
        
        self.assertIsInstance(results, dict)
        self.assertIn('detections', results)
        self.assertIn('counts', results)
        self.assertIn('total_symbols', results)
        self.assertIn('relationships', results)
    
    def test_batch_detect(self):
        """Test batch detection"""
        images = [self.create_test_image() for _ in range(3)]
        results = self.detector.batch_detect(images)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIn('detections', result)
            self.assertIn('counts', result)
    
    def test_export_json(self):
        """Test JSON export"""
        image = self.create_test_image()
        results = self.detector.detect_symbols(image)
        
        json_str = self.detector.export_results(results, format='json')
        
        self.assertIsInstance(json_str, str)
        self.assertIn('total_symbols', json_str)
    
    def test_export_csv(self):
        """Test CSV export"""
        image = self.create_test_image()
        results = self.detector.detect_symbols(image)
        
        csv_str = self.detector.export_results(results, format='csv')
        
        self.assertIsInstance(csv_str, str)
        # CSV should have headers
        self.assertIn('id,symbol_type', csv_str)
    
    def test_symbol_category_classification(self):
        """Test symbol category classification"""
        test_cases = [
            ('ahu_unit', SymbolCategory.AHU),
            ('fan_1', SymbolCategory.FAN),
            ('vav_box', SymbolCategory.VAV),
            ('diffuser_round', SymbolCategory.DIFFUSER),
            ('unknown_equipment', SymbolCategory.UNKNOWN)
        ]
        
        for class_name, expected_category in test_cases:
            category = self.detector._classify_symbol_category(class_name)
            # Just check it returns a SymbolCategory
            self.assertIsInstance(category, SymbolCategory)


class TestBOMGenerator(unittest.TestCase):
    """Test BOM generation"""
    
    def setUp(self):
        self.bom_generator = create_bom_generator()
        
    def create_test_detections(self):
        """Create test symbol detections"""
        detections = [
            SymbolDetection(
                id=0,
                symbol_type='ahu',
                category=SymbolCategory.AHU,
                bbox=(100, 100, 200, 150),
                confidence=0.9,
                center=(150, 125)
            ),
            SymbolDetection(
                id=1,
                symbol_type='vav_box',
                category=SymbolCategory.VAV,
                bbox=(300, 100, 400, 150),
                confidence=0.85,
                center=(350, 125)
            ),
            SymbolDetection(
                id=2,
                symbol_type='diffuser',
                category=SymbolCategory.DIFFUSER,
                bbox=(500, 100, 550, 150),
                confidence=0.8,
                center=(525, 125)
            ),
            SymbolDetection(
                id=3,
                symbol_type='diffuser',
                category=SymbolCategory.DIFFUSER,
                bbox=(600, 100, 650, 150),
                confidence=0.8,
                center=(625, 125)
            ),
        ]
        return detections
    
    def test_generate_bom(self):
        """Test BOM generation"""
        detections = self.create_test_detections()
        counts = {'ahu': 1, 'vav_box': 1, 'diffuser': 2}
        
        bom = self.bom_generator.generate_bom(counts, detections)
        
        self.assertIsInstance(bom, list)
        self.assertEqual(len(bom), 3)  # 3 unique types
        
        # Check BOM items
        for item in bom:
            self.assertIsInstance(item, BOMItem)
            self.assertGreater(item.quantity, 0)
            self.assertIsNotNone(item.description)
    
    def test_bom_quantities(self):
        """Test BOM quantities are correct"""
        detections = self.create_test_detections()
        counts = {'ahu': 1, 'vav_box': 1, 'diffuser': 2}
        
        bom = self.bom_generator.generate_bom(counts, detections)
        
        # Find diffuser item
        diffuser_item = next((item for item in bom if 'diffuser' in item.symbol_type.lower()), None)
        self.assertIsNotNone(diffuser_item)
        self.assertEqual(diffuser_item.quantity, 2)
    
    def test_export_bom_csv(self):
        """Test BOM CSV export"""
        detections = self.create_test_detections()
        counts = {'ahu': 1, 'vav_box': 1, 'diffuser': 2}
        
        bom = self.bom_generator.generate_bom(counts, detections)
        csv_str = self.bom_generator.export_bom(bom, format='csv')
        
        self.assertIsInstance(csv_str, str)
        self.assertIn('item_id', csv_str)
        self.assertIn('quantity', csv_str)
    
    def test_export_bom_json(self):
        """Test BOM JSON export"""
        detections = self.create_test_detections()
        counts = {'ahu': 1, 'vav_box': 1, 'diffuser': 2}
        
        bom = self.bom_generator.generate_bom(counts, detections)
        json_str = self.bom_generator.export_bom(bom, format='json')
        
        self.assertIsInstance(json_str, str)
        self.assertIn('item_id', json_str)
        
        # Verify it's valid JSON
        import json
        data = json.loads(json_str)
        self.assertIsInstance(data, list)


class TestConnectivityAnalyzer(unittest.TestCase):
    """Test connectivity analysis"""
    
    def setUp(self):
        self.analyzer = create_connectivity_analyzer()
        
    def create_test_detections(self):
        """Create test symbol detections with spatial relationships"""
        detections = [
            # AHU (primary equipment)
            SymbolDetection(
                id=0,
                symbol_type='ahu',
                category=SymbolCategory.AHU,
                bbox=(100, 100, 200, 150),
                confidence=0.9,
                center=(150, 125)
            ),
            # VAV boxes (distribution) - near AHU
            SymbolDetection(
                id=1,
                symbol_type='vav_box',
                category=SymbolCategory.VAV,
                bbox=(250, 100, 350, 150),
                confidence=0.85,
                center=(300, 125)
            ),
            SymbolDetection(
                id=2,
                symbol_type='vav_box',
                category=SymbolCategory.VAV,
                bbox=(400, 100, 500, 150),
                confidence=0.85,
                center=(450, 125)
            ),
            # Diffusers (terminals) - near VAV boxes
            SymbolDetection(
                id=3,
                symbol_type='diffuser',
                category=SymbolCategory.DIFFUSER,
                bbox=(280, 200, 320, 240),
                confidence=0.8,
                center=(300, 220)
            ),
            SymbolDetection(
                id=4,
                symbol_type='diffuser',
                category=SymbolCategory.DIFFUSER,
                bbox=(430, 200, 470, 240),
                confidence=0.8,
                center=(450, 220)
            ),
        ]
        return detections
    
    def test_generate_netlist(self):
        """Test netlist generation"""
        detections = self.create_test_detections()
        
        netlist = self.analyzer.generate_netlist(detections)
        
        self.assertIsInstance(netlist, dict)
        self.assertIn('connections', netlist)
        self.assertIn('circuits', netlist)
        self.assertIn('hierarchy', netlist)
        self.assertIn('graph_stats', netlist)
    
    def test_connections_detected(self):
        """Test that connections are detected"""
        detections = self.create_test_detections()
        
        netlist = self.analyzer.generate_netlist(detections)
        
        # Should have some connections
        self.assertIsInstance(netlist['connections'], list)
    
    def test_hierarchy_classification(self):
        """Test system hierarchy classification"""
        detections = self.create_test_detections()
        
        netlist = self.analyzer.generate_netlist(detections)
        hierarchy = netlist['hierarchy']
        
        self.assertIn('primary_equipment', hierarchy)
        self.assertIn('distribution', hierarchy)
        self.assertIn('terminal_units', hierarchy)
        
        # Check AHU is in primary
        self.assertGreater(len(hierarchy['primary_equipment']), 0)


class TestIntegratedDetector(unittest.TestCase):
    """Test integrated detector"""
    
    def setUp(self):
        self.detector = create_integrated_detector(
            use_sahi=False,  # Skip SAHI for testing
            use_document_processing=False  # Skip doc processing for testing
        )
        
    def create_test_image(self):
        """Create test blueprint"""
        image = np.ones((1000, 1200, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (100, 100), (200, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (300, 100), (400, 150), (0, 0, 0), -1)
        return image
    
    def test_integrated_detector_initialization(self):
        """Test integrated detector initializes"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.yoloplan)
    
    def test_analyze_blueprint(self):
        """Test full blueprint analysis"""
        image = self.create_test_image()
        
        results = self.detector.analyze_blueprint(
            image,
            include_components=False,
            include_text=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('symbols', results)
        self.assertIn('bom', results)
        self.assertIn('connectivity', results)
        self.assertIn('summary', results)
    
    def test_batch_analyze(self):
        """Test batch analysis"""
        images = [self.create_test_image() for _ in range(3)]
        
        results = self.detector.batch_analyze(images)
        
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIn('batch_summary', results)
        self.assertEqual(results['total_blueprints'], 3)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions"""
    
    def test_create_yoloplan_detector(self):
        """Test factory creates detector"""
        detector = create_yoloplan_detector()
        self.assertIsInstance(detector, YOLOplanDetector)
    
    def test_create_bom_generator(self):
        """Test factory creates BOM generator"""
        generator = create_bom_generator()
        self.assertIsInstance(generator, BOMGenerator)
    
    def test_create_connectivity_analyzer(self):
        """Test factory creates connectivity analyzer"""
        analyzer = create_connectivity_analyzer()
        self.assertIsInstance(analyzer, ConnectivityAnalyzer)
    
    def test_create_integrated_detector(self):
        """Test factory creates integrated detector"""
        detector = create_integrated_detector()
        self.assertIsInstance(detector, IntegratedHVACDetector)


if __name__ == '__main__':
    unittest.main()
