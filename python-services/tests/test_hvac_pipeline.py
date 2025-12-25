"""
Unit tests for HVAC Drawing Analysis Pipeline.
Tests all three stages and error handling.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from core.ai.pipeline_models import (
    DetectionBox, DetectionResult, TextRecognitionResult,
    HVACInterpretation, HVACInterpretationResult, HVACResult,
    PipelineConfig, PipelineStage, HVACEquipmentType
)
from core.ai.hvac_pipeline import HVACDrawingAnalyzer


class TestPipelineModels:
    """Test data models."""
    
    def test_detection_box_properties(self):
        """Test DetectionBox calculated properties."""
        box = DetectionBox(
            x1=100.0, y1=150.0, x2=200.0, y2=250.0,
            confidence=0.95, class_id=3, class_name="valve"
        )
        
        assert box.width == 100.0
        assert box.height == 100.0
        assert box.center == (150.0, 200.0)
        assert box.area == 10000.0
    
    def test_detection_result_validation(self):
        """Test DetectionResult validation."""
        result = DetectionResult(
            detections=[],
            text_regions=[],
            processing_time_ms=9.5,
            image_width=1920,
            image_height=1080,
            model_version="yolo11m-obb"
        )
        
        assert result.image_width == 1920
        assert result.image_height == 1080
        assert result.model_version == "yolo11m-obb"
    
    def test_hvac_result_success_property(self):
        """Test HVACResult success property."""
        result = HVACResult(
            request_id="test_123",
            stage=PipelineStage.COMPLETE,
            total_processing_time_ms=18.5
        )
        
        assert result.success is True
        
        # Add critical error
        from core.ai.pipeline_models import PipelineError, ErrorSeverity
        result.errors.append(
            PipelineError(
                stage=PipelineStage.DETECTION,
                severity=ErrorSeverity.CRITICAL,
                message="Test error"
            )
        )
        
        assert result.success is False
    
    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()
        
        assert config.confidence_threshold == 0.7
        assert config.ocr_min_size == 8
        assert config.max_processing_time_ms == 25.0
        assert config.enable_gpu is True


class TestHVACDrawingAnalyzer:
    """Test HVAC Drawing Analyzer pipeline."""
    
    @pytest.fixture
    def mock_yolo_model(self):
        """Create mock YOLO model."""
        mock_model = Mock()
        mock_model.names = {
            0: 'id_letters',
            1: 'tag_number',
            2: 'valve',
            3: 'damper'
        }
        return mock_model
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_path = tmp_file.name
            Image.fromarray(img).save(tmp_path)
        
        yield tmp_path
        
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    @pytest.fixture
    def analyzer_config(self):
        """Create test configuration."""
        return PipelineConfig(
            confidence_threshold=0.7,
            max_processing_time_ms=50.0,
            enable_gpu=False  # Disable GPU for tests
        )
    
    @patch('core.ai.hvac_pipeline.YOLOInferenceEngine')
    @patch('core.ai.hvac_pipeline.easyocr')
    def test_analyzer_initialization(self, mock_easyocr, mock_yolo, analyzer_config):
        """Test analyzer initialization."""
        # Setup mocks
        mock_yolo_instance = Mock()
        mock_yolo.return_value = mock_yolo_instance
        
        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader
        
        # Create analyzer
        analyzer = HVACDrawingAnalyzer(
            yolo_model_path="/fake/model.pt",
            config=analyzer_config,
            device='cpu'
        )
        
        assert analyzer.yolo_engine is not None
        assert analyzer.device == 'cpu'
        assert analyzer.config.confidence_threshold == 0.7
    
    @patch('core.ai.hvac_pipeline.YOLOInferenceEngine')
    @patch('core.ai.hvac_pipeline.easyocr', None)  # Simulate EasyOCR not available
    def test_analyzer_without_ocr(self, mock_yolo, analyzer_config):
        """Test analyzer works without EasyOCR."""
        mock_yolo_instance = Mock()
        mock_yolo.return_value = mock_yolo_instance
        
        analyzer = HVACDrawingAnalyzer(
            yolo_model_path="/fake/model.pt",
            config=analyzer_config,
            device='cpu'
        )
        
        assert analyzer.ocr_reader is None
    
    def test_load_image(self, sample_image, analyzer_config):
        """Test image loading."""
        with patch('core.ai.hvac_pipeline.YOLOInferenceEngine'), \
             patch('core.ai.hvac_pipeline.easyocr'):
            
            analyzer = HVACDrawingAnalyzer(
                yolo_model_path="/fake/model.pt",
                config=analyzer_config,
                device='cpu'
            )
            
            image = analyzer._load_image(sample_image)
            
            assert image is not None
            assert image.shape[2] == 3  # RGB
            assert image.dtype == np.uint8
    
    def test_stage1_detection(self, analyzer_config):
        """Test Stage 1 detection."""
        with patch('core.ai.hvac_pipeline.YOLOInferenceEngine') as mock_yolo, \
             patch('core.ai.hvac_pipeline.easyocr'):
            
            # Setup mock YOLO results
            mock_engine = Mock()
            mock_engine.predict.return_value = {
                'segments': [
                    {
                        'bbox': [100, 150, 200, 250],
                        'score': 0.95,
                        'label': 'valve',
                        'class_id': 2
                    },
                    {
                        'bbox': [205, 155, 280, 175],
                        'score': 0.87,
                        'label': 'id_letters',
                        'class_id': 0
                    }
                ],
                'total_objects_found': 2
            }
            mock_yolo.return_value = mock_engine
            
            analyzer = HVACDrawingAnalyzer(
                yolo_model_path="/fake/model.pt",
                config=analyzer_config,
                device='cpu'
            )
            
            # Create test image
            test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            
            # Run detection
            result = analyzer._stage1_detection(test_image, "test_req")
            
            assert isinstance(result, DetectionResult)
            assert len(result.detections) == 2
            assert len(result.text_regions) == 1
            assert result.text_regions[0].class_name == 'id_letters'
            assert result.image_width == 480
            assert result.image_height == 640
    
    def test_stage3_interpretation_vav(self, analyzer_config):
        """Test Stage 3 interpretation for VAV pattern."""
        with patch('core.ai.hvac_pipeline.YOLOInferenceEngine'), \
             patch('core.ai.hvac_pipeline.easyocr'):
            
            analyzer = HVACDrawingAnalyzer(
                yolo_model_path="/fake/model.pt",
                config=analyzer_config,
                device='cpu'
            )
            
            # Create mock text result
            text_result = TextRecognitionResult(
                region=DetectionBox(
                    x1=205, y1=155, x2=280, y2=175,
                    confidence=0.87, class_id=0, class_name='id_letters'
                ),
                text="VAV-101",
                confidence=0.92,
                preprocessing_metadata={}
            )
            
            # Create mock detections
            detections = [
                DetectionBox(
                    x1=200, y1=150, x2=300, y2=250,
                    confidence=0.95, class_id=2, class_name='valve'
                )
            ]
            
            # Run interpretation
            interpretation = analyzer._interpret_text(
                text_result,
                detections,
                image_width=640,
                image_height=480
            )
            
            assert interpretation is not None
            assert interpretation.equipment_type == HVACEquipmentType.VAV
            assert interpretation.system_id == "VAV"
            assert interpretation.zone_number == "101"
            assert "VAV" in interpretation.pattern_matched
    
    def test_stage3_interpretation_ahu(self, analyzer_config):
        """Test Stage 3 interpretation for AHU pattern."""
        with patch('core.ai.hvac_pipeline.YOLOInferenceEngine'), \
             patch('core.ai.hvac_pipeline.easyocr'):
            
            analyzer = HVACDrawingAnalyzer(
                yolo_model_path="/fake/model.pt",
                config=analyzer_config,
                device='cpu'
            )
            
            text_result = TextRecognitionResult(
                region=DetectionBox(
                    x1=100, y1=100, x2=150, y2=120,
                    confidence=0.9, class_id=0, class_name='id_letters'
                ),
                text="AHU-5",
                confidence=0.95,
                preprocessing_metadata={}
            )
            
            interpretation = analyzer._interpret_text(
                text_result, [], 640, 480
            )
            
            assert interpretation.equipment_type == HVACEquipmentType.AHU
            assert interpretation.system_id == "AHU"
            assert interpretation.zone_number == "5"
    
    def test_find_associated_component(self, analyzer_config):
        """Test spatial component association."""
        with patch('core.ai.hvac_pipeline.YOLOInferenceEngine'), \
             patch('core.ai.hvac_pipeline.easyocr'):
            
            analyzer = HVACDrawingAnalyzer(
                yolo_model_path="/fake/model.pt",
                config=analyzer_config,
                device='cpu'
            )
            
            # Text region
            text_region = DetectionBox(
                x1=205, y1=155, x2=280, y2=175,
                confidence=0.87, class_id=0, class_name='id_letters'
            )
            
            # Nearby component
            nearby_component = DetectionBox(
                x1=200, y1=150, x2=300, y2=250,
                confidence=0.95, class_id=2, class_name='valve'
            )
            
            # Far component
            far_component = DetectionBox(
                x1=500, y1=400, x2=600, y2=500,
                confidence=0.90, class_id=3, class_name='damper'
            )
            
            detections = [nearby_component, far_component]
            
            # Find association
            associated = analyzer._find_associated_component(
                text_region,
                detections,
                image_width=640,
                image_height=480
            )
            
            assert associated is not None
            assert associated.class_name == 'valve'
    
    def test_health_check(self, analyzer_config):
        """Test health check functionality."""
        with patch('core.ai.hvac_pipeline.YOLOInferenceEngine'), \
             patch('core.ai.hvac_pipeline.easyocr'):
            
            analyzer = HVACDrawingAnalyzer(
                yolo_model_path="/fake/model.pt",
                config=analyzer_config,
                device='cpu'
            )
            
            health = analyzer.health_check()
            
            assert health['status'] == 'healthy'
            assert 'yolo_loaded' in health
            assert 'ocr_loaded' in health
            assert 'device' in health
            assert health['device'] == 'cpu'
    
    def test_statistics(self, analyzer_config):
        """Test statistics tracking."""
        with patch('core.ai.hvac_pipeline.YOLOInferenceEngine'), \
             patch('core.ai.hvac_pipeline.easyocr'):
            
            analyzer = HVACDrawingAnalyzer(
                yolo_model_path="/fake/model.pt",
                config=analyzer_config,
                device='cpu'
            )
            
            stats = analyzer.get_statistics()
            
            assert 'total_requests' in stats
            assert 'average_processing_time_ms' in stats
            assert 'device' in stats
            assert 'models_loaded' in stats


class TestPipelineIntegration:
    """Integration tests for full pipeline."""
    
    @pytest.fixture
    def sample_hvac_image(self):
        """Create a sample HVAC drawing for testing."""
        # Create a more realistic test image with text-like regions
        img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        
        # Draw some boxes to simulate components
        cv2_available = True
        try:
            import cv2
        except ImportError:
            cv2_available = False
        
        if cv2_available:
            import cv2
            # Valve box
            cv2.rectangle(img, (200, 150), (300, 250), (100, 100, 100), -1)
            # Text region (simulated)
            cv2.rectangle(img, (305, 155), (380, 175), (50, 50, 50), -1)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_path = tmp_file.name
            Image.fromarray(img).save(tmp_path)
        
        yield tmp_path
        
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    @patch('core.ai.hvac_pipeline.YOLOInferenceEngine')
    @patch('core.ai.hvac_pipeline.easyocr')
    def test_full_pipeline_success(self, mock_easyocr, mock_yolo, sample_hvac_image):
        """Test successful end-to-end pipeline execution."""
        # Setup YOLO mock
        mock_yolo_instance = Mock()
        mock_yolo_instance.predict.return_value = {
            'segments': [
                {
                    'bbox': [200, 150, 300, 250],
                    'score': 0.95,
                    'label': 'valve',
                    'class_id': 2
                },
                {
                    'bbox': [305, 155, 380, 175],
                    'score': 0.87,
                    'label': 'id_letters',
                    'class_id': 0
                }
            ],
            'total_objects_found': 2
        }
        mock_yolo.return_value = mock_yolo_instance
        
        # Setup OCR mock
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [75, 0], [75, 20], [0, 20]], "VAV-101", 0.92)
        ]
        mock_easyocr.Reader.return_value = mock_reader
        
        # Create analyzer
        config = PipelineConfig(
            confidence_threshold=0.7,
            enable_gpu=False
        )
        
        analyzer = HVACDrawingAnalyzer(
            yolo_model_path="/fake/model.pt",
            config=config,
            device='cpu'
        )
        
        # Run full pipeline
        result = analyzer.analyze_drawing(sample_hvac_image)
        
        # Verify result structure
        assert isinstance(result, HVACResult)
        assert result.stage == PipelineStage.COMPLETE
        assert result.detection_result is not None
        assert len(result.detection_result.detections) == 2
        assert len(result.detection_result.text_regions) == 1
        assert len(result.text_results) > 0
        assert result.interpretation_result is not None
        
        # Check timing
        assert result.total_processing_time_ms > 0
        assert 'detection' in result.stage_timings
        assert 'text_recognition' in result.stage_timings
        assert 'interpretation' in result.stage_timings


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
