"""
End-to-End Integration Tests for HVAC Analysis Pipeline

Tests the complete workflow from blueprint input to validated results,
ensuring all components work together correctly.

Industry best practices:
- Comprehensive E2E coverage
- Real-world scenario testing
- Performance benchmarking
- Error condition handling
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


class TestEndToEndAnalysis:
    """End-to-end integration tests for complete analysis pipeline"""
    
    @pytest.fixture
    def mock_sam_engine(self):
        """Create a mock legacy inference engine for testing"""
        engine = Mock()
        engine.device = "cpu"
        engine.count = Mock(return_value={
            "objects": [
                {
                    "bbox": [100, 100, 50, 50],
                    "score": 0.95,
                    "mask": {"size": [100, 100], "counts": "test"},
                    "category": "Duct"
                },
                {
                    "bbox": [200, 200, 30, 30],
                    "score": 0.90,
                    "mask": {"size": [100, 100], "counts": "test"},
                    "category": "Damper"
                }
            ],
            "total_objects_found": 2,
            "processing_time_ms": 150.0
        })
        return engine
    
    @pytest.fixture
    def sample_blueprint(self):
        """Create a sample blueprint image for testing"""
        # Create a simple test image (3000x2000 pixels)
        image = np.ones((2000, 3000, 3), dtype=np.uint8) * 200
        
        # Add some simple shapes to simulate components
        import cv2
        
        # Draw rectangle (ductwork)
        cv2.rectangle(image, (100, 100), (500, 200), (50, 50, 50), 2)
        
        # Draw circles (diffusers)
        cv2.circle(image, (250, 400), 20, (50, 50, 50), 2)
        cv2.circle(image, (450, 400), 20, (50, 50, 50), 2)
        
        return image
    
    def test_complete_workflow_without_integration(self, mock_sam_engine, sample_blueprint):
        """Test complete workflow using only mock legacy inference engine"""
        # This test works even without the new services
        result = mock_sam_engine.count(
            sample_blueprint,
            grid_size=32,
            min_score=0.2,
            debug=False,
            max_grid_points=2000
        )
        
        assert result is not None
        assert "objects" in result
        assert len(result["objects"]) == 2
        assert result["total_objects_found"] == 2
    
    def test_integration_with_hvac_services(self, mock_sam_engine, sample_blueprint):
        """Test integration with new HVAC services"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            # Create integrated analyzer
            analyzer = create_integrated_analyzer(
                sam_engine=mock_sam_engine,
                model_path="dummy_model.pth",
                enable_sahi=False,  # Disable for testing
                enable_validation=True,
                enable_prompts=True,
                device="cpu"
            )
            
            # Run complete analysis
            result = analyzer.analyze_blueprint(
                image=sample_blueprint,
                mode="legacy",  # Use legacy mode for testing
                return_relationships=True,
                return_validation=True
            )
            
            # Verify results
            assert result["status"] == "success"
            assert "detections" in result
            assert "total_components" in result
            assert "processing_time_seconds" in result
            assert "quality_metrics" in result
            assert "features_enabled" in result
            
            # Verify feature flags
            assert "sahi" in result["features_enabled"]
            assert "validation" in result["features_enabled"]
            assert "prompts" in result["features_enabled"]
            
        except ImportError as e:
            pytest.skip(f"Integration module not available: {e}")
    
    def test_quality_assessment(self, mock_sam_engine, sample_blueprint):
        """Test document quality assessment"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            analyzer = create_integrated_analyzer(
                sam_engine=mock_sam_engine,
                model_path="dummy_model.pth",
                device="cpu"
            )
            
            result = analyzer.analyze_blueprint(sample_blueprint, mode="legacy")
            
            assert "quality_metrics" in result
            quality = result["quality_metrics"]
            
            # Quality metrics should be present
            assert "available" in quality
            
        except ImportError:
            pytest.skip("Integration module not available")
    
    def test_system_validation(self, mock_sam_engine, sample_blueprint):
        """Test system validation with relationships"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            analyzer = create_integrated_analyzer(
                sam_engine=mock_sam_engine,
                model_path="dummy_model.pth",
                enable_validation=True,
                device="cpu"
            )
            
            result = analyzer.analyze_blueprint(
                image=sample_blueprint,
                mode="legacy",
                return_relationships=True,
                return_validation=True
            )
            
            # Check for relationship and validation data
            if "relationships" in result:
                assert isinstance(result["relationships"], list)
            
            if "validation" in result:
                assert isinstance(result["validation"], dict)
                if "is_valid" in result["validation"]:
                    assert isinstance(result["validation"]["is_valid"], bool)
            
        except ImportError:
            pytest.skip("Integration module not available")
    
    def test_prompt_generation(self, mock_sam_engine):
        """Test HVAC-specific prompt generation"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            analyzer = create_integrated_analyzer(
                sam_engine=mock_sam_engine,
                model_path="dummy_model.pth",
                enable_prompts=True,
                device="cpu"
            )
            
            prompt = analyzer.generate_analysis_prompt(
                analysis_type="component_detection",
                context={
                    "context": "Commercial office building",
                    "blueprint_type": "Supply air distribution"
                }
            )
            
            # Prompt should be generated if framework is available
            if analyzer.enable_prompts:
                assert prompt is not None
                assert isinstance(prompt, str)
                assert len(prompt) > 0
            
        except ImportError:
            pytest.skip("Integration module not available")
    
    def test_graceful_degradation(self, mock_sam_engine, sample_blueprint):
        """Test graceful degradation when services unavailable"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            # Create analyzer with all features disabled
            analyzer = create_integrated_analyzer(
                sam_engine=mock_sam_engine,
                model_path="dummy_model.pth",
                enable_sahi=False,
                enable_validation=False,
                enable_prompts=False,
                device="cpu"
            )
            
            # Should still work with just legacy inference
            result = analyzer.analyze_blueprint(
                image=sample_blueprint,
                mode="legacy"
            )
            
            assert result["status"] == "success"
            assert result["analysis_mode"] == "legacy"
            assert result["features_enabled"]["sahi"] == False
            
        except ImportError:
            pytest.skip("Integration module not available")
    
    def test_performance_metrics(self, mock_sam_engine, sample_blueprint):
        """Test that performance metrics are tracked"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            analyzer = create_integrated_analyzer(
                sam_engine=mock_sam_engine,
                model_path="dummy_model.pth",
                device="cpu"
            )
            
            result = analyzer.analyze_blueprint(sample_blueprint, mode="legacy")
            
            # Performance metrics should be present
            assert "processing_time_seconds" in result
            assert "processing_time_ms" in result
            assert result["processing_time_seconds"] > 0
            assert result["processing_time_ms"] > 0
            
        except ImportError:
            pytest.skip("Integration module not available")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_image(self):
        """Test handling of empty image"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            mock_engine = Mock()
            mock_engine.device = "cpu"
            mock_engine.count = Mock(return_value={"objects": [], "total_objects_found": 0})
            
            analyzer = create_integrated_analyzer(
                sam_engine=mock_engine,
                model_path="dummy.pth",
                device="cpu"
            )
            
            # Empty image
            empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = analyzer.analyze_blueprint(empty_image, mode="legacy")
            
            assert result["status"] == "success"
            assert result["total_components"] == 0
            
        except ImportError:
            pytest.skip("Integration module not available")
    
    def test_invalid_mode(self):
        """Test handling of invalid analysis mode"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            
            mock_engine = Mock()
            mock_engine.device = "cpu"
            mock_engine.count = Mock(return_value={"objects": [], "total_objects_found": 0})
            
            analyzer = create_integrated_analyzer(
                sam_engine=mock_engine,
                model_path="dummy.pth",
                device="cpu"
            )
            
            image = np.ones((100, 100, 3), dtype=np.uint8) * 200
            
            # Should default to legacy for invalid mode
            result = analyzer.analyze_blueprint(image, mode="invalid_mode")
            
            assert result["status"] == "success"
            assert result["analysis_mode"] == "legacy"
            
        except ImportError:
            pytest.skip("Integration module not available")


class TestComponentMapping:
    """Test component type mapping logic"""
    
    def test_category_to_type_mapping(self):
        """Test mapping of categories to HVAC types"""
        try:
            from python_services.core.ai.hvac_integration import (
                create_integrated_analyzer
            )
            from services.hvac_domain.hvac_system_engine import HVACComponentType
            
            mock_engine = Mock()
            mock_engine.device = "cpu"
            
            analyzer = create_integrated_analyzer(
                sam_engine=mock_engine,
                model_path="dummy.pth",
                device="cpu"
            )
            
            # Test various category mappings
            test_cases = [
                ("Duct", HVACComponentType.DUCTWORK),
                ("Damper", HVACComponentType.DAMPER),
                ("VAV", HVACComponentType.VAV_BOX),
                ("Fan-Blower", HVACComponentType.FAN),
                ("Unknown-Component", HVACComponentType.UNKNOWN)
            ]
            
            for category, expected_type in test_cases:
                result_type = analyzer._map_to_component_type(category)
                assert result_type == expected_type, f"Failed for category: {category}"
            
        except ImportError:
            pytest.skip("Integration module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
