"""
Integration tests for HVAC Compliance Analyzer
"""

import pytest
import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from services.hvac_domain.hvac_compliance_analyzer import (
    HVACComplianceAnalyzer,
    ComplianceAnalysisRequest
)


class TestComplianceIntegration:
    """Integration tests for compliance analysis"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return HVACComplianceAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes all components"""
        assert analyzer.ashrae_validator is not None
        assert analyzer.smacna_validator is not None
        assert analyzer.imc_validator is not None
        assert analyzer.ductwork_validator is not None
        assert analyzer.equipment_validator is not None
        assert analyzer.confidence_scorer is not None
        assert analyzer.regional_manager is not None
        assert analyzer.graph_builder is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
