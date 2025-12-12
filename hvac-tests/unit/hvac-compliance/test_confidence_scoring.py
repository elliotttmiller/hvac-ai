"""
Unit tests for Confidence Scoring System
"""

import pytest
import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from services.hvac_domain.compliance.confidence_scoring import (
    ConfidenceScorer,
    ViolationSeverity,
    ConfidenceLevel
)


class TestConfidenceScorer:
    """Tests for confidence scoring system"""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer instance"""
        return ConfidenceScorer()
    
    @pytest.fixture
    def sample_violations(self):
        """Create sample violations"""
        return [
            {
                "severity": "CRITICAL",
                "confidence": 0.95,
                "cost_impact": 5000.0,
                "description": "Missing fire damper"
            },
            {
                "severity": "WARNING",
                "confidence": 0.80,
                "cost_impact": 1500.0,
                "description": "Undersized duct"
            },
            {
                "severity": "INFO",
                "confidence": 0.70,
                "cost_impact": 500.0,
                "description": "Optimization suggestion"
            }
        ]
    
    def test_calculate_risk_score_critical(self, scorer):
        """Test risk score calculation for critical violation"""
        violation = {
            "severity": "CRITICAL",
            "confidence": 0.95,
            "cost_impact": 5000.0
        }
        
        risk_score = scorer.calculate_risk_score(violation)
        
        assert risk_score.severity == ViolationSeverity.CRITICAL
        assert risk_score.confidence == 0.95
        assert risk_score.priority == 1  # Highest priority
        assert risk_score.risk_score > 0
    
    def test_calculate_risk_score_warning(self, scorer):
        """Test risk score calculation for warning"""
        violation = {
            "severity": "WARNING",
            "confidence": 0.75,
            "cost_impact": 1000.0
        }
        
        risk_score = scorer.calculate_risk_score(violation)
        
        assert risk_score.severity == ViolationSeverity.WARNING
        assert risk_score.priority in [2, 3, 4]  # Lower priority than critical
    
    def test_classify_confidence_level(self, scorer):
        """Test confidence level classification"""
        assert scorer.classify_confidence_level(0.95) == ConfidenceLevel.HIGH
        assert scorer.classify_confidence_level(0.75) == ConfidenceLevel.MEDIUM
        assert scorer.classify_confidence_level(0.50) == ConfidenceLevel.LOW
    
    def test_enrich_violations(self, scorer, sample_violations):
        """Test violation enrichment"""
        enriched = scorer.enrich_violations(sample_violations)
        
        assert len(enriched) == 3
        
        # Check enrichment fields added
        for violation in enriched:
            assert "risk_score" in violation
            assert "priority" in violation
            assert "confidence_level" in violation
        
        # Check sorting by priority and risk score
        assert enriched[0]["priority"] <= enriched[1]["priority"]
    
    def test_calculate_compliance_score(self, scorer, sample_violations):
        """Test overall compliance score calculation"""
        score = scorer.calculate_compliance_score(
            sample_violations,
            total_components=10
        )
        
        assert 0.0 <= score <= 100.0
        # With violations, score should be less than 100
        assert score < 100.0
    
    def test_generate_compliance_summary(self, scorer, sample_violations):
        """Test compliance summary generation"""
        summary = scorer.generate_compliance_summary(
            sample_violations,
            total_components=10
        )
        
        assert "compliance_score" in summary
        assert "total_violations" in summary
        assert "critical_violations" in summary
        assert "warning_violations" in summary
        assert "info_violations" in summary
        assert "compliance_grade" in summary
        
        assert summary["total_violations"] == 3
        assert summary["critical_violations"] == 1
        assert summary["warning_violations"] == 1
        assert summary["info_violations"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
