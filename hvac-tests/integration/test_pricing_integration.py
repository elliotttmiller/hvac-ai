"""
Integration test for Pricing Engine integration with Inference Graph
Tests that the AI service can import and use the Pricing Engine from hvac-domain
"""

import pytest
import sys
from pathlib import Path

# Add services to path
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_ROOT = REPO_ROOT / "services"
sys.path.insert(0, str(SERVICES_ROOT))
sys.path.insert(0, str(SERVICES_ROOT / "hvac-ai"))
sys.path.insert(0, str(SERVICES_ROOT / "hvac-domain"))


def test_pricing_engine_import():
    """Test that PricingEngine can be imported from hvac-domain"""
    try:
        from pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData
        assert PricingEngine is not None
        assert QuoteRequest is not None
        assert AnalysisData is not None
    except ImportError as e:
        pytest.fail(f"Failed to import PricingEngine: {e}")


def test_pricing_engine_initialization():
    """Test that PricingEngine can be initialized with catalog"""
    from pricing.pricing_service import PricingEngine
    
    catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
    assert catalog_path.exists(), f"Catalog not found at {catalog_path}"
    
    engine = PricingEngine(catalog_path=catalog_path)
    assert engine is not None
    assert engine.catalog is not None
    assert 'components' in engine.catalog


def test_quote_generation_from_detections():
    """Test that a quote can be generated from detection counts"""
    from pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData
    
    catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
    engine = PricingEngine(catalog_path=catalog_path)
    
    # Simulate detection results
    analysis_data = AnalysisData(
        total_objects=10,
        counts_by_category={
            "vav_box": 3,
            "thermostat": 5,
            "diffuser_square": 2
        }
    )
    
    request = QuoteRequest(
        project_id="TEST-001",
        location="Chicago, IL",
        analysis_data=analysis_data
    )
    
    quote = engine.generate_quote(request)
    
    # Verify quote structure
    assert quote is not None
    assert quote.quote_id == "Q-TEST-001"
    assert quote.currency == "USD"
    assert len(quote.line_items) == 3
    assert quote.summary.total_cost > 0
    assert quote.summary.final_price > quote.summary.total_cost  # includes margin


def test_label_normalization():
    """Test that detection labels are properly normalized for catalog lookup"""
    from pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData
    
    catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
    engine = PricingEngine(catalog_path=catalog_path)
    
    # Test with various label formats
    analysis_data = AnalysisData(
        total_objects=3,
        counts_by_category={
            "vav_box": 1,       # exact match
            "VAV_Box": 1,       # case variation (should be normalized to lowercase)
            "vav box": 1        # space variation (should be normalized to underscore)
        }
    )
    
    request = QuoteRequest(
        project_id="TEST-002",
        location="default",
        analysis_data=analysis_data
    )
    
    quote = engine.generate_quote(request)
    
    # Should handle all variations
    assert len(quote.line_items) == 3


def test_fallback_pricing_for_unknown_components():
    """Test that unknown components get default pricing"""
    from pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData
    
    catalog_path = SERVICES_ROOT / "hvac-domain" / "pricing" / "catalog.json"
    engine = PricingEngine(catalog_path=catalog_path)
    
    # Use an unknown component
    analysis_data = AnalysisData(
        total_objects=1,
        counts_by_category={
            "unknown_component_xyz": 1
        }
    )
    
    request = QuoteRequest(
        project_id="TEST-003",
        location="default",
        analysis_data=analysis_data
    )
    
    quote = engine.generate_quote(request)
    
    # Should generate quote with default pricing
    assert len(quote.line_items) == 1
    assert quote.summary.total_cost > 0
    # Check that the line item uses the default rate
    line_item = quote.line_items[0]
    assert line_item.unit_material_cost == 100.00  # default from catalog


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
