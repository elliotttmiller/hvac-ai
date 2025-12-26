"""
Unit tests for PricingEngine
"""

import pytest
import json
from pathlib import Path
from core.pricing.pricing_service import (
    PricingEngine,
    QuoteRequest,
    AnalysisData,
    QuoteSettings
)


def test_pricing_engine_initialization():
    """Test that pricing engine loads catalog successfully"""
    engine = PricingEngine()
    assert engine.catalog is not None
    assert 'components' in engine.catalog
    assert 'regional_multipliers' in engine.catalog
    assert len(engine.catalog['components']) > 0


def test_regional_multipliers():
    """Test regional multiplier logic"""
    engine = PricingEngine()
    
    # Test exact match
    chicago = engine._get_regional_multipliers("Chicago, IL")
    assert chicago['labor'] == 1.15
    assert chicago['material'] == 1.05
    
    # Test partial match
    chicago_partial = engine._get_regional_multipliers("Chicago")
    assert chicago_partial['labor'] == 1.15
    
    # Test default fallback
    unknown = engine._get_regional_multipliers("Unknown City")
    assert unknown['labor'] == 1.0
    assert unknown['material'] == 1.0


def test_component_pricing_lookup():
    """Test component pricing lookup"""
    engine = PricingEngine()
    
    # Test exact match
    vav = engine._get_component_pricing("vav_box")
    assert vav['material_cost'] == 800.00
    assert vav['labor_hours'] == 4.0
    assert vav['sku_name'] == "VAV Box with Controls"
    
    # Test unknown component (should use default)
    unknown = engine._get_component_pricing("unknown_component")
    assert 'material_cost' in unknown
    assert unknown['material_cost'] == 100.00  # default


def test_quote_generation_simple():
    """Test basic quote generation"""
    engine = PricingEngine()
    
    request = QuoteRequest(
        project_id="TEST-001",
        location="Chicago, IL",
        analysis_data=AnalysisData(
            total_objects=15,
            counts_by_category={
                "vav_box": 5,
                "thermostat": 10
            }
        )
    )
    
    quote = engine.generate_quote(request)
    
    # Check structure
    assert quote.quote_id == "Q-TEST-001"
    assert quote.currency == "USD"
    assert len(quote.line_items) == 2
    
    # Check summary totals are positive
    assert quote.summary.subtotal_materials > 0
    assert quote.summary.subtotal_labor > 0
    assert quote.summary.total_cost > 0
    assert quote.summary.final_price > quote.summary.total_cost  # includes margin


def test_quote_generation_with_settings():
    """Test quote generation with custom settings"""
    engine = PricingEngine()
    
    request = QuoteRequest(
        project_id="TEST-002",
        location="New York, NY",
        analysis_data=AnalysisData(
            total_objects=10,
            counts_by_category={
                "diffuser_square": 10
            }
        ),
        settings=QuoteSettings(
            margin_percent=25.0,
            labor_hourly_rate=95.0
        )
    )
    
    quote = engine.generate_quote(request)
    
    assert quote.quote_id == "Q-TEST-002"
    assert len(quote.line_items) == 1
    
    # Check that NY multipliers are applied (higher costs)
    line_item = quote.line_items[0]
    assert line_item.category == "diffuser_square"
    assert line_item.count == 10
    
    # Material cost should be higher than base (1.10 multiplier for NY)
    base_material = 45.00
    assert line_item.unit_material_cost > base_material


def test_quote_calculation_accuracy():
    """Test that calculations are accurate to 2 decimal places"""
    engine = PricingEngine()
    
    request = QuoteRequest(
        project_id="TEST-003",
        location="default",
        analysis_data=AnalysisData(
            total_objects=3,
            counts_by_category={
                "thermostat": 3
            }
        ),
        settings=QuoteSettings(
            margin_percent=20.0,
            labor_hourly_rate=85.0
        )
    )
    
    quote = engine.generate_quote(request)
    
    # Manual calculation:
    # thermostat: material_cost=125, labor_hours=2.0
    # Unit cost: 125 + (2.0 * 85) = 125 + 170 = 295
    # Total for 3: 295 * 3 = 885
    # With 20% margin: 885 * 1.20 = 1062.00
    
    assert quote.summary.total_cost == 885.00
    assert quote.summary.final_price == 1062.00


def test_empty_counts():
    """Test quote generation with empty counts"""
    engine = PricingEngine()
    
    request = QuoteRequest(
        project_id="TEST-004",
        location="Chicago, IL",
        analysis_data=AnalysisData(
            total_objects=0,
            counts_by_category={}
        )
    )
    
    quote = engine.generate_quote(request)
    
    assert len(quote.line_items) == 0
    assert quote.summary.subtotal_materials == 0.0
    assert quote.summary.subtotal_labor == 0.0
    assert quote.summary.total_cost == 0.0
    assert quote.summary.final_price == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
