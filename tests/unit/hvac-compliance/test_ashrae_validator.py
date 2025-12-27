"""
Unit tests for ASHRAE 62.1 Ventilation Validator
"""

import pytest
import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from services.hvac_domain.compliance.ashrae_62_1_standards import (
    ASHRAE621Validator,
    VentilationZone,
    OccupancyType
)


class TestASHRAE621Validator:
    """Tests for ASHRAE 62.1 compliance validator"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return ASHRAE621Validator()
    
    @pytest.fixture
    def office_zone(self):
        """Create sample office zone"""
        return VentilationZone(
            zone_id="zone_001",
            occupancy_type=OccupancyType.OFFICE,
            floor_area=2000.0,  # sq ft
            design_airflow=500.0,  # CFM
            outdoor_air_flow=200.0  # CFM
        )
    
    def test_calculate_minimum_outdoor_air_office(self, validator, office_zone):
        """Test minimum outdoor air calculation for office space"""
        calculation = validator.calculate_minimum_outdoor_air(office_zone)
        
        assert "minimum_outdoor_air_cfm" in calculation
        assert calculation["occupancy_type"] == "office"
        assert calculation["floor_area"] == 2000.0
        
        # For office: Rp=5 CFM/person, Ra=0.06 CFM/sqft, density=5 people/1000sqft
        # Population = 2000 * 5 / 1000 = 10 people
        # Min OA = (5 * 10) + (0.06 * 2000) = 50 + 120 = 170 CFM
        assert abs(calculation["minimum_outdoor_air_cfm"] - 170.0) < 1.0
    
    def test_validate_zone_compliant(self, validator, office_zone):
        """Test validation of compliant zone"""
        validation = validator.validate_zone_ventilation(office_zone)
        
        assert validation["is_compliant"] is True
        assert len(validation["violations"]) == 0
        assert validation["summary"]["compliance_status"] == "PASS"
    
    def test_validate_zone_non_compliant(self, validator):
        """Test validation of non-compliant zone"""
        # Zone with insufficient outdoor air
        zone = VentilationZone(
            zone_id="zone_002",
            occupancy_type=OccupancyType.OFFICE,
            floor_area=2000.0,
            design_airflow=500.0,
            outdoor_air_flow=100.0  # Too low (should be ~170 CFM)
        )
        
        validation = validator.validate_zone_ventilation(zone)
        
        assert validation["is_compliant"] is False
        assert len(validation["violations"]) > 0
        
        # Check for critical or warning violation
        violation = validation["violations"][0]
        assert violation["severity"] in ["CRITICAL", "WARNING"]
        assert "outdoor air flow" in violation["description"].lower()
    
    def test_validate_zone_no_oa_specified(self, validator):
        """Test validation when outdoor air is not specified"""
        zone = VentilationZone(
            zone_id="zone_003",
            occupancy_type=OccupancyType.CLASSROOM,
            floor_area=1000.0,
            design_airflow=800.0,
            outdoor_air_flow=None  # Not specified
        )
        
        validation = validator.validate_zone_ventilation(zone)
        
        # Should have warning about unverified OA
        assert len(validation["violations"]) > 0
        assert validation["violations"][0]["severity"] == "WARNING"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
