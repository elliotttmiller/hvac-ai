"""
Location Intelligence Module
Regional building codes, climate zones, and location-specific adjustments
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ClimateZone(str, Enum):
    """ASHRAE Climate Zones"""
    ZONE_1A = "1A"  # Very Hot-Humid
    ZONE_2A = "2A"  # Hot-Humid
    ZONE_2B = "2B"  # Hot-Dry
    ZONE_3A = "3A"  # Warm-Humid
    ZONE_3B = "3B"  # Warm-Dry
    ZONE_3C = "3C"  # Warm-Marine
    ZONE_4A = "4A"  # Mixed-Humid
    ZONE_4B = "4B"  # Mixed-Dry
    ZONE_4C = "4C"  # Mixed-Marine
    ZONE_5A = "5A"  # Cool-Humid
    ZONE_5B = "5B"  # Cool-Dry
    ZONE_5C = "5C"  # Cool-Marine
    ZONE_6A = "6A"  # Cold-Humid
    ZONE_6B = "6B"  # Cold-Dry
    ZONE_7 = "7"    # Very Cold
    ZONE_8 = "8"    # Subarctic

class BuildingCodeType(str, Enum):
    """Building code types"""
    INTERNATIONAL_MECHANICAL = "IMC"
    UNIFORM_MECHANICAL = "UMC"
    ASHRAE_90_1 = "ASHRAE_90.1"
    IECC = "IECC"
    LOCAL = "LOCAL"

class LocationIntelligence:
    """
    Location-specific intelligence for HVAC systems
    """
    
    def __init__(self):
        self._load_location_data()
        
    def _load_location_data(self):
        """Load location-specific data"""
        # In production, load from database or API
        self.climate_zones_by_state = {
            'CA': [ClimateZone.ZONE_3B, ClimateZone.ZONE_3C],
            'NY': [ClimateZone.ZONE_4A, ClimateZone.ZONE_5A],
            'FL': [ClimateZone.ZONE_1A, ClimateZone.ZONE_2A],
            'TX': [ClimateZone.ZONE_2A, ClimateZone.ZONE_2B, ClimateZone.ZONE_3A],
            'IL': [ClimateZone.ZONE_5A],
            # Add more states...
        }
        
        self.cost_multipliers = {
            'CA': {'labor': 1.35, 'material': 1.15},
            'NY': {'labor': 1.40, 'material': 1.20},
            'FL': {'labor': 1.10, 'material': 1.05},
            'TX': {'labor': 1.05, 'material': 1.02},
            'IL': {'labor': 1.25, 'material': 1.10},
            'default': {'labor': 1.00, 'material': 1.00}
        }
        
        self.building_codes = {
            'CA': [BuildingCodeType.INTERNATIONAL_MECHANICAL, BuildingCodeType.ASHRAE_90_1],
            'NY': [BuildingCodeType.INTERNATIONAL_MECHANICAL],
            'FL': [BuildingCodeType.INTERNATIONAL_MECHANICAL],
            'default': [BuildingCodeType.INTERNATIONAL_MECHANICAL]
        }
    
    def get_climate_zone(self, location: str) -> Optional[ClimateZone]:
        """
        Determine climate zone for location
        
        Args:
            location: Location string (city, state, zip)
            
        Returns:
            Climate zone enum
        """
        # Parse location to extract state
        state = self._parse_state(location)
        
        if state in self.climate_zones_by_state:
            # Return first zone for simplicity
            return self.climate_zones_by_state[state][0]
        
        return None
    
    def get_cost_adjustments(self, location: str) -> Dict[str, float]:
        """
        Get regional cost adjustment multipliers
        
        Args:
            location: Location string
            
        Returns:
            Dictionary with labor and material multipliers
        """
        state = self._parse_state(location)
        return self.cost_multipliers.get(state, self.cost_multipliers['default'])
    
    def get_building_codes(self, location: str) -> List[BuildingCodeType]:
        """
        Get applicable building codes for location
        
        Args:
            location: Location string
            
        Returns:
            List of applicable building code types
        """
        state = self._parse_state(location)
        return self.building_codes.get(state, self.building_codes['default'])
    
    def get_equipment_requirements(self, climate_zone: ClimateZone) -> Dict[str, Any]:
        """
        Get equipment requirements based on climate zone
        
        Args:
            climate_zone: Climate zone enum
            
        Returns:
            Equipment requirements dictionary
        """
        # Climate-specific requirements
        requirements = {
            'min_seer': 13,
            'min_hspf': 7.7,
            'min_eer': 11,
            'ventilation_cfm_per_person': 15
        }
        
        # Adjust based on zone
        if climate_zone.value.startswith('1') or climate_zone.value.startswith('2'):
            # Hot zones - higher cooling efficiency
            requirements['min_seer'] = 14
            requirements['min_eer'] = 12
        elif climate_zone.value.startswith('7') or climate_zone.value.startswith('8'):
            # Cold zones - higher heating efficiency
            requirements['min_hspf'] = 8.5
        
        return requirements
    
    def check_compliance(self, system_specs: Dict[str, Any], location: str) -> List[str]:
        """
        Check system compliance with regional requirements
        
        Args:
            system_specs: System specification dictionary
            location: Location string
            
        Returns:
            List of compliance notes/violations
        """
        notes = []
        climate_zone = self.get_climate_zone(location)
        
        if climate_zone:
            requirements = self.get_equipment_requirements(climate_zone)
            
            # Check SEER rating
            if 'seer' in system_specs:
                if system_specs['seer'] < requirements['min_seer']:
                    notes.append(f"SEER rating {system_specs['seer']} below minimum {requirements['min_seer']} for {climate_zone.value}")
                else:
                    notes.append(f"SEER rating meets requirements for climate zone {climate_zone.value}")
            
            # Check ventilation
            if 'ventilation_cfm' in system_specs and 'occupancy' in system_specs:
                required_cfm = requirements['ventilation_cfm_per_person'] * system_specs['occupancy']
                if system_specs['ventilation_cfm'] < required_cfm:
                    notes.append(f"Ventilation CFM below required {required_cfm} CFM")
        
        # Check building codes
        codes = self.get_building_codes(location)
        notes.append(f"Applicable codes: {', '.join([code.value for code in codes])}")
        
        return notes
    
    def _parse_state(self, location: str) -> str:
        """Parse state code from location string"""
        # Simple implementation - in production, use geocoding service
        location_upper = location.upper()
        
        # Check for state codes
        for state in self.climate_zones_by_state.keys():
            if state in location_upper:
                return state
        
        return 'default'


def create_location_intelligence() -> LocationIntelligence:
    """Factory function to create location intelligence"""
    return LocationIntelligence()
