"""
ASHRAE Standard 62.1 Ventilation Requirements Validator

Implements validation logic for ASHRAE Standard 62.1-2019:
"Ventilation for Acceptable Indoor Air Quality"

Key validation features:
- Minimum outdoor air requirements by occupancy type
- Ventilation rate procedure calculations
- Zone-by-zone ventilation compliance checking
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OccupancyType(Enum):
    """ASHRAE 62.1 Table 6.2.2.1 Occupancy Categories"""
    OFFICE = "office"
    CLASSROOM = "classroom"
    CONFERENCE_ROOM = "conference_room"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    MECHANICAL_ROOM = "mechanical_room"
    CORRIDOR = "corridor"
    LOBBY = "lobby"
    STORAGE = "storage"
    UNKNOWN = "unknown"


@dataclass
class VentilationRequirement:
    """Ventilation requirements per ASHRAE 62.1"""
    occupancy_type: OccupancyType
    people_outdoor_air_rate: float  # CFM per person
    area_outdoor_air_rate: float    # CFM per sq ft
    occupant_density: float         # People per 1000 sq ft
    code_reference: str


# ASHRAE 62.1-2019 Table 6.2.2.1 - Minimum Ventilation Rates
ASHRAE_621_VENTILATION_TABLE = {
    OccupancyType.OFFICE: VentilationRequirement(
        occupancy_type=OccupancyType.OFFICE,
        people_outdoor_air_rate=5.0,  # CFM per person
        area_outdoor_air_rate=0.06,   # CFM per sq ft
        occupant_density=5.0,         # People per 1000 sq ft
        code_reference="ASHRAE 62.1-2019 Table 6.2.2.1"
    ),
    OccupancyType.CLASSROOM: VentilationRequirement(
        occupancy_type=OccupancyType.CLASSROOM,
        people_outdoor_air_rate=10.0,
        area_outdoor_air_rate=0.12,
        occupant_density=35.0,
        code_reference="ASHRAE 62.1-2019 Table 6.2.2.1"
    ),
    OccupancyType.CONFERENCE_ROOM: VentilationRequirement(
        occupancy_type=OccupancyType.CONFERENCE_ROOM,
        people_outdoor_air_rate=5.0,
        area_outdoor_air_rate=0.06,
        occupant_density=50.0,
        code_reference="ASHRAE 62.1-2019 Table 6.2.2.1"
    ),
    OccupancyType.RESTAURANT: VentilationRequirement(
        occupancy_type=OccupancyType.RESTAURANT,
        people_outdoor_air_rate=7.5,
        area_outdoor_air_rate=0.18,
        occupant_density=70.0,
        code_reference="ASHRAE 62.1-2019 Table 6.2.2.1"
    ),
    OccupancyType.RETAIL: VentilationRequirement(
        occupancy_type=OccupancyType.RETAIL,
        people_outdoor_air_rate=7.5,
        area_outdoor_air_rate=0.12,
        occupant_density=15.0,
        code_reference="ASHRAE 62.1-2019 Table 6.2.2.1"
    ),
    OccupancyType.WAREHOUSE: VentilationRequirement(
        occupancy_type=OccupancyType.WAREHOUSE,
        people_outdoor_air_rate=5.0,
        area_outdoor_air_rate=0.06,
        occupant_density=5.0,
        code_reference="ASHRAE 62.1-2019 Table 6.2.2.1"
    ),
    OccupancyType.CORRIDOR: VentilationRequirement(
        occupancy_type=OccupancyType.CORRIDOR,
        people_outdoor_air_rate=0.0,
        area_outdoor_air_rate=0.06,
        occupant_density=0.0,
        code_reference="ASHRAE 62.1-2019 Table 6.2.2.1"
    ),
}


@dataclass
class VentilationZone:
    """Represents a ventilation zone in the HVAC system"""
    zone_id: str
    occupancy_type: OccupancyType
    floor_area: float  # sq ft
    design_airflow: float  # CFM
    outdoor_air_flow: Optional[float] = None  # CFM
    occupant_count: Optional[int] = None


class ASHRAE621Validator:
    """
    ASHRAE Standard 62.1 Compliance Validator
    
    Validates HVAC system ventilation requirements against
    ASHRAE Standard 62.1-2019 specifications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ventilation_table = ASHRAE_621_VENTILATION_TABLE
    
    def calculate_minimum_outdoor_air(
        self,
        zone: VentilationZone
    ) -> Dict[str, Any]:
        """
        Calculate minimum outdoor air requirement for a zone
        
        Per ASHRAE 62.1-2019 Section 6.2.2.1:
        Voz = Rp * Pz + Ra * Az
        
        Where:
        - Voz = Outdoor air flow required in breathing zone (CFM)
        - Rp = People outdoor air rate (CFM/person)
        - Pz = Zone population (people)
        - Ra = Area outdoor air rate (CFM/sq ft)
        - Az = Zone floor area (sq ft)
        
        Args:
            zone: VentilationZone to validate
            
        Returns:
            Dictionary with calculation results and requirements
        """
        if zone.occupancy_type not in self.ventilation_table:
            self.logger.warning(
                f"Unknown occupancy type: {zone.occupancy_type}. "
                f"Using office defaults."
            )
            requirement = self.ventilation_table[OccupancyType.OFFICE]
        else:
            requirement = self.ventilation_table[zone.occupancy_type]
        
        # Calculate zone population if not provided
        if zone.occupant_count is None:
            zone_population = (
                zone.floor_area * requirement.occupant_density / 1000.0
            )
        else:
            zone_population = zone.occupant_count
        
        # Calculate minimum outdoor air (Voz)
        people_component = requirement.people_outdoor_air_rate * zone_population
        area_component = requirement.area_outdoor_air_rate * zone.floor_area
        minimum_outdoor_air = people_component + area_component
        
        return {
            "zone_id": zone.zone_id,
            "occupancy_type": zone.occupancy_type.value,
            "floor_area": zone.floor_area,
            "zone_population": zone_population,
            "minimum_outdoor_air_cfm": minimum_outdoor_air,
            "people_component_cfm": people_component,
            "area_component_cfm": area_component,
            "code_reference": requirement.code_reference,
            "calculation_method": "ASHRAE 62.1 Ventilation Rate Procedure"
        }
    
    def validate_zone_ventilation(
        self,
        zone: VentilationZone
    ) -> Dict[str, Any]:
        """
        Validate zone ventilation against ASHRAE 62.1 requirements
        
        Args:
            zone: VentilationZone to validate
            
        Returns:
            Dictionary with validation results including violations
        """
        calculation = self.calculate_minimum_outdoor_air(zone)
        minimum_oa = calculation["minimum_outdoor_air_cfm"]
        
        violations = []
        
        # Check if outdoor air flow is provided
        if zone.outdoor_air_flow is None:
            # If not specified, assume it's design airflow * 0.15 (typical minimum)
            estimated_oa = zone.design_airflow * 0.15
            
            if estimated_oa < minimum_oa:
                violations.append({
                    "severity": "WARNING",
                    "code_reference": calculation["code_reference"],
                    "description": (
                        f"Zone {zone.zone_id}: Estimated outdoor air "
                        f"({estimated_oa:.0f} CFM) may be below minimum "
                        f"requirement ({minimum_oa:.0f} CFM). "
                        f"Outdoor air flow not specified on blueprint."
                    ),
                    "remediation": (
                        f"Verify outdoor air flow meets minimum requirement "
                        f"of {minimum_oa:.0f} CFM for {zone.occupancy_type.value}"
                    ),
                    "confidence": 0.60  # Lower confidence without explicit OA spec
                })
        else:
            # Outdoor air flow is specified
            if zone.outdoor_air_flow < minimum_oa:
                deficit = minimum_oa - zone.outdoor_air_flow
                deficit_pct = (deficit / minimum_oa) * 100
                
                severity = "CRITICAL" if deficit_pct > 20 else "WARNING"
                
                violations.append({
                    "severity": severity,
                    "code_reference": calculation["code_reference"],
                    "description": (
                        f"Zone {zone.zone_id}: Outdoor air flow "
                        f"({zone.outdoor_air_flow:.0f} CFM) is below minimum "
                        f"requirement ({minimum_oa:.0f} CFM) by {deficit:.0f} CFM "
                        f"({deficit_pct:.1f}%)"
                    ),
                    "remediation": (
                        f"Increase outdoor air flow by {deficit:.0f} CFM to meet "
                        f"ASHRAE 62.1 minimum ventilation requirements"
                    ),
                    "cost_impact": self._estimate_ventilation_cost_impact(deficit),
                    "confidence": 0.92
                })
        
        return {
            "zone_id": zone.zone_id,
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "calculation": calculation,
            "summary": {
                "minimum_outdoor_air_cfm": minimum_oa,
                "provided_outdoor_air_cfm": zone.outdoor_air_flow,
                "design_airflow_cfm": zone.design_airflow,
                "compliance_status": "PASS" if len(violations) == 0 else "FAIL"
            }
        }
    
    def validate_multiple_zones(
        self,
        zones: List[VentilationZone]
    ) -> Dict[str, Any]:
        """
        Validate ventilation for multiple zones
        
        Args:
            zones: List of VentilationZone objects to validate
            
        Returns:
            Aggregated validation results for all zones
        """
        results = []
        total_violations = []
        
        for zone in zones:
            validation = self.validate_zone_ventilation(zone)
            results.append(validation)
            total_violations.extend(validation["violations"])
        
        critical_count = sum(
            1 for v in total_violations if v["severity"] == "CRITICAL"
        )
        warning_count = sum(
            1 for v in total_violations if v["severity"] == "WARNING"
        )
        
        return {
            "total_zones": len(zones),
            "compliant_zones": sum(1 for r in results if r["is_compliant"]),
            "non_compliant_zones": sum(1 for r in results if not r["is_compliant"]),
            "total_violations": len(total_violations),
            "critical_violations": critical_count,
            "warning_violations": warning_count,
            "zone_results": results,
            "overall_compliance": len(total_violations) == 0
        }
    
    def _estimate_ventilation_cost_impact(self, airflow_deficit: float) -> float:
        """
        Estimate cost impact of ventilation deficiency
        
        Args:
            airflow_deficit: CFM deficit from requirement
            
        Returns:
            Estimated cost to correct in dollars
        """
        # Cost estimation based on typical HVAC modification costs
        # $150-200 per CFM for ventilation system upgrades
        cost_per_cfm = 175.0
        base_cost = 500.0  # Minimum engineering/labor cost
        
        return base_cost + (airflow_deficit * cost_per_cfm)
