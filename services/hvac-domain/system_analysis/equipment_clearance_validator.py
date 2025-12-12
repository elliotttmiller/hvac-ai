"""
Equipment Clearance and Placement Validator

Validates equipment clearance requirements for service access,
maintenance, and safety compliance.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class EquipmentType(Enum):
    """Major HVAC equipment types"""
    AHU = "air_handling_unit"
    CHILLER = "chiller"
    BOILER = "boiler"
    COOLING_TOWER = "cooling_tower"
    HEAT_PUMP = "heat_pump"
    FAN = "fan"
    PUMP = "pump"
    COMPRESSOR = "compressor"


@dataclass
class EquipmentClearanceRequirement:
    """Clearance requirements for equipment type"""
    equipment_type: EquipmentType
    front_clearance: float  # inches (for access panels)
    rear_clearance: float   # inches
    side_clearance: float   # inches
    top_clearance: float    # inches
    code_reference: str


# ASHRAE and IMC equipment clearance requirements
EQUIPMENT_CLEARANCE_TABLE = {
    EquipmentType.AHU: EquipmentClearanceRequirement(
        equipment_type=EquipmentType.AHU,
        front_clearance=36.0,  # 36" for filter/coil access
        rear_clearance=24.0,
        side_clearance=24.0,
        top_clearance=36.0,
        code_reference="IMC 2021 Section 306.3"
    ),
    EquipmentType.CHILLER: EquipmentClearanceRequirement(
        equipment_type=EquipmentType.CHILLER,
        front_clearance=48.0,  # 48" for tube removal
        rear_clearance=36.0,
        side_clearance=36.0,
        top_clearance=48.0,
        code_reference="ASHRAE Equipment Handbook & IMC 2021"
    ),
    EquipmentType.BOILER: EquipmentClearanceRequirement(
        equipment_type=EquipmentType.BOILER,
        front_clearance=36.0,
        rear_clearance=24.0,
        side_clearance=24.0,
        top_clearance=36.0,
        code_reference="IMC 2021 Section 1005.1"
    ),
    EquipmentType.COOLING_TOWER: EquipmentClearanceRequirement(
        equipment_type=EquipmentType.COOLING_TOWER,
        front_clearance=48.0,
        rear_clearance=36.0,
        side_clearance=36.0,
        top_clearance=48.0,
        code_reference="ASHRAE Equipment Handbook"
    ),
    EquipmentType.HEAT_PUMP: EquipmentClearanceRequirement(
        equipment_type=EquipmentType.HEAT_PUMP,
        front_clearance=30.0,
        rear_clearance=24.0,
        side_clearance=24.0,
        top_clearance=48.0,
        code_reference="IMC 2021 Section 306.3"
    ),
}


@dataclass
class EquipmentComponent:
    """Represents a major equipment component"""
    id: str
    equipment_type: EquipmentType
    bbox: List[float]  # [x, y, width, height] in pixels
    location: Tuple[float, float]
    dimensions: Optional[Tuple[float, float, float]] = None  # (width, depth, height) in inches
    
    @property
    def footprint_area(self) -> float:
        """Calculate equipment footprint in square feet"""
        if self.dimensions:
            width_ft = self.dimensions[0] / 12.0
            depth_ft = self.dimensions[1] / 12.0
            return width_ft * depth_ft
        return 0.0


@dataclass
class Obstruction:
    """Represents an obstruction near equipment"""
    id: str
    type: str  # "wall", "column", "other_equipment", "ductwork"
    location: Tuple[float, float]
    bbox: List[float]


class EquipmentClearanceValidator:
    """
    Equipment Clearance and Placement Validator
    
    Validates that HVAC equipment has adequate clearance for
    service, maintenance, and code compliance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clearance_table = EQUIPMENT_CLEARANCE_TABLE
    
    def validate_equipment_clearance(
        self,
        equipment: EquipmentComponent,
        nearby_obstructions: List[Obstruction],
        pixel_to_inch_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validate equipment clearance requirements
        
        Args:
            equipment: Equipment to validate
            nearby_obstructions: List of nearby obstructions
            pixel_to_inch_scale: Scale factor to convert pixels to inches
            
        Returns:
            Dictionary with validation results
        """
        violations = []
        
        # Get clearance requirements for equipment type
        requirements = self.clearance_table.get(equipment.equipment_type)
        
        if requirements is None:
            self.logger.warning(
                f"No clearance requirements defined for {equipment.equipment_type}"
            )
            return {
                "equipment_id": equipment.id,
                "is_compliant": False,
                "violations": [{
                    "severity": "INFO",
                    "description": f"Clearance requirements not defined for {equipment.equipment_type.value}"
                }]
            }
        
        # Check clearance to each obstruction
        for obstruction in nearby_obstructions:
            clearance_violations = self._check_clearance_to_obstruction(
                equipment,
                obstruction,
                requirements,
                pixel_to_inch_scale
            )
            violations.extend(clearance_violations)
        
        # Check for adequate working space
        working_space_violations = self._check_working_space(
            equipment,
            nearby_obstructions,
            requirements
        )
        violations.extend(working_space_violations)
        
        return {
            "equipment_id": equipment.id,
            "equipment_type": equipment.equipment_type.value,
            "is_compliant": len([v for v in violations if v["severity"] in ["CRITICAL", "WARNING"]]) == 0,
            "violations": violations,
            "clearance_requirements": {
                "front_inches": requirements.front_clearance,
                "rear_inches": requirements.rear_clearance,
                "side_inches": requirements.side_clearance,
                "top_inches": requirements.top_clearance,
                "code_reference": requirements.code_reference
            }
        }
    
    def _check_clearance_to_obstruction(
        self,
        equipment: EquipmentComponent,
        obstruction: Obstruction,
        requirements: EquipmentClearanceRequirement,
        scale: float
    ) -> List[Dict[str, Any]]:
        """Check clearance between equipment and specific obstruction"""
        violations = []
        
        # Calculate distance between equipment and obstruction
        distance = self._calculate_distance(
            equipment.location,
            obstruction.location
        )
        distance_inches = distance * scale
        
        # Determine which side of equipment the obstruction is on
        # and check appropriate clearance
        dx = obstruction.location[0] - equipment.location[0]
        dy = obstruction.location[1] - equipment.location[1]
        
        # Simplified directional check
        if abs(dx) > abs(dy):
            # Obstruction is primarily to left or right
            required_clearance = requirements.side_clearance
            direction = "side"
        else:
            # Obstruction is primarily to front or back
            # Assume front for conservatism
            required_clearance = requirements.front_clearance
            direction = "front"
        
        if distance_inches < required_clearance:
            deficit = required_clearance - distance_inches
            deficit_pct = (deficit / required_clearance) * 100
            
            severity = "CRITICAL" if deficit_pct > 30 else "WARNING"
            
            violations.append({
                "severity": severity,
                "code_reference": requirements.code_reference,
                "description": (
                    f"Equipment {equipment.id} ({equipment.equipment_type.value}): "
                    f"Insufficient {direction} clearance to {obstruction.type}. "
                    f"Measured: {distance_inches:.1f}\", Required: {required_clearance:.1f}\" "
                    f"(deficit: {deficit:.1f}\" or {deficit_pct:.1f}%)"
                ),
                "remediation": (
                    f"Relocate {obstruction.type} to provide minimum {required_clearance:.1f}\" "
                    f"clearance, or relocate equipment"
                ),
                "cost_impact": self._estimate_clearance_correction_cost(
                    equipment.equipment_type,
                    deficit
                ),
                "confidence": 0.85,
                "priority": 1 if severity == "CRITICAL" else 2
            })
        
        return violations
    
    def _check_working_space(
        self,
        equipment: EquipmentComponent,
        nearby_obstructions: List[Obstruction],
        requirements: EquipmentClearanceRequirement
    ) -> List[Dict[str, Any]]:
        """Check for adequate working space around equipment"""
        violations = []
        
        # Count obstructions very close to equipment
        close_obstructions = [
            o for o in nearby_obstructions
            if self._calculate_distance(equipment.location, o.location) < 100  # Within 100 pixels
        ]
        
        if len(close_obstructions) >= 3:
            # Equipment is surrounded by obstructions
            violations.append({
                "severity": "WARNING",
                "code_reference": requirements.code_reference,
                "description": (
                    f"Equipment {equipment.id} ({equipment.equipment_type.value}): "
                    f"Working space is restricted by {len(close_obstructions)} "
                    f"nearby obstructions"
                ),
                "remediation": (
                    "Ensure adequate working space per IMC Section 306.3 for "
                    "maintenance and service access"
                ),
                "confidence": 0.75
            })
        
        return violations
    
    def validate_equipment_room_size(
        self,
        equipment_list: List[EquipmentComponent],
        room_dimensions: Tuple[float, float]  # (width, height) in feet
    ) -> Dict[str, Any]:
        """
        Validate mechanical room size for equipment
        
        Args:
            equipment_list: List of equipment in room
            room_dimensions: Room dimensions (width, height) in feet
            
        Returns:
            Dictionary with validation results
        """
        violations = []
        
        room_area = room_dimensions[0] * room_dimensions[1]
        total_equipment_footprint = sum(
            eq.footprint_area for eq in equipment_list
            if eq.footprint_area > 0
        )
        
        # Rule of thumb: Equipment should occupy <40% of room area
        # to allow adequate circulation and service space
        if total_equipment_footprint / room_area > 0.40:
            utilization_pct = (total_equipment_footprint / room_area) * 100
            
            violations.append({
                "severity": "WARNING",
                "code_reference": "ASHRAE Applications Handbook - Mechanical Room Design",
                "description": (
                    f"Mechanical room utilization is {utilization_pct:.1f}%. "
                    f"Recommended maximum is 40% to allow adequate service space."
                ),
                "remediation": (
                    "Consider enlarging mechanical room or relocating some "
                    "equipment to maintain adequate service clearances"
                ),
                "confidence": 0.70
            })
        
        return {
            "room_area_sqft": room_area,
            "equipment_footprint_sqft": total_equipment_footprint,
            "utilization_percentage": (total_equipment_footprint / room_area) * 100,
            "equipment_count": len(equipment_list),
            "violations": violations,
            "is_adequate": len(violations) == 0
        }
    
    def _calculate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _estimate_clearance_correction_cost(
        self,
        equipment_type: EquipmentType,
        clearance_deficit: float
    ) -> float:
        """
        Estimate cost to correct clearance violations
        
        Args:
            equipment_type: Type of equipment
            clearance_deficit: Clearance shortage in inches
            
        Returns:
            Estimated cost in dollars
        """
        # Base costs for equipment relocation
        relocation_costs = {
            EquipmentType.AHU: 5000.0,
            EquipmentType.CHILLER: 15000.0,
            EquipmentType.BOILER: 12000.0,
            EquipmentType.COOLING_TOWER: 8000.0,
            EquipmentType.HEAT_PUMP: 3000.0,
            EquipmentType.FAN: 2000.0,
            EquipmentType.PUMP: 1500.0,
        }
        
        base_cost = relocation_costs.get(equipment_type, 3000.0)
        
        # Adjust based on severity of deficit
        severity_multiplier = 1.0 + (clearance_deficit / 24.0)  # 24" = 100% increase
        
        return base_cost * severity_multiplier
