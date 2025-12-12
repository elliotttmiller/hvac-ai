"""
SMACNA (Sheet Metal and Air Conditioning Contractors' National Association)
Duct Construction Standards Validator

Implements validation logic for SMACNA Duct Construction Standards:
- Minimum duct sizing requirements based on airflow and velocity
- Support spacing requirements
- Static pressure classifications
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class DuctType(Enum):
    """SMACNA duct classification types"""
    SUPPLY = "supply"
    RETURN = "return"
    EXHAUST = "exhaust"
    OUTSIDE_AIR = "outside_air"


class DuctMaterial(Enum):
    """Duct construction materials"""
    GALVANIZED_STEEL = "galvanized_steel"
    STAINLESS_STEEL = "stainless_steel"
    ALUMINUM = "aluminum"
    FIBERGLASS = "fiberglass"
    FLEXIBLE = "flexible"


@dataclass
class DuctSegment:
    """Represents a duct segment in the HVAC system"""
    segment_id: str
    duct_type: DuctType
    material: DuctMaterial
    width: Optional[float] = None  # inches (for rectangular)
    height: Optional[float] = None  # inches (for rectangular)
    diameter: Optional[float] = None  # inches (for round)
    length: float = 0.0  # feet
    design_airflow: float = 0.0  # CFM
    static_pressure: float = 2.0  # inches water column (default)
    
    @property
    def cross_section_area(self) -> float:
        """Calculate cross-sectional area in square feet"""
        if self.diameter:
            # Round duct: A = π * r²
            radius_ft = (self.diameter / 2.0) / 12.0
            return math.pi * radius_ft ** 2
        elif self.width and self.height:
            # Rectangular duct: A = width * height
            return (self.width / 12.0) * (self.height / 12.0)
        else:
            return 0.0
    
    @property
    def is_rectangular(self) -> bool:
        """Check if duct is rectangular"""
        return self.width is not None and self.height is not None
    
    @property
    def is_round(self) -> bool:
        """Check if duct is round"""
        return self.diameter is not None


# SMACNA Maximum Recommended Velocities (FPM - Feet Per Minute)
# Based on SMACNA HVAC Systems Duct Design, 4th Edition
MAX_VELOCITY_TABLE = {
    DuctType.SUPPLY: {
        "main_trunk": 1800,      # Main supply trunks
        "branch": 1200,          # Branch ducts
        "terminal": 800          # Terminal ducts near diffusers
    },
    DuctType.RETURN: {
        "main_trunk": 1500,
        "branch": 1000,
        "terminal": 700
    },
    DuctType.EXHAUST: {
        "main_trunk": 2000,
        "branch": 1500,
        "terminal": 1000
    },
    DuctType.OUTSIDE_AIR: {
        "main_trunk": 1500,
        "branch": 1200,
        "terminal": 800
    }
}


# SMACNA Support Spacing Requirements (feet)
# Based on duct size and material
SUPPORT_SPACING_TABLE = {
    DuctMaterial.GALVANIZED_STEEL: {
        "up_to_24": 10,      # Ducts up to 24" wide
        "24_to_60": 8,       # Ducts 24" to 60" wide
        "over_60": 6         # Ducts over 60" wide
    },
    DuctMaterial.FIBERGLASS: {
        "up_to_24": 8,
        "24_to_60": 6,
        "over_60": 4
    },
    DuctMaterial.FLEXIBLE: {
        "all_sizes": 4       # Flexible duct requires closer support
    }
}


class SMACNAValidator:
    """
    SMACNA Standards Compliance Validator
    
    Validates ductwork sizing, velocity, and construction requirements
    against SMACNA standards.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.velocity_table = MAX_VELOCITY_TABLE
        self.support_table = SUPPORT_SPACING_TABLE
    
    def calculate_duct_velocity(
        self,
        duct: DuctSegment
    ) -> float:
        """
        Calculate air velocity in duct
        
        Velocity (FPM) = Airflow (CFM) / Area (sq ft)
        
        Args:
            duct: DuctSegment to analyze
            
        Returns:
            Air velocity in feet per minute (FPM)
        """
        area = duct.cross_section_area
        if area == 0:
            return 0.0
        
        return duct.design_airflow / area
    
    def get_recommended_duct_size(
        self,
        airflow: float,
        duct_type: DuctType,
        duct_location: str = "main_trunk",
        round_duct: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate recommended duct size based on airflow and velocity limits
        
        Args:
            airflow: Design airflow in CFM
            duct_type: Type of duct (supply, return, etc.)
            duct_location: Location in system (main_trunk, branch, terminal)
            round_duct: True for round duct, False for rectangular
            
        Returns:
            Dictionary with recommended dimensions
        """
        max_velocity = self.velocity_table[duct_type].get(
            duct_location,
            self.velocity_table[duct_type]["main_trunk"]
        )
        
        # Calculate required area: Area = Airflow / Velocity
        required_area = airflow / max_velocity  # sq ft
        
        if round_duct:
            # For round duct: A = π * r²
            # Solve for diameter: d = 2 * sqrt(A / π)
            diameter_ft = 2.0 * math.sqrt(required_area / math.pi)
            diameter_in = diameter_ft * 12.0
            
            # Round up to standard duct sizes
            standard_sizes = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30, 36, 42, 48]
            recommended_diameter = next(
                (size for size in standard_sizes if size >= diameter_in),
                int(math.ceil(diameter_in))
            )
            
            return {
                "duct_type": "round",
                "diameter_inches": recommended_diameter,
                "area_sq_ft": math.pi * ((recommended_diameter / 2.0) / 12.0) ** 2,
                "max_velocity_fpm": max_velocity,
                "actual_velocity_fpm": airflow / (math.pi * ((recommended_diameter / 2.0) / 12.0) ** 2)
            }
        else:
            # For rectangular duct: Use aspect ratio of 2:1 for efficiency
            # A = width * height, with height = width / 2
            # A = width * (width / 2) = width² / 2
            # width = sqrt(2 * A)
            width_ft = math.sqrt(2.0 * required_area)
            height_ft = width_ft / 2.0
            
            width_in = int(math.ceil(width_ft * 12.0))
            height_in = int(math.ceil(height_ft * 12.0))
            
            return {
                "duct_type": "rectangular",
                "width_inches": width_in,
                "height_inches": height_in,
                "area_sq_ft": (width_in / 12.0) * (height_in / 12.0),
                "max_velocity_fpm": max_velocity,
                "actual_velocity_fpm": airflow / ((width_in / 12.0) * (height_in / 12.0))
            }
    
    def validate_duct_sizing(
        self,
        duct: DuctSegment,
        duct_location: str = "main_trunk"
    ) -> Dict[str, Any]:
        """
        Validate duct sizing against SMACNA velocity requirements
        
        Args:
            duct: DuctSegment to validate
            duct_location: Location in system (main_trunk, branch, terminal)
            
        Returns:
            Dictionary with validation results including violations
        """
        violations = []
        
        # Calculate actual velocity
        actual_velocity = self.calculate_duct_velocity(duct)
        
        # Get maximum allowed velocity
        max_velocity = self.velocity_table[duct.duct_type].get(
            duct_location,
            self.velocity_table[duct.duct_type]["main_trunk"]
        )
        
        # Check if velocity exceeds maximum
        if actual_velocity > max_velocity:
            excess_pct = ((actual_velocity - max_velocity) / max_velocity) * 100
            severity = "CRITICAL" if excess_pct > 50 else "WARNING"
            
            # Calculate recommended size
            recommended = self.get_recommended_duct_size(
                duct.design_airflow,
                duct.duct_type,
                duct_location,
                duct.is_round
            )
            
            violations.append({
                "severity": severity,
                "code_reference": "SMACNA HVAC Systems Duct Design, 4th Edition",
                "description": (
                    f"Duct {duct.segment_id}: Air velocity ({actual_velocity:.0f} FPM) "
                    f"exceeds maximum recommended velocity ({max_velocity:.0f} FPM) "
                    f"by {excess_pct:.1f}%"
                ),
                "remediation": (
                    f"Increase duct size to {recommended.get('diameter_inches', recommended.get('width_inches'))} "
                    f"inches {'diameter' if duct.is_round else 'width'} "
                    f"to reduce velocity to {recommended['actual_velocity_fpm']:.0f} FPM"
                ),
                "cost_impact": self._estimate_duct_sizing_cost(duct, recommended),
                "confidence": 0.88
            })
        
        # Check for undersized ducts (too low velocity can indicate oversizing)
        min_velocity = max_velocity * 0.3  # Minimum 30% of max to avoid oversizing
        if actual_velocity > 0 and actual_velocity < min_velocity:
            violations.append({
                "severity": "INFO",
                "code_reference": "SMACNA Best Practices",
                "description": (
                    f"Duct {duct.segment_id}: Air velocity ({actual_velocity:.0f} FPM) "
                    f"is below recommended minimum ({min_velocity:.0f} FPM). "
                    f"Duct may be oversized."
                ),
                "remediation": (
                    f"Consider reducing duct size for better air distribution "
                    f"and cost efficiency"
                ),
                "confidence": 0.65
            })
        
        return {
            "segment_id": duct.segment_id,
            "is_compliant": len([v for v in violations if v["severity"] in ["CRITICAL", "WARNING"]]) == 0,
            "violations": violations,
            "analysis": {
                "actual_velocity_fpm": actual_velocity,
                "max_velocity_fpm": max_velocity,
                "cross_section_area_sqft": duct.cross_section_area,
                "design_airflow_cfm": duct.design_airflow
            }
        }
    
    def validate_support_spacing(
        self,
        duct: DuctSegment,
        support_locations: List[float]
    ) -> Dict[str, Any]:
        """
        Validate duct support spacing against SMACNA requirements
        
        Args:
            duct: DuctSegment to validate
            support_locations: List of support positions along duct (in feet)
            
        Returns:
            Dictionary with validation results
        """
        violations = []
        
        # Determine required support spacing
        if duct.material == DuctMaterial.FLEXIBLE:
            max_spacing = self.support_table[DuctMaterial.FLEXIBLE]["all_sizes"]
        else:
            # Get duct width for spacing determination
            width = duct.width if duct.width else duct.diameter
            
            if width is None:
                return {
                    "segment_id": duct.segment_id,
                    "is_compliant": False,
                    "violations": [{
                        "severity": "WARNING",
                        "description": "Cannot validate support spacing: duct dimensions unknown"
                    }]
                }
            
            spacing_table = self.support_table.get(
                duct.material,
                self.support_table[DuctMaterial.GALVANIZED_STEEL]
            )
            
            if width <= 24:
                max_spacing = spacing_table["up_to_24"]
            elif width <= 60:
                max_spacing = spacing_table["24_to_60"]
            else:
                max_spacing = spacing_table["over_60"]
        
        # Check spacing between consecutive supports
        sorted_locations = sorted(support_locations)
        for i in range(len(sorted_locations) - 1):
            spacing = sorted_locations[i + 1] - sorted_locations[i]
            if spacing > max_spacing:
                violations.append({
                    "severity": "WARNING",
                    "code_reference": "SMACNA HVAC Systems Duct Design - Support Requirements",
                    "description": (
                        f"Duct {duct.segment_id}: Support spacing ({spacing:.1f} ft) "
                        f"exceeds maximum allowed ({max_spacing} ft)"
                    ),
                    "remediation": (
                        f"Add additional support between {sorted_locations[i]:.1f} ft "
                        f"and {sorted_locations[i + 1]:.1f} ft"
                    ),
                    "confidence": 0.75
                })
        
        return {
            "segment_id": duct.segment_id,
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "analysis": {
                "max_spacing_ft": max_spacing,
                "support_count": len(support_locations),
                "duct_length_ft": duct.length
            }
        }
    
    def _estimate_duct_sizing_cost(
        self,
        current_duct: DuctSegment,
        recommended_size: Dict[str, Any]
    ) -> float:
        """
        Estimate cost impact of duct resizing
        
        Args:
            current_duct: Current duct configuration
            recommended_size: Recommended duct size from validation
            
        Returns:
            Estimated cost to correct in dollars
        """
        # Cost estimation: $15-25 per linear foot for duct replacement
        cost_per_linear_foot = 20.0
        labor_multiplier = 1.5  # 50% additional for labor and fittings
        
        base_cost = current_duct.length * cost_per_linear_foot * labor_multiplier
        
        # Add minimum cost for engineering and mobilization
        return max(base_cost, 800.0)
