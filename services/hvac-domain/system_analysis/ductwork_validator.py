"""
Ductwork Sizing and Connectivity Validator

Validates ductwork sizing, connectivity, and airflow distribution
using industry-standard calculation methods.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class DiffuserComponent:
    """Represents a diffuser or terminal unit"""
    id: str
    design_airflow: float  # CFM
    location: Tuple[float, float]
    zone_id: Optional[str] = None


@dataclass
class DuctSegmentInfo:
    """Detailed information about a duct segment"""
    id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    width: Optional[float] = None  # inches
    height: Optional[float] = None  # inches
    diameter: Optional[float] = None  # inches
    connected_diffuser_ids: List[str] = None
    
    def __post_init__(self):
        if self.connected_diffuser_ids is None:
            self.connected_diffuser_ids = []
    
    @property
    def length(self) -> float:
        """Calculate segment length in feet"""
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        return math.sqrt(dx*dx + dy*dy)
    
    @property
    def cross_section_area(self) -> float:
        """Calculate cross-sectional area in square feet"""
        if self.diameter:
            radius_ft = (self.diameter / 2.0) / 12.0
            return math.pi * radius_ft ** 2
        elif self.width and self.height:
            return (self.width / 12.0) * (self.height / 12.0)
        return 0.0


class DuctworkValidator:
    """
    Ductwork Sizing and Connectivity Validator
    
    Validates ductwork sizing based on connected diffusers and
    ensures proper connectivity throughout the system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Standard maximum velocities (FPM) by duct type
        self.max_velocities = {
            "main_trunk": 1800,
            "branch": 1200,
            "terminal": 800
        }
    
    def validate_ductwork_sizing(
        self,
        duct_segment: DuctSegmentInfo,
        connected_diffusers: List[DiffuserComponent],
        duct_location: str = "branch"
    ) -> Dict[str, Any]:
        """
        Validate ductwork sizing against ASHRAE 62.1 requirements
        
        Args:
            duct_segment: Duct segment to validate
            connected_diffusers: List of diffusers connected to this segment
            duct_location: Location type (main_trunk, branch, terminal)
            
        Returns:
            Dictionary with validation results and violations
        """
        violations = []
        
        # Calculate total airflow requirement from connected diffusers
        total_airflow = sum(d.design_airflow for d in connected_diffusers)
        
        if total_airflow == 0:
            return {
                "segment_id": duct_segment.id,
                "is_compliant": False,
                "violations": [{
                    "severity": "WARNING",
                    "description": f"Duct segment {duct_segment.id} has no connected diffusers or zero airflow",
                    "confidence": 0.70
                }],
                "analysis": {
                    "total_airflow_cfm": 0,
                    "actual_velocity_fpm": 0
                }
            }
        
        # Calculate air velocity
        area = duct_segment.cross_section_area
        if area == 0:
            violations.append({
                "severity": "WARNING",
                "code_reference": "ASHRAE 62.1-2019 Section 6.3.2",
                "description": (
                    f"Duct segment {duct_segment.id}: Cannot determine duct size. "
                    f"Dimensions not specified on blueprint."
                ),
                "remediation": "Specify duct dimensions on blueprint for validation",
                "confidence": 0.60
            })
            
            return {
                "segment_id": duct_segment.id,
                "is_compliant": False,
                "violations": violations,
                "analysis": {
                    "total_airflow_cfm": total_airflow,
                    "connected_diffuser_count": len(connected_diffusers)
                }
            }
        
        velocity = total_airflow / area
        max_velocity = self.max_velocities.get(duct_location, 1200)
        
        # Check if velocity exceeds maximum
        if velocity > max_velocity:
            excess_pct = ((velocity - max_velocity) / max_velocity) * 100
            severity = "CRITICAL" if excess_pct > 50 else "WARNING"
            
            # Calculate required duct size
            required_area = total_airflow / max_velocity
            
            if duct_segment.diameter:
                required_diameter = 2.0 * math.sqrt(required_area / math.pi) * 12.0
                size_recommendation = f"{math.ceil(required_diameter)}-inch diameter"
            else:
                required_width = math.sqrt(2.0 * required_area) * 12.0
                required_height = required_width / 2.0
                size_recommendation = f"{math.ceil(required_width)}x{math.ceil(required_height)} inches"
            
            violations.append({
                "severity": severity,
                "code_reference": "ASHRAE 62.1-2019 Section 6.3.2",
                "description": (
                    f"Duct segment {duct_segment.id}: Air velocity ({velocity:.0f} FPM) "
                    f"exceeds maximum allowed ({max_velocity:.0f} FPM) by {excess_pct:.1f}%"
                ),
                "remediation": (
                    f"Increase duct size to {size_recommendation} or reduce airflow. "
                    f"Current size is undersized for {total_airflow:.0f} CFM."
                ),
                "cost_impact": self._estimate_duct_resizing_cost(
                    duct_segment.length,
                    excess_pct
                ),
                "confidence": 0.92,
                "priority": 1 if severity == "CRITICAL" else 2
            })
        
        # Check for excessively low velocity (oversizing)
        min_velocity = max_velocity * 0.3
        if velocity < min_velocity:
            violations.append({
                "severity": "INFO",
                "description": (
                    f"Duct segment {duct_segment.id}: Air velocity ({velocity:.0f} FPM) "
                    f"is below recommended minimum ({min_velocity:.0f} FPM). "
                    f"Duct may be oversized, leading to increased costs."
                ),
                "remediation": "Consider reducing duct size for cost efficiency",
                "confidence": 0.65
            })
        
        return {
            "segment_id": duct_segment.id,
            "is_compliant": len([v for v in violations if v["severity"] in ["CRITICAL", "WARNING"]]) == 0,
            "violations": violations,
            "analysis": {
                "total_airflow_cfm": total_airflow,
                "actual_velocity_fpm": velocity,
                "max_velocity_fpm": max_velocity,
                "cross_section_area_sqft": area,
                "connected_diffuser_count": len(connected_diffusers),
                "velocity_ratio": velocity / max_velocity if max_velocity > 0 else 0
            }
        }
    
    def validate_duct_connectivity(
        self,
        diffusers: List[DiffuserComponent],
        duct_segments: List[DuctSegmentInfo]
    ) -> Dict[str, Any]:
        """
        Validate that all diffusers are connected to ductwork
        
        Args:
            diffusers: List of diffusers to validate
            duct_segments: List of duct segments in system
            
        Returns:
            Dictionary with connectivity validation results
        """
        violations = []
        
        # Build set of connected diffuser IDs
        connected_ids = set()
        for duct in duct_segments:
            connected_ids.update(duct.connected_diffuser_ids)
        
        # Check each diffuser for connectivity
        for diffuser in diffusers:
            if diffuser.id not in connected_ids:
                violations.append({
                    "severity": "CRITICAL",
                    "code_reference": "SMACNA Duct Design Standards",
                    "description": (
                        f"Diffuser {diffuser.id} is not connected to ductwork. "
                        f"Air terminal devices must connect to supply ductwork."
                    ),
                    "remediation": (
                        f"Connect diffuser {diffuser.id} to supply ductwork"
                    ),
                    "cost_impact": 1500.0,  # Estimated cost for new duct run
                    "confidence": 0.88,
                    "priority": 1
                })
        
        connected_count = len(diffusers) - len(violations)
        
        return {
            "total_diffusers": len(diffusers),
            "connected_diffusers": connected_count,
            "disconnected_diffusers": len(violations),
            "violations": violations,
            "overall_connectivity": len(violations) == 0
        }
    
    def validate_airflow_distribution(
        self,
        duct_segments: List[DuctSegmentInfo],
        diffusers: List[DiffuserComponent]
    ) -> Dict[str, Any]:
        """
        Validate airflow distribution throughout duct system
        
        Args:
            duct_segments: List of duct segments
            diffusers: List of diffusers
            
        Returns:
            Dictionary with airflow distribution analysis
        """
        violations = []
        
        # Group diffusers by zone if available
        zones = {}
        for diffuser in diffusers:
            zone_id = diffuser.zone_id or "default"
            if zone_id not in zones:
                zones[zone_id] = []
            zones[zone_id].append(diffuser)
        
        # Analyze airflow per zone
        zone_analysis = []
        for zone_id, zone_diffusers in zones.items():
            total_zone_airflow = sum(d.design_airflow for d in zone_diffusers)
            avg_airflow = total_zone_airflow / len(zone_diffusers) if zone_diffusers else 0
            
            # Check for highly unbalanced airflow
            if len(zone_diffusers) > 1:
                max_deviation = max(
                    abs(d.design_airflow - avg_airflow) / avg_airflow
                    for d in zone_diffusers
                    if avg_airflow > 0
                )
                
                if max_deviation > 0.5:  # More than 50% deviation
                    violations.append({
                        "severity": "WARNING",
                        "description": (
                            f"Zone {zone_id}: Airflow distribution is unbalanced. "
                            f"Maximum deviation is {max_deviation * 100:.1f}% from average."
                        ),
                        "remediation": (
                            "Review diffuser sizing and duct design for better "
                            "airflow balance across zone"
                        ),
                        "confidence": 0.70
                    })
            
            zone_analysis.append({
                "zone_id": zone_id,
                "diffuser_count": len(zone_diffusers),
                "total_airflow_cfm": total_zone_airflow,
                "average_airflow_cfm": avg_airflow
            })
        
        return {
            "zone_count": len(zones),
            "zone_analysis": zone_analysis,
            "violations": violations,
            "is_balanced": len(violations) == 0
        }
    
    def _estimate_duct_resizing_cost(
        self,
        duct_length: float,
        excess_percentage: float
    ) -> float:
        """
        Estimate cost of duct resizing
        
        Args:
            duct_length: Length of duct in feet
            excess_percentage: Percentage over maximum velocity
            
        Returns:
            Estimated cost in dollars
        """
        # Base cost: $20-30 per linear foot for duct replacement
        cost_per_foot = 25.0
        
        # Increase cost for severe violations (more extensive rework)
        severity_multiplier = 1.0 + (excess_percentage / 100.0)
        
        # Add labor and fittings (50% markup)
        labor_multiplier = 1.5
        
        base_cost = duct_length * cost_per_foot * severity_multiplier * labor_multiplier
        
        # Minimum cost for any duct modification
        return max(base_cost, 800.0)
