"""
International Mechanical Code (IMC) Fire Damper Validation

Implements validation logic for IMC fire damper and smoke damper requirements:
- Fire damper placement at fire-rated assembly penetrations
- Smoke damper placement in smoke control zones
- Fire resistance ratings and UL listings
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DamperType(Enum):
    """Types of fire and smoke dampers"""
    FIRE_DAMPER = "fire_damper"
    SMOKE_DAMPER = "smoke_damper"
    COMBINATION_FIRE_SMOKE = "combination_fire_smoke"
    CEILING_RADIATION = "ceiling_radiation_damper"


class FireRating(Enum):
    """Fire resistance ratings (hours)"""
    ONE_HALF_HOUR = 0.5
    THREE_QUARTER_HOUR = 0.75
    ONE_HOUR = 1.0
    ONE_AND_HALF_HOUR = 1.5
    TWO_HOUR = 2.0
    THREE_HOUR = 3.0
    FOUR_HOUR = 4.0


@dataclass
class FireRatedAssembly:
    """Represents a fire-rated wall, floor, or partition"""
    assembly_id: str
    fire_rating: FireRating
    assembly_type: str  # "wall", "floor", "ceiling", "shaft"
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]


@dataclass
class DuctPenetration:
    """Represents a duct penetrating a fire-rated assembly"""
    penetration_id: str
    duct_id: str
    assembly_id: str
    location: Tuple[float, float]
    has_damper: bool = False
    damper_type: Optional[DamperType] = None
    damper_rating: Optional[FireRating] = None


class IMCFireCodeValidator:
    """
    International Mechanical Code Fire Safety Validator
    
    Validates fire damper and smoke damper placement according to
    IMC 2021 requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_fire_damper_placement(
        self,
        penetration: DuctPenetration,
        assembly: FireRatedAssembly
    ) -> Dict[str, Any]:
        """
        Validate fire damper placement at duct penetration
        
        Per IMC 2021 Section 607.5.1:
        "Fire dampers shall be listed and bear the label of an approved
        testing agency indicating compliance with UL 555."
        
        Args:
            penetration: DuctPenetration to validate
            assembly: FireRatedAssembly being penetrated
            
        Returns:
            Dictionary with validation results
        """
        violations = []
        
        # Check if damper is present
        if not penetration.has_damper:
            violations.append({
                "severity": "CRITICAL",
                "code_reference": "IMC 2021 Section 607.5.1",
                "description": (
                    f"Missing fire damper at penetration {penetration.penetration_id} "
                    f"of {assembly.fire_rating.value}-hour fire-rated {assembly.assembly_type}"
                ),
                "remediation": (
                    f"Install UL 555 listed fire damper with {assembly.fire_rating.value}-hour "
                    f"rating at indicated location"
                ),
                "cost_impact": self._estimate_damper_installation_cost(
                    DamperType.FIRE_DAMPER,
                    assembly.fire_rating
                ),
                "confidence": 0.95,
                "priority": 1
            })
        else:
            # Check if damper rating matches assembly rating
            if penetration.damper_rating != assembly.fire_rating:
                violations.append({
                    "severity": "CRITICAL",
                    "code_reference": "IMC 2021 Section 607.5.1",
                    "description": (
                        f"Fire damper rating ({penetration.damper_rating.value if penetration.damper_rating else 'unknown'}-hour) "
                        f"does not match fire-rated assembly rating ({assembly.fire_rating.value}-hour)"
                    ),
                    "remediation": (
                        f"Replace with UL 555 listed fire damper rated for "
                        f"{assembly.fire_rating.value}-hour fire resistance"
                    ),
                    "cost_impact": 2500.0,
                    "confidence": 0.90,
                    "priority": 1
                })
            
            # Verify damper type is appropriate
            if penetration.damper_type not in [
                DamperType.FIRE_DAMPER,
                DamperType.COMBINATION_FIRE_SMOKE
            ]:
                violations.append({
                    "severity": "WARNING",
                    "code_reference": "IMC 2021 Section 607.5",
                    "description": (
                        f"Damper type ({penetration.damper_type.value if penetration.damper_type else 'unknown'}) "
                        f"may not be appropriate for fire-rated assembly penetration"
                    ),
                    "remediation": (
                        "Verify damper is UL 555 listed fire damper or "
                        "UL 555S combination fire/smoke damper"
                    ),
                    "confidence": 0.75
                })
        
        return {
            "penetration_id": penetration.penetration_id,
            "is_compliant": len([v for v in violations if v["severity"] == "CRITICAL"]) == 0,
            "violations": violations,
            "assembly_info": {
                "assembly_id": assembly.assembly_id,
                "fire_rating": assembly.fire_rating.value,
                "assembly_type": assembly.assembly_type
            }
        }
    
    def validate_smoke_damper_placement(
        self,
        penetration: DuctPenetration,
        is_smoke_barrier: bool = False
    ) -> Dict[str, Any]:
        """
        Validate smoke damper placement per IMC requirements
        
        Per IMC 2021 Section 607.5.3:
        "Smoke dampers shall be listed and labeled, and shall bear the
        label of an approved testing agency indicating compliance with UL 555S."
        
        Args:
            penetration: DuctPenetration to validate
            is_smoke_barrier: True if penetration is through smoke barrier
            
        Returns:
            Dictionary with validation results
        """
        violations = []
        
        if is_smoke_barrier:
            # Smoke damper required at smoke barrier penetrations
            if not penetration.has_damper:
                violations.append({
                    "severity": "CRITICAL",
                    "code_reference": "IMC 2021 Section 607.5.3",
                    "description": (
                        f"Missing smoke damper at penetration {penetration.penetration_id} "
                        f"of smoke barrier"
                    ),
                    "remediation": (
                        "Install UL 555S listed smoke damper at indicated location"
                    ),
                    "cost_impact": self._estimate_damper_installation_cost(
                        DamperType.SMOKE_DAMPER,
                        FireRating.ONE_HALF_HOUR
                    ),
                    "confidence": 0.92,
                    "priority": 1
                })
            elif penetration.damper_type not in [
                DamperType.SMOKE_DAMPER,
                DamperType.COMBINATION_FIRE_SMOKE
            ]:
                violations.append({
                    "severity": "CRITICAL",
                    "code_reference": "IMC 2021 Section 607.5.3",
                    "description": (
                        f"Damper at smoke barrier penetration is not smoke damper "
                        f"(current type: {penetration.damper_type.value if penetration.damper_type else 'unknown'})"
                    ),
                    "remediation": (
                        "Replace with UL 555S listed smoke damper or "
                        "combination fire/smoke damper"
                    ),
                    "cost_impact": 2800.0,
                    "confidence": 0.88,
                    "priority": 1
                })
        
        return {
            "penetration_id": penetration.penetration_id,
            "is_compliant": len([v for v in violations if v["severity"] == "CRITICAL"]) == 0,
            "violations": violations,
            "barrier_info": {
                "is_smoke_barrier": is_smoke_barrier
            }
        }
    
    def validate_damper_system(
        self,
        penetrations: List[DuctPenetration],
        assemblies: List[FireRatedAssembly],
        smoke_barriers: List[str]  # List of assembly IDs that are smoke barriers
    ) -> Dict[str, Any]:
        """
        Validate entire damper system for multiple penetrations
        
        Args:
            penetrations: List of duct penetrations to validate
            assemblies: List of fire-rated assemblies
            smoke_barriers: List of assembly IDs that are smoke barriers
            
        Returns:
            Aggregated validation results
        """
        results = []
        all_violations = []
        
        # Create assembly lookup
        assembly_map = {a.assembly_id: a for a in assemblies}
        
        for penetration in penetrations:
            assembly = assembly_map.get(penetration.assembly_id)
            
            if assembly is None:
                self.logger.warning(
                    f"Assembly {penetration.assembly_id} not found for "
                    f"penetration {penetration.penetration_id}"
                )
                continue
            
            # Validate fire damper
            fire_validation = self.validate_fire_damper_placement(
                penetration,
                assembly
            )
            results.append(fire_validation)
            all_violations.extend(fire_validation["violations"])
            
            # Validate smoke damper if at smoke barrier
            if penetration.assembly_id in smoke_barriers:
                smoke_validation = self.validate_smoke_damper_placement(
                    penetration,
                    is_smoke_barrier=True
                )
                all_violations.extend(smoke_validation["violations"])
        
        critical_count = sum(
            1 for v in all_violations if v["severity"] == "CRITICAL"
        )
        warning_count = sum(
            1 for v in all_violations if v["severity"] == "WARNING"
        )
        
        return {
            "total_penetrations": len(penetrations),
            "compliant_penetrations": sum(1 for r in results if r["is_compliant"]),
            "total_violations": len(all_violations),
            "critical_violations": critical_count,
            "warning_violations": warning_count,
            "penetration_results": results,
            "overall_compliance": critical_count == 0
        }
    
    def _estimate_damper_installation_cost(
        self,
        damper_type: DamperType,
        fire_rating: FireRating
    ) -> float:
        """
        Estimate cost of damper installation
        
        Args:
            damper_type: Type of damper to install
            fire_rating: Fire resistance rating required
            
        Returns:
            Estimated cost in dollars
        """
        # Base damper costs
        base_costs = {
            DamperType.FIRE_DAMPER: 800.0,
            DamperType.SMOKE_DAMPER: 1200.0,
            DamperType.COMBINATION_FIRE_SMOKE: 1800.0,
            DamperType.CEILING_RADIATION: 900.0
        }
        
        # Fire rating multiplier (higher ratings cost more due to complexity)
        rating_multiplier = {
            FireRating.ONE_HALF_HOUR: 1.0,
            FireRating.THREE_QUARTER_HOUR: 1.1,
            FireRating.ONE_HOUR: 1.2,
            FireRating.ONE_AND_HALF_HOUR: 1.4,
            FireRating.TWO_HOUR: 1.6,
            FireRating.THREE_HOUR: 2.0,
            FireRating.FOUR_HOUR: 2.5
        }
        
        base_cost = base_costs.get(damper_type, 1000.0)
        multiplier = rating_multiplier.get(fire_rating, 1.0)
        
        # Add installation labor (typically 100-150% of material cost)
        labor_cost = base_cost * 1.25
        
        return (base_cost * multiplier) + labor_cost
