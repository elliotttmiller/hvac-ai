"""
HVAC Engineering Rule Validator

Implements heuristic validation based on ASHRAE/SMACNA standards
for use in RKLF (Reinforcement Learning from Knowledge Feedback).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..data_schema import HVACComponentType, RelationshipType, HVACDataSchema

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation violations"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationViolation:
    """Represents a validation rule violation"""
    rule_id: str
    rule_name: str
    severity: ValidationSeverity
    message: str
    component_ids: List[str]
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    violations: List[ValidationViolation]
    warnings: int
    errors: int
    info: int
    
    def __str__(self) -> str:
        return (
            f"ValidationResult(valid={self.is_valid}, "
            f"errors={self.errors}, warnings={self.warnings}, info={self.info})"
        )


class HVACValidator:
    """Validates HVAC designs against engineering rules"""
    
    def __init__(self):
        """Initialize validator with HVAC rules"""
        self.schema = HVACDataSchema()
        self.rules = self.schema.ENGINEERING_RULES
    
    def validate_system(
        self,
        components: List[Dict],
        relationships: List[Dict]
    ) -> ValidationResult:
        """
        Validate complete HVAC system
        
        Args:
            components: List of detected components
            relationships: List of relationships between components
            
        Returns:
            Validation result with violations
        """
        violations = []
        
        # Run all validation checks
        violations.extend(self._validate_supply_exhaust_separation(components, relationships))
        violations.extend(self._validate_cfm_balance(components, relationships))
        violations.extend(self._validate_relationships(components, relationships))
        violations.extend(self._validate_component_attributes(components))
        
        # Count by severity
        errors = sum(1 for v in violations if v.severity == ValidationSeverity.CRITICAL)
        warnings = sum(1 for v in violations if v.severity == ValidationSeverity.WARNING)
        info = sum(1 for v in violations if v.severity == ValidationSeverity.INFO)
        
        is_valid = errors == 0
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            errors=errors,
            info=info
        )
    
    def _validate_supply_exhaust_separation(
        self,
        components: List[Dict],
        relationships: List[Dict]
    ) -> List[ValidationViolation]:
        """Validate that supply and exhaust systems are not connected"""
        violations = []
        
        # Build component type map
        component_types = {
            comp["component_id"]: comp["component_type"]
            for comp in components
        }
        
        # Check for invalid connections
        for rel in relationships:
            source_type = component_types.get(rel.get("source"))
            target_type = component_types.get(rel.get("target"))
            
            if not source_type or not target_type:
                continue
            
            # Check if supply connects to exhaust
            is_supply = "supply" in source_type.lower()
            is_exhaust = "exhaust" in target_type.lower()
            
            if is_supply and is_exhaust:
                violations.append(ValidationViolation(
                    rule_id="supply_exhaust_separation",
                    rule_name="Supply/Exhaust Separation",
                    severity=ValidationSeverity.CRITICAL,
                    message="Supply air duct cannot connect to exhaust system",
                    component_ids=[rel.get("source"), rel.get("target")],
                    expected="No connection",
                    actual="Direct connection found"
                ))
        
        return violations
    
    def _validate_cfm_balance(
        self,
        components: List[Dict],
        relationships: List[Dict]
    ) -> List[ValidationViolation]:
        """Validate CFM balance between supply and return"""
        violations = []
        
        # Calculate total supply and return CFM
        total_supply_cfm = 0
        total_return_cfm = 0
        
        for comp in components:
            comp_type = comp.get("component_type", "")
            attrs = comp.get("attributes", {})
            cfm = attrs.get("cfm", 0)
            
            if "supply" in comp_type.lower():
                total_supply_cfm += cfm
            elif "return" in comp_type.lower():
                total_return_cfm += cfm
        
        # Check balance (within 10% tolerance)
        if total_supply_cfm > 0 and total_return_cfm > 0:
            imbalance = abs(total_supply_cfm - total_return_cfm) / total_supply_cfm
            
            if imbalance > 0.10:  # 10% tolerance
                violations.append(ValidationViolation(
                    rule_id="cfm_balance",
                    rule_name="CFM Balance",
                    severity=ValidationSeverity.WARNING,
                    message=f"Supply/return CFM imbalance: {imbalance*100:.1f}%",
                    component_ids=[],
                    expected=f"Balance within 10% ({total_supply_cfm * 0.9:.0f}-{total_supply_cfm * 1.1:.0f} CFM)",
                    actual=f"Return CFM: {total_return_cfm:.0f}"
                ))
        
        return violations
    
    def _validate_relationships(
        self,
        components: List[Dict],
        relationships: List[Dict]
    ) -> List[ValidationViolation]:
        """Validate that relationships follow HVAC rules"""
        violations = []
        
        # Build component type map
        component_types = {
            comp["component_id"]: HVACComponentType(comp["component_type"])
            for comp in components
        }
        
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            rel_type_str = rel.get("type")
            
            if not all([source_id, target_id, rel_type_str]):
                continue
            
            source_type = component_types.get(source_id)
            target_type = component_types.get(target_id)
            
            if not source_type or not target_type:
                continue
            
            try:
                rel_type = RelationshipType(rel_type_str)
            except ValueError:
                violations.append(ValidationViolation(
                    rule_id="invalid_relationship_type",
                    rule_name="Invalid Relationship Type",
                    severity=ValidationSeverity.WARNING,
                    message=f"Unknown relationship type: {rel_type_str}",
                    component_ids=[source_id, target_id]
                ))
                continue
            
            # Validate relationship is allowed
            if not self.schema.validate_relationship(source_type, target_type, rel_type):
                violations.append(ValidationViolation(
                    rule_id="invalid_relationship",
                    rule_name="Invalid Component Relationship",
                    severity=ValidationSeverity.WARNING,
                    message=f"Invalid relationship: {source_type.value} -> {target_type.value}",
                    component_ids=[source_id, target_id],
                    expected="Valid HVAC relationship per schema",
                    actual=f"{rel_type.value} relationship not allowed"
                ))
        
        return violations
    
    def _validate_component_attributes(
        self,
        components: List[Dict]
    ) -> List[ValidationViolation]:
        """Validate component attributes (CFM ranges, sizes, etc.)"""
        violations = []
        
        for comp in components:
            comp_id = comp.get("component_id")
            comp_type_str = comp.get("component_type")
            attrs = comp.get("attributes", {})
            
            try:
                comp_type = HVACComponentType(comp_type_str)
            except ValueError:
                continue
            
            # Validate CFM range
            cfm = attrs.get("cfm")
            if cfm is not None:
                if not self.schema.validate_cfm_range(comp_type, cfm):
                    typical_attrs = self.schema.get_typical_attributes(comp_type)
                    cfm_range = typical_attrs.get("cfm_range", (0, 999999))
                    
                    violations.append(ValidationViolation(
                        rule_id="cfm_out_of_range",
                        rule_name="CFM Out of Range",
                        severity=ValidationSeverity.INFO,
                        message=f"CFM value unusual for {comp_type.value}",
                        component_ids=[comp_id],
                        expected=f"{cfm_range[0]}-{cfm_range[1]} CFM",
                        actual=f"{cfm} CFM"
                    ))
        
        return violations
    
    def calculate_reward(self, validation_result: ValidationResult) -> float:
        """
        Calculate reward for RKLF based on validation result
        
        Args:
            validation_result: Validation result
            
        Returns:
            Reward score (0.0 to 1.0)
        """
        # Start with perfect score
        reward = 1.0
        
        # Penalize based on violations
        for violation in validation_result.violations:
            if violation.severity == ValidationSeverity.CRITICAL:
                reward -= 0.3
            elif violation.severity == ValidationSeverity.WARNING:
                reward -= 0.1
            elif violation.severity == ValidationSeverity.INFO:
                reward -= 0.02
        
        # Ensure non-negative
        reward = max(0.0, reward)
        
        return reward
