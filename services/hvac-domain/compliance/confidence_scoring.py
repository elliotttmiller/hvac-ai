"""
Confidence Scoring System for Code Compliance Validation

Provides risk-based scoring and severity classification for violations
to help prioritize remediation efforts.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViolationSeverity(Enum):
    """Violation severity levels"""
    CRITICAL = "CRITICAL"  # Life safety issues
    WARNING = "WARNING"    # Performance/compliance issues
    INFO = "INFO"          # Best practice recommendations


class ConfidenceLevel(Enum):
    """Detection confidence levels"""
    HIGH = "HIGH"       # >85% confidence
    MEDIUM = "MEDIUM"   # 60-85% confidence
    LOW = "LOW"         # <60% confidence


@dataclass
class RiskScore:
    """Risk score for a violation"""
    severity: ViolationSeverity
    confidence: float  # 0.0 to 1.0
    cost_impact: float  # Estimated cost in dollars
    priority: int  # 1 (highest) to 5 (lowest)
    risk_score: float  # Computed overall risk score


class ConfidenceScorer:
    """
    Confidence Scoring System for HVAC Code Compliance
    
    Assigns confidence scores and risk-based priorities to violations
    based on detection quality, severity, and impact.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Severity weights for risk calculation
        self.severity_weights = {
            ViolationSeverity.CRITICAL: 10.0,
            ViolationSeverity.WARNING: 5.0,
            ViolationSeverity.INFO: 1.0
        }
    
    def calculate_risk_score(
        self,
        violation: Dict[str, Any]
    ) -> RiskScore:
        """
        Calculate comprehensive risk score for a violation
        
        Risk Score = (Severity Weight × Confidence × Cost Factor)
        
        Args:
            violation: Dictionary containing violation details
            
        Returns:
            RiskScore object with computed metrics
        """
        # Extract violation details
        severity_str = violation.get("severity", "WARNING")
        try:
            severity = ViolationSeverity[severity_str]
        except KeyError:
            severity = ViolationSeverity.WARNING
        
        confidence = violation.get("confidence", 0.7)
        cost_impact = violation.get("cost_impact", 1000.0)
        
        # Calculate severity component
        severity_weight = self.severity_weights[severity]
        
        # Calculate cost factor (normalized to 0-10 scale)
        # $0-1000 = 1, $10,000+ = 10
        cost_factor = min(10.0, max(1.0, cost_impact / 1000.0))
        
        # Compute overall risk score
        risk_score = severity_weight * confidence * cost_factor
        
        # Assign priority (1-5) based on risk score
        if risk_score >= 80:
            priority = 1
        elif risk_score >= 50:
            priority = 2
        elif risk_score >= 20:
            priority = 3
        elif risk_score >= 5:
            priority = 4
        else:
            priority = 5
        
        return RiskScore(
            severity=severity,
            confidence=confidence,
            cost_impact=cost_impact,
            priority=priority,
            risk_score=risk_score
        )
    
    def classify_confidence_level(
        self,
        confidence: float
    ) -> ConfidenceLevel:
        """
        Classify confidence score into discrete levels
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            ConfidenceLevel enum
        """
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def enrich_violations(
        self,
        violations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich violations with risk scores and confidence levels
        
        Args:
            violations: List of violation dictionaries
            
        Returns:
            Enriched violations with additional scoring metadata
        """
        enriched = []
        
        for violation in violations:
            risk_score = self.calculate_risk_score(violation)
            confidence_level = self.classify_confidence_level(
                violation.get("confidence", 0.7)
            )
            
            # Add enrichment data
            enriched_violation = violation.copy()
            enriched_violation.update({
                "risk_score": risk_score.risk_score,
                "priority": risk_score.priority,
                "confidence_level": confidence_level.value
            })
            
            enriched.append(enriched_violation)
        
        # Sort by priority (ascending) and risk score (descending)
        enriched.sort(key=lambda v: (v["priority"], -v["risk_score"]))
        
        return enriched
    
    def calculate_compliance_score(
        self,
        violations: List[Dict[str, Any]],
        total_components: int
    ) -> float:
        """
        Calculate overall compliance score (0-100)
        
        Compliance Score = 100 - (Total Risk / Max Possible Risk) × 100
        
        Args:
            violations: List of violations
            total_components: Total number of components analyzed
            
        Returns:
            Compliance score (0.0 to 100.0)
        """
        if total_components == 0:
            return 100.0
        
        # Calculate total risk
        total_risk = sum(
            self.calculate_risk_score(v).risk_score
            for v in violations
        )
        
        # Estimate maximum possible risk
        # Assume each component could have one critical violation
        max_risk_per_component = self.severity_weights[ViolationSeverity.CRITICAL] * 1.0 * 5.0
        max_possible_risk = total_components * max_risk_per_component
        
        if max_possible_risk == 0:
            return 100.0
        
        # Calculate compliance score
        risk_ratio = total_risk / max_possible_risk
        compliance_score = max(0.0, min(100.0, 100.0 - (risk_ratio * 100.0)))
        
        return round(compliance_score, 1)
    
    def generate_compliance_summary(
        self,
        violations: List[Dict[str, Any]],
        total_components: int
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance summary with scoring
        
        Args:
            violations: List of violations
            total_components: Total number of components analyzed
            
        Returns:
            Dictionary with compliance summary metrics
        """
        enriched_violations = self.enrich_violations(violations)
        
        # Count by severity
        critical_count = sum(
            1 for v in violations if v.get("severity") == "CRITICAL"
        )
        warning_count = sum(
            1 for v in violations if v.get("severity") == "WARNING"
        )
        info_count = sum(
            1 for v in violations if v.get("severity") == "INFO"
        )
        
        # Count by priority
        priority_distribution = {}
        for v in enriched_violations:
            priority = v.get("priority", 5)
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        # Calculate total estimated cost
        total_cost = sum(v.get("cost_impact", 0.0) for v in violations)
        
        # Calculate compliance score
        compliance_score = self.calculate_compliance_score(
            violations,
            total_components
        )
        
        return {
            "compliance_score": compliance_score,
            "total_violations": len(violations),
            "critical_violations": critical_count,
            "warning_violations": warning_count,
            "info_violations": info_count,
            "priority_distribution": priority_distribution,
            "estimated_total_cost": round(total_cost, 2),
            "highest_priority_violations": enriched_violations[:5],  # Top 5
            "compliance_grade": self._get_compliance_grade(compliance_score)
        }
    
    def _get_compliance_grade(self, score: float) -> str:
        """
        Convert compliance score to letter grade
        
        Args:
            score: Compliance score (0-100)
            
        Returns:
            Letter grade (A+ to F)
        """
        if score >= 97:
            return "A+"
        elif score >= 93:
            return "A"
        elif score >= 90:
            return "A-"
        elif score >= 87:
            return "B+"
        elif score >= 83:
            return "B"
        elif score >= 80:
            return "B-"
        elif score >= 77:
            return "C+"
        elif score >= 73:
            return "C"
        elif score >= 70:
            return "C-"
        elif score >= 67:
            return "D+"
        elif score >= 63:
            return "D"
        elif score >= 60:
            return "D-"
        else:
            return "F"
