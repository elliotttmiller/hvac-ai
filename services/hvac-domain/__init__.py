"""
HVAC Domain Services

HVAC business logic and system validation:
- System relationship analysis
- Code compliance validation
- HVAC engineering rules
- Cost estimation
- Pricing services
- Location intelligence
"""

__version__ = "1.0.0"

# Import key components
from .hvac_system_engine import (
    HVACSystemEngine, HVACComponentType, SystemRelationship
)
from .hvac_compliance_analyzer import HVACComplianceAnalyzer
# relationship_graph provides RelationshipGraphBuilder; export a
# backwards-compatible name RelationshipGraph that points to the builder
from .relationship_graph import RelationshipGraphBuilder
RelationshipGraph = RelationshipGraphBuilder

# estimation.calculator provides EstimationEngine; export as CostCalculator
from .estimation.calculator import EstimationEngine
CostCalculator = EstimationEngine

# pricing.pricing_service exports PricingEngine; provide PricingService alias
from .pricing.pricing_service import PricingEngine, QuoteRequest, AnalysisData
PricingService = PricingEngine

from .location.intelligence import LocationIntelligence

__all__ = [
    "HVACSystemEngine",
    "HVACComponentType",
    "SystemRelationship",
    "HVACComplianceAnalyzer",
    "RelationshipGraph",
    "CostCalculator",
    "PricingService",
    "QuoteRequest",
    "AnalysisData",
    "LocationIntelligence",
]
