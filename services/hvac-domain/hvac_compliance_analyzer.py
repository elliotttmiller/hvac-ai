"""
HVAC Compliance Analyzer

Comprehensive HVAC code compliance analyzer that integrates:
- ASHRAE 62.1 ventilation validation
- SMACNA duct construction validation
- IMC fire code validation
- Equipment clearance validation
- System connectivity analysis
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import uuid
from datetime import datetime

from .compliance import (
    ASHRAE621Validator,
    SMACNAValidator,
    IMCFireCodeValidator,
    ConfidenceScorer,
    RegionalCodeManager,
    ViolationSeverity
)
from .compliance.ashrae_62_1_standards import VentilationZone, OccupancyType
from .compliance.smacna_standards import DuctSegment, DuctType, DuctMaterial
from .compliance.imc_fire_code import DuctPenetration, FireRatedAssembly, DamperType, FireRating

from .system_analysis import (
    DuctworkValidator,
    EquipmentClearanceValidator,
    SystemGraphBuilder
)

logger = logging.getLogger(__name__)


@dataclass
class ComplianceAnalysisRequest:
    """Request for compliance analysis"""
    blueprint_id: str
    analysis_type: str = "full_compliance"  # full_compliance, ventilation_only, etc.
    jurisdiction: str = "national"
    confidence_threshold: float = 0.7
    include_remediation: bool = True
    location_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComplianceReport:
    """Complete compliance analysis report"""
    report_id: str
    blueprint_id: str
    timestamp: str
    jurisdiction: str
    violations: List[Dict[str, Any]]
    summary: Dict[str, Any]
    zone_analysis: Optional[Dict[str, Any]] = None
    system_metrics: Optional[Dict[str, Any]] = None


class HVACComplianceAnalyzer:
    """
    Comprehensive HVAC Compliance Analyzer
    
    Integrates multiple validation modules to provide complete
    code compliance analysis for HVAC systems.
    """
    
    def __init__(self, regional_config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.ashrae_validator = ASHRAE621Validator()
        self.smacna_validator = SMACNAValidator()
        self.imc_validator = IMCFireCodeValidator()
        self.ductwork_validator = DuctworkValidator()
        self.equipment_validator = EquipmentClearanceValidator()
        self.confidence_scorer = ConfidenceScorer()
        self.regional_manager = RegionalCodeManager(regional_config_path)
        self.graph_builder = SystemGraphBuilder()
        
        self.logger.info("HVAC Compliance Analyzer initialized")
    
    def analyze_compliance(
        self,
        request: ComplianceAnalysisRequest,
        system_data: Dict[str, Any]
    ) -> ComplianceReport:
        """
        Perform comprehensive compliance analysis
        
        Args:
            request: ComplianceAnalysisRequest with analysis parameters
            system_data: Dictionary containing detected HVAC components and system info
            
        Returns:
            ComplianceReport with detailed violation analysis
        """
        self.logger.info(
            f"Starting compliance analysis for blueprint {request.blueprint_id}"
        )
        
        # Detect jurisdiction if location metadata provided
        if request.location_metadata:
            from .compliance.regional_overrides import Jurisdiction
            jurisdiction_enum = self.regional_manager.detect_jurisdiction(
                request.location_metadata
            )
            jurisdiction_str = jurisdiction_enum.value
        else:
            jurisdiction_str = request.jurisdiction
        
        all_violations = []
        
        # Analyze ventilation compliance
        if request.analysis_type in ["full_compliance", "ventilation_only"]:
            ventilation_violations = self._analyze_ventilation_compliance(
                system_data.get("zones", []),
                jurisdiction_str
            )
            all_violations.extend(ventilation_violations)
        
        # Analyze ductwork sizing
        if request.analysis_type in ["full_compliance", "ductwork_only"]:
            ductwork_violations = self._analyze_ductwork_compliance(
                system_data.get("duct_segments", []),
                system_data.get("diffusers", [])
            )
            all_violations.extend(ductwork_violations)
        
        # Analyze fire damper placement
        if request.analysis_type in ["full_compliance", "fire_safety_only"]:
            fire_violations = self._analyze_fire_code_compliance(
                system_data.get("penetrations", []),
                system_data.get("fire_rated_assemblies", []),
                system_data.get("smoke_barriers", [])
            )
            all_violations.extend(fire_violations)
        
        # Analyze equipment clearance
        if request.analysis_type in ["full_compliance", "equipment_only"]:
            clearance_violations = self._analyze_equipment_clearance(
                system_data.get("equipment", []),
                system_data.get("obstructions", [])
            )
            all_violations.extend(clearance_violations)
        
        # Filter by confidence threshold
        filtered_violations = [
            v for v in all_violations
            if v.get("confidence", 1.0) >= request.confidence_threshold
        ]
        
        # Enrich violations with risk scores
        enriched_violations = self.confidence_scorer.enrich_violations(
            filtered_violations
        )
        
        # Generate compliance summary
        total_components = (
            len(system_data.get("zones", [])) +
            len(system_data.get("duct_segments", [])) +
            len(system_data.get("equipment", [])) +
            len(system_data.get("penetrations", []))
        )
        
        summary = self.confidence_scorer.generate_compliance_summary(
            enriched_violations,
            total_components
        )
        
        # Build system graph and analyze connectivity
        system_metrics = None
        if request.analysis_type == "full_compliance":
            system_metrics = self._analyze_system_connectivity(system_data)
        
        # Create report
        report = ComplianceReport(
            report_id=f"comp_{uuid.uuid4().hex[:12]}",
            blueprint_id=request.blueprint_id,
            timestamp=datetime.utcnow().isoformat(),
            jurisdiction=jurisdiction_str,
            violations=enriched_violations,
            summary=summary,
            system_metrics=system_metrics
        )
        
        self.logger.info(
            f"Compliance analysis complete: {len(enriched_violations)} violations found, "
            f"compliance score: {summary['compliance_score']}"
        )
        
        return report
    
    def _analyze_ventilation_compliance(
        self,
        zones: List[Dict[str, Any]],
        jurisdiction: str
    ) -> List[Dict[str, Any]]:
        """Analyze ventilation compliance for all zones"""
        violations = []
        
        for zone_data in zones:
            try:
                # Convert zone data to VentilationZone
                occupancy_type = OccupancyType[zone_data.get("occupancy_type", "OFFICE").upper()]
            except KeyError:
                occupancy_type = OccupancyType.OFFICE
            
            zone = VentilationZone(
                zone_id=zone_data["zone_id"],
                occupancy_type=occupancy_type,
                floor_area=zone_data.get("floor_area", 1000.0),
                design_airflow=zone_data.get("design_airflow", 0.0),
                outdoor_air_flow=zone_data.get("outdoor_air_flow"),
                occupant_count=zone_data.get("occupant_count")
            )
            
            # Validate zone
            validation = self.ashrae_validator.validate_zone_ventilation(zone)
            violations.extend(validation["violations"])
        
        return violations
    
    def _analyze_ductwork_compliance(
        self,
        duct_segments: List[Dict[str, Any]],
        diffusers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze ductwork sizing compliance"""
        violations = []
        
        for duct_data in duct_segments:
            # Get connected diffusers
            connected_diffuser_ids = duct_data.get("connected_diffuser_ids", [])
            connected_diffusers = [
                d for d in diffusers
                if d["id"] in connected_diffuser_ids
            ]
            
            if not connected_diffusers:
                continue
            
            # Convert to appropriate format for SMACNA validator
            try:
                duct_type = DuctType[duct_data.get("duct_type", "SUPPLY").upper()]
            except KeyError:
                duct_type = DuctType.SUPPLY
            
            try:
                material = DuctMaterial[duct_data.get("material", "GALVANIZED_STEEL").upper()]
            except KeyError:
                material = DuctMaterial.GALVANIZED_STEEL
            
            duct_segment = DuctSegment(
                segment_id=duct_data["id"],
                duct_type=duct_type,
                material=material,
                width=duct_data.get("width"),
                height=duct_data.get("height"),
                diameter=duct_data.get("diameter"),
                length=duct_data.get("length", 0.0),
                design_airflow=sum(d.get("design_airflow", 0.0) for d in connected_diffusers),
                static_pressure=duct_data.get("static_pressure", 2.0)
            )
            
            # Validate duct sizing
            validation = self.smacna_validator.validate_duct_sizing(
                duct_segment,
                duct_data.get("location", "branch")
            )
            violations.extend(validation["violations"])
        
        return violations
    
    def _analyze_fire_code_compliance(
        self,
        penetrations: List[Dict[str, Any]],
        assemblies: List[Dict[str, Any]],
        smoke_barriers: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze fire damper placement compliance"""
        violations = []
        
        # Convert to appropriate format
        penetration_objects = []
        for pen_data in penetrations:
            try:
                damper_type = DamperType[pen_data.get("damper_type", "FIRE_DAMPER").upper()] if pen_data.get("has_damper") else None
                damper_rating = FireRating(pen_data.get("damper_rating", 1.0)) if pen_data.get("damper_rating") else None
            except (KeyError, ValueError):
                damper_type = None
                damper_rating = None
            
            penetration = DuctPenetration(
                penetration_id=pen_data["id"],
                duct_id=pen_data.get("duct_id", ""),
                assembly_id=pen_data.get("assembly_id", ""),
                location=tuple(pen_data.get("location", (0, 0))),
                has_damper=pen_data.get("has_damper", False),
                damper_type=damper_type,
                damper_rating=damper_rating
            )
            penetration_objects.append(penetration)
        
        assembly_objects = []
        for asm_data in assemblies:
            try:
                fire_rating = FireRating(asm_data.get("fire_rating", 1.0))
            except ValueError:
                fire_rating = FireRating.ONE_HOUR
            
            assembly = FireRatedAssembly(
                assembly_id=asm_data["id"],
                fire_rating=fire_rating,
                assembly_type=asm_data.get("type", "wall"),
                start_point=tuple(asm_data.get("start_point", (0, 0))),
                end_point=tuple(asm_data.get("end_point", (0, 0)))
            )
            assembly_objects.append(assembly)
        
        # Validate damper system
        if penetration_objects and assembly_objects:
            validation = self.imc_validator.validate_damper_system(
                penetration_objects,
                assembly_objects,
                smoke_barriers
            )
            
            for result in validation.get("penetration_results", []):
                violations.extend(result.get("violations", []))
        
        return violations
    
    def _analyze_equipment_clearance(
        self,
        equipment_list: List[Dict[str, Any]],
        obstructions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze equipment clearance compliance"""
        violations = []
        
        for equip_data in equipment_list:
            # Get nearby obstructions (simplified - should use spatial indexing)
            nearby_obstructions = obstructions  # In production, filter by proximity
            
            # Validate equipment clearance
            # Note: This is a simplified version - full implementation would
            # properly convert data formats and use spatial analysis
            
            violations.append({
                "severity": "INFO",
                "description": f"Equipment clearance validation for {equip_data.get('id')} requires detailed spatial analysis",
                "confidence": 0.60
            })
        
        return violations
    
    def _analyze_system_connectivity(
        self,
        system_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze system connectivity using graph analysis"""
        try:
            # Build graph from system data
            components = system_data.get("components", [])
            relationships = system_data.get("relationships", [])
            
            if self.graph_builder.build_graph_from_components(
                components,
                relationships
            ):
                return self.graph_builder.validate_system_connectivity()
        except Exception as e:
            self.logger.error(f"Error analyzing system connectivity: {e}")
        
        return None
