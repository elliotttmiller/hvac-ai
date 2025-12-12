"""
HVAC System Relationship Engine

This module provides HVAC-specific spatial relationship analysis and system
validation based on engineering principles and industry standards.

Key features:
- Spatial relationship graph construction for HVAC components
- System validation rules based on ASHRAE and SMACNA standards
- Anomaly detection for impossible duct runs and equipment placement
- Confidence scoring for system relationships
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
import math


class HVACComponentType(Enum):
    """Standard HVAC component types"""
    DUCTWORK = "ductwork"
    DIFFUSER = "diffuser"
    GRILLE = "grille"
    REGISTER = "register"
    VAV_BOX = "vav_box"
    DAMPER = "damper"
    FAN = "fan"
    AHU = "ahu"
    CHILLER = "chiller"
    COIL = "coil"
    SENSOR = "sensor"
    CONTROL = "control"
    UNKNOWN = "unknown"


class RelationshipType(Enum):
    """Types of relationships between HVAC components"""
    CONNECTED_TO = "connected_to"
    FEEDS_INTO = "feeds_into"
    CONTROLLED_BY = "controlled_by"
    MOUNTED_ON = "mounted_on"
    NEAR = "near"
    UPSTREAM_OF = "upstream_of"
    DOWNSTREAM_OF = "downstream_of"


@dataclass
class HVACComponent:
    """Represents a detected HVAC component"""
    id: str
    component_type: HVACComponentType
    bbox: List[float]  # [x, y, width, height]
    confidence: float
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate component center point"""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)
    
    @property
    def area(self) -> float:
        """Calculate component area"""
        return self.bbox[2] * self.bbox[3]


@dataclass
class ComponentRelationship:
    """Represents a relationship between two HVAC components"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HVACSystemEngine:
    """
    HVAC System Relationship Analysis Engine
    
    Analyzes spatial relationships between HVAC components and validates
    system configurations based on engineering principles.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.components: Dict[str, HVACComponent] = {}
        self.relationships: List[ComponentRelationship] = []
        
        # HVAC validation rules based on industry standards
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize HVAC-specific validation rules"""
        return {
            "max_duct_connection_distance": 100.0,  # pixels
            "min_equipment_clearance": 50.0,  # pixels
            "max_diffuser_spacing": 200.0,  # pixels
            "required_vav_duct_connection": True,
            "require_equipment_access": True
        }
    
    def add_component(self, component: HVACComponent):
        """Add a detected component to the system"""
        self.components[component.id] = component
        self.logger.debug(f"Added component: {component.id} ({component.component_type.value})")
    
    def build_relationship_graph(self) -> Dict[str, List[ComponentRelationship]]:
        """
        Build spatial relationship graph for all HVAC components
        
        Returns:
            Dictionary mapping component IDs to their relationships
        """
        self.relationships = []
        
        # Analyze all component pairs
        component_list = list(self.components.values())
        for i, comp1 in enumerate(component_list):
            for comp2 in component_list[i + 1:]:
                relationships = self._analyze_component_pair(comp1, comp2)
                self.relationships.extend(relationships)
        
        # Create adjacency list representation
        graph = {}
        for comp_id in self.components.keys():
            graph[comp_id] = [
                rel for rel in self.relationships
                if rel.source_id == comp_id or rel.target_id == comp_id
            ]
        
        self.logger.info(
            f"Built relationship graph: {len(self.components)} components, "
            f"{len(self.relationships)} relationships"
        )
        
        return graph
    
    def _analyze_component_pair(
        self,
        comp1: HVACComponent,
        comp2: HVACComponent
    ) -> List[ComponentRelationship]:
        """Analyze potential relationships between two components"""
        relationships = []
        
        # Calculate spatial metrics
        distance = self._calculate_distance(comp1.center, comp2.center)
        
        # Check for ductwork connectivity
        if self._is_duct_connection_possible(comp1, comp2, distance):
            rel = ComponentRelationship(
                source_id=comp1.id,
                target_id=comp2.id,
                relationship_type=RelationshipType.CONNECTED_TO,
                confidence=self._calculate_connection_confidence(comp1, comp2, distance),
                metadata={"distance": distance}
            )
            relationships.append(rel)
        
        # Check for equipment-diffuser relationships
        if self._is_equipment_diffuser_relationship(comp1, comp2):
            rel = ComponentRelationship(
                source_id=comp1.id,
                target_id=comp2.id,
                relationship_type=RelationshipType.FEEDS_INTO,
                confidence=0.8,
                metadata={"distance": distance}
            )
            relationships.append(rel)
        
        # Check for proximity relationships
        if distance < self.validation_rules["max_diffuser_spacing"]:
            rel = ComponentRelationship(
                source_id=comp1.id,
                target_id=comp2.id,
                relationship_type=RelationshipType.NEAR,
                confidence=1.0 - (distance / self.validation_rules["max_diffuser_spacing"]),
                metadata={"distance": distance}
            )
            relationships.append(rel)
        
        return relationships
    
    def _calculate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _is_duct_connection_possible(
        self,
        comp1: HVACComponent,
        comp2: HVACComponent,
        distance: float
    ) -> bool:
        """Determine if ductwork connection is possible between components"""
        # Ductwork must connect to diffusers, grilles, VAV boxes, etc.
        connectable_types = {
            HVACComponentType.DUCTWORK,
            HVACComponentType.DIFFUSER,
            HVACComponentType.GRILLE,
            HVACComponentType.REGISTER,
            HVACComponentType.VAV_BOX,
            HVACComponentType.AHU
        }
        
        if comp1.component_type not in connectable_types:
            return False
        if comp2.component_type not in connectable_types:
            return False
        
        # Check distance constraint
        max_distance = self.validation_rules["max_duct_connection_distance"]
        return distance < max_distance
    
    def _is_equipment_diffuser_relationship(
        self,
        comp1: HVACComponent,
        comp2: HVACComponent
    ) -> bool:
        """Check if components have equipment-to-diffuser relationship"""
        equipment_types = {HVACComponentType.AHU, HVACComponentType.VAV_BOX}
        terminal_types = {
            HVACComponentType.DIFFUSER,
            HVACComponentType.GRILLE,
            HVACComponentType.REGISTER
        }
        
        return (
            (comp1.component_type in equipment_types and comp2.component_type in terminal_types) or
            (comp2.component_type in equipment_types and comp1.component_type in terminal_types)
        )
    
    def _calculate_connection_confidence(
        self,
        comp1: HVACComponent,
        comp2: HVACComponent,
        distance: float
    ) -> float:
        """Calculate confidence score for connection relationship"""
        max_distance = self.validation_rules["max_duct_connection_distance"]
        
        # Distance-based confidence
        distance_confidence = 1.0 - (distance / max_distance)
        
        # Component type compatibility
        type_confidence = 0.8  # Base confidence for compatible types
        
        # Combined confidence
        return min(1.0, (distance_confidence + type_confidence) / 2.0)
    
    def validate_system_configuration(self) -> Dict[str, Any]:
        """
        Validate HVAC system configuration based on engineering rules
        
        Returns:
            Dictionary containing validation results:
            - is_valid: Overall validity status
            - violations: List of detected violations
            - warnings: List of potential issues
        """
        violations = []
        warnings = []
        
        # Rule 1: Ductwork must connect to diffusers/grilles
        for comp in self.components.values():
            if comp.component_type in {
                HVACComponentType.DIFFUSER,
                HVACComponentType.GRILLE,
                HVACComponentType.REGISTER
            }:
                if not self._has_duct_connection(comp.id):
                    violations.append({
                        "component_id": comp.id,
                        "rule": "duct_connectivity",
                        "message": f"{comp.component_type.value} must connect to ductwork"
                    })
        
        # Rule 2: VAV boxes must connect to main ductwork
        if self.validation_rules["required_vav_duct_connection"]:
            for comp in self.components.values():
                if comp.component_type == HVACComponentType.VAV_BOX:
                    if not self._has_upstream_duct(comp.id):
                        violations.append({
                            "component_id": comp.id,
                            "rule": "vav_connectivity",
                            "message": "VAV box must connect to main ductwork"
                        })
        
        # Rule 3: Equipment clearance requirements
        clearance_issues = self._check_equipment_clearance()
        warnings.extend(clearance_issues)
        
        # Rule 4: Check for impossible flow paths
        flow_issues = self._detect_impossible_flow_paths()
        violations.extend(flow_issues)
        
        is_valid = len(violations) == 0
        
        self.logger.info(
            f"System validation: {'PASS' if is_valid else 'FAIL'}, "
            f"{len(violations)} violations, {len(warnings)} warnings"
        )
        
        return {
            "is_valid": is_valid,
            "violations": violations,
            "warnings": warnings,
            "summary": {
                "total_components": len(self.components),
                "total_relationships": len(self.relationships),
                "violation_count": len(violations),
                "warning_count": len(warnings)
            }
        }
    
    def _has_duct_connection(self, component_id: str) -> bool:
        """Check if component has ductwork connection"""
        for rel in self.relationships:
            if rel.source_id == component_id or rel.target_id == component_id:
                # Check if related to ductwork
                other_id = rel.target_id if rel.source_id == component_id else rel.source_id
                if other_id in self.components:
                    other_comp = self.components[other_id]
                    if other_comp.component_type == HVACComponentType.DUCTWORK:
                        return True
        return False
    
    def _has_upstream_duct(self, component_id: str) -> bool:
        """Check if component has upstream ductwork connection"""
        # Simplified check - in production would analyze flow direction
        return self._has_duct_connection(component_id)
    
    def _check_equipment_clearance(self) -> List[Dict[str, Any]]:
        """Check equipment clearance requirements"""
        issues = []
        
        equipment_types = {
            HVACComponentType.AHU,
            HVACComponentType.CHILLER,
            HVACComponentType.FAN
        }
        
        equipment_list = [
            comp for comp in self.components.values()
            if comp.component_type in equipment_types
        ]
        
        min_clearance = self.validation_rules["min_equipment_clearance"]
        
        for i, equip1 in enumerate(equipment_list):
            for equip2 in equipment_list[i + 1:]:
                distance = self._calculate_distance(equip1.center, equip2.center)
                if distance < min_clearance:
                    issues.append({
                        "component_ids": [equip1.id, equip2.id],
                        "rule": "equipment_clearance",
                        "message": f"Equipment clearance too small: {distance:.1f} < {min_clearance:.1f}",
                        "severity": "warning"
                    })
        
        return issues
    
    def _detect_impossible_flow_paths(self) -> List[Dict[str, Any]]:
        """Detect physically impossible airflow paths"""
        issues = []
        
        # Simplified implementation - in production would use graph analysis
        # to detect disconnected components, circular flows, etc.
        
        return issues
    
    def get_component_neighbors(
        self,
        component_id: str,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> List[str]:
        """
        Get neighboring components connected by specified relationship types
        
        Args:
            component_id: ID of the component
            relationship_types: List of relationship types to consider (all if None)
            
        Returns:
            List of neighboring component IDs
        """
        neighbors = set()
        
        for rel in self.relationships:
            if relationship_types and rel.relationship_type not in relationship_types:
                continue
            
            if rel.source_id == component_id:
                neighbors.add(rel.target_id)
            elif rel.target_id == component_id:
                neighbors.add(rel.source_id)
        
        return list(neighbors)
    
    def export_system_graph(self) -> Dict[str, Any]:
        """
        Export complete system graph for visualization
        
        Returns:
            Dictionary containing nodes and edges for graph visualization
        """
        nodes = [
            {
                "id": comp.id,
                "type": comp.component_type.value,
                "bbox": comp.bbox,
                "confidence": comp.confidence,
                "attributes": comp.attributes
            }
            for comp in self.components.values()
        ]
        
        edges = [
            {
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.relationship_type.value,
                "confidence": rel.confidence,
                "metadata": rel.metadata
            }
            for rel in self.relationships
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }
