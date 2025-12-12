"""
Unit tests for HVAC System Relationship Engine
"""

import pytest
import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from services.hvac_domain.hvac_system_engine import (
    HVACSystemEngine,
    HVACComponent,
    HVACComponentType,
    ComponentRelationship,
    RelationshipType
)


class TestHVACComponent:
    """Tests for HVAC Component data class"""
    
    def test_component_creation(self):
        """Test basic component creation"""
        component = HVACComponent(
            id="comp_001",
            component_type=HVACComponentType.DIFFUSER,
            bbox=[100.0, 200.0, 50.0, 50.0],
            confidence=0.95
        )
        
        assert component.id == "comp_001"
        assert component.component_type == HVACComponentType.DIFFUSER
        assert component.confidence == 0.95
    
    def test_component_center_calculation(self):
        """Test center point calculation"""
        component = HVACComponent(
            id="comp_001",
            component_type=HVACComponentType.DUCTWORK,
            bbox=[100.0, 200.0, 50.0, 60.0],
            confidence=0.9
        )
        
        center = component.center
        assert center == (125.0, 230.0)  # (100 + 50/2, 200 + 60/2)
    
    def test_component_area_calculation(self):
        """Test area calculation"""
        component = HVACComponent(
            id="comp_001",
            component_type=HVACComponentType.VAV_BOX,
            bbox=[0.0, 0.0, 100.0, 50.0],
            confidence=0.85
        )
        
        assert component.area == 5000.0  # 100 * 50
    
    def test_component_with_attributes(self):
        """Test component with custom attributes"""
        component = HVACComponent(
            id="comp_001",
            component_type=HVACComponentType.DIFFUSER,
            bbox=[0, 0, 10, 10],
            confidence=0.9,
            attributes={"airflow": "200 CFM", "size": "12x12"}
        )
        
        assert component.attributes["airflow"] == "200 CFM"
        assert component.attributes["size"] == "12x12"


class TestComponentRelationship:
    """Tests for Component Relationship data class"""
    
    def test_relationship_creation(self):
        """Test basic relationship creation"""
        rel = ComponentRelationship(
            source_id="comp_001",
            target_id="comp_002",
            relationship_type=RelationshipType.CONNECTED_TO,
            confidence=0.9
        )
        
        assert rel.source_id == "comp_001"
        assert rel.target_id == "comp_002"
        assert rel.relationship_type == RelationshipType.CONNECTED_TO
        assert rel.confidence == 0.9
    
    def test_relationship_with_metadata(self):
        """Test relationship with metadata"""
        rel = ComponentRelationship(
            source_id="comp_001",
            target_id="comp_002",
            relationship_type=RelationshipType.FEEDS_INTO,
            confidence=0.85,
            metadata={"distance": 50.0, "flow_direction": "supply"}
        )
        
        assert rel.metadata["distance"] == 50.0
        assert rel.metadata["flow_direction"] == "supply"


class TestHVACSystemEngine:
    """Tests for HVAC System Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create a fresh engine for each test"""
        return HVACSystemEngine()
    
    @pytest.fixture
    def sample_components(self):
        """Create sample HVAC components for testing"""
        return [
            HVACComponent(
                id="duct_001",
                component_type=HVACComponentType.DUCTWORK,
                bbox=[100.0, 100.0, 200.0, 50.0],
                confidence=0.95
            ),
            HVACComponent(
                id="diffuser_001",
                component_type=HVACComponentType.DIFFUSER,
                bbox=[250.0, 100.0, 30.0, 30.0],
                confidence=0.90
            ),
            HVACComponent(
                id="vav_001",
                component_type=HVACComponentType.VAV_BOX,
                bbox=[50.0, 100.0, 40.0, 40.0],
                confidence=0.88
            )
        ]
    
    def test_add_component(self, engine):
        """Test adding components to engine"""
        component = HVACComponent(
            id="test_001",
            component_type=HVACComponentType.DIFFUSER,
            bbox=[0, 0, 10, 10],
            confidence=0.9
        )
        
        engine.add_component(component)
        
        assert "test_001" in engine.components
        assert engine.components["test_001"] == component
    
    def test_build_relationship_graph_empty(self, engine):
        """Test building graph with no components"""
        graph = engine.build_relationship_graph()
        
        assert len(graph) == 0
        assert len(engine.relationships) == 0
    
    def test_build_relationship_graph_single_component(self, engine):
        """Test building graph with single component"""
        component = HVACComponent(
            id="comp_001",
            component_type=HVACComponentType.DUCTWORK,
            bbox=[0, 0, 100, 50],
            confidence=0.9
        )
        engine.add_component(component)
        
        graph = engine.build_relationship_graph()
        
        assert "comp_001" in graph
        # Single component should have no relationships
        assert len(graph["comp_001"]) == 0
    
    def test_build_relationship_graph_connected_components(self, engine, sample_components):
        """Test building graph with connected components"""
        for comp in sample_components:
            engine.add_component(comp)
        
        graph = engine.build_relationship_graph()
        
        # Should have entries for all components
        assert len(graph) == 3
        
        # Should have found some relationships
        assert len(engine.relationships) > 0
    
    def test_distance_calculation(self, engine):
        """Test distance calculation between points"""
        point1 = (0.0, 0.0)
        point2 = (3.0, 4.0)
        
        distance = engine._calculate_distance(point1, point2)
        
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_duct_connection_possible_valid(self, engine):
        """Test valid duct connection detection"""
        comp1 = HVACComponent(
            id="duct",
            component_type=HVACComponentType.DUCTWORK,
            bbox=[0, 0, 100, 50],
            confidence=0.9
        )
        comp2 = HVACComponent(
            id="diffuser",
            component_type=HVACComponentType.DIFFUSER,
            bbox=[90, 0, 30, 30],
            confidence=0.9
        )
        
        distance = engine._calculate_distance(comp1.center, comp2.center)
        possible = engine._is_duct_connection_possible(comp1, comp2, distance)
        
        assert possible is True
    
    def test_duct_connection_possible_invalid_type(self, engine):
        """Test duct connection with invalid component types"""
        comp1 = HVACComponent(
            id="sensor",
            component_type=HVACComponentType.SENSOR,
            bbox=[0, 0, 10, 10],
            confidence=0.9
        )
        comp2 = HVACComponent(
            id="control",
            component_type=HVACComponentType.CONTROL,
            bbox=[20, 0, 10, 10],
            confidence=0.9
        )
        
        distance = engine._calculate_distance(comp1.center, comp2.center)
        possible = engine._is_duct_connection_possible(comp1, comp2, distance)
        
        assert possible is False
    
    def test_validate_system_configuration_empty(self, engine):
        """Test validation with no components"""
        validation = engine.validate_system_configuration()
        
        assert validation['is_valid'] is True
        assert len(validation['violations']) == 0
        assert 'summary' in validation
    
    def test_validate_system_configuration_with_violations(self, engine):
        """Test validation with components that should have violations"""
        # Add diffuser without ductwork connection
        diffuser = HVACComponent(
            id="diffuser_001",
            component_type=HVACComponentType.DIFFUSER,
            bbox=[1000.0, 1000.0, 30.0, 30.0],  # Far from any ductwork
            confidence=0.9
        )
        engine.add_component(diffuser)
        
        # Build relationships (none should be found due to distance)
        engine.build_relationship_graph()
        
        validation = engine.validate_system_configuration()
        
        # Should find violation: diffuser not connected to ductwork
        assert validation['is_valid'] is False
        assert len(validation['violations']) > 0
    
    def test_get_component_neighbors(self, engine, sample_components):
        """Test getting neighboring components"""
        for comp in sample_components:
            engine.add_component(comp)
        
        engine.build_relationship_graph()
        
        # Get neighbors of ductwork component
        neighbors = engine.get_component_neighbors("duct_001")
        
        # Should have neighbors if relationships were found
        assert isinstance(neighbors, list)
    
    def test_export_system_graph(self, engine, sample_components):
        """Test exporting system graph"""
        for comp in sample_components:
            engine.add_component(comp)
        
        engine.build_relationship_graph()
        
        graph_export = engine.export_system_graph()
        
        assert 'nodes' in graph_export
        assert 'edges' in graph_export
        assert 'metadata' in graph_export
        
        assert len(graph_export['nodes']) == 3
        assert graph_export['metadata']['node_count'] == 3


class TestValidationRules:
    """Tests for HVAC validation rules"""
    
    def test_equipment_clearance_check(self):
        """Test equipment clearance validation"""
        engine = HVACSystemEngine()
        
        # Add two equipment items too close together
        ahu1 = HVACComponent(
            id="ahu_001",
            component_type=HVACComponentType.AHU,
            bbox=[100.0, 100.0, 80.0, 80.0],
            confidence=0.9
        )
        ahu2 = HVACComponent(
            id="ahu_002",
            component_type=HVACComponentType.AHU,
            bbox=[150.0, 100.0, 80.0, 80.0],  # Very close
            confidence=0.9
        )
        
        engine.add_component(ahu1)
        engine.add_component(ahu2)
        engine.build_relationship_graph()
        
        validation = engine.validate_system_configuration()
        
        # Should have warnings about clearance
        assert len(validation['warnings']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
