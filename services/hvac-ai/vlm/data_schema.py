"""
HVAC Data Schema for VLM Training

Defines the component taxonomy, attributes, and relationships for HVAC systems.
This schema guides synthetic data generation and model training.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class HVACComponentType(Enum):
    """HVAC component types for detection and classification"""
    
    # Ductwork
    SUPPLY_AIR_DUCT = "supply_air_duct"
    RETURN_AIR_DUCT = "return_air_duct"
    EXHAUST_DUCT = "exhaust_duct"
    FLEXIBLE_DUCT = "flexible_duct"
    TRANSFER_DUCT = "transfer_duct"
    
    # Equipment
    AHU = "ahu"  # Air Handling Unit
    RTU = "rtu"  # Rooftop Unit
    VAV = "vav"  # Variable Air Volume Box
    FAN = "fan"
    CHILLER = "chiller"
    BOILER = "boiler"
    HEAT_PUMP = "heat_pump"
    
    # Controls
    DAMPER = "damper"
    VALVE = "valve"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    THERMOSTAT = "thermostat"
    
    # Terminals
    DIFFUSER = "diffuser"
    GRILLE = "grille"
    REGISTER = "register"
    
    # Coils
    HEATING_COIL = "heating_coil"
    COOLING_COIL = "cooling_coil"
    PREHEAT_COIL = "preheat_coil"
    
    # Accessories
    FILTER = "filter"
    HUMIDIFIER = "humidifier"
    DEHUMIDIFIER = "dehumidifier"
    SILENCER = "silencer"


class RelationshipType(Enum):
    """Types of relationships between HVAC components"""
    CONNECTS_TO = "connects_to"
    FEEDS = "feeds"
    RETURNS_FROM = "returns_from"
    CONTROLS = "controls"
    SUPPLIES = "supplies"
    EXHAUSTS = "exhausts"
    PART_OF = "part_of"


@dataclass
class ComponentAttributes:
    """Attributes for HVAC components"""
    size: Optional[str] = None  # e.g., "12x10", "8\""
    material: Optional[str] = None  # e.g., "galvanized_steel", "copper"
    cfm: Optional[int] = None  # Cubic Feet per Minute
    pressure: Optional[float] = None  # Pressure rating
    temperature: Optional[float] = None  # Temperature rating
    capacity: Optional[float] = None  # Capacity (BTU, tons, etc.)
    efficiency: Optional[float] = None  # Efficiency rating
    designation: Optional[str] = None  # Tag/label (e.g., "SD-1", "VAV-101")
    notes: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComponentAnnotation:
    """Annotation for a single HVAC component"""
    component_id: str
    component_type: HVACComponentType
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    polygon: Optional[List[Tuple[int, int]]] = None
    attributes: ComponentAttributes = field(default_factory=ComponentAttributes)
    relationships: List[Dict] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class DrawingMetadata:
    """Metadata for HVAC drawings"""
    drawing_type: str  # e.g., "supply_air_plan", "return_air_plan"
    system_type: str  # e.g., "commercial", "residential", "industrial"
    complexity: str  # "simple", "medium", "complex"
    resolution: str  # e.g., "300dpi"
    scale: Optional[str] = None  # e.g., "1/4\" = 1'-0\""
    building_type: Optional[str] = None
    floor_level: Optional[str] = None
    date: Optional[str] = None
    project_name: Optional[str] = None


@dataclass
class HVACTrainingExample:
    """Complete training example for VLM"""
    image_id: str
    image_path: str
    metadata: DrawingMetadata
    annotations: List[ComponentAnnotation]
    prompts: List[Dict[str, str]]


class HVACDataSchema:
    """HVAC data schema definition and validation"""
    
    # Component type to typical attributes mapping
    COMPONENT_ATTRIBUTES = {
        HVACComponentType.SUPPLY_AIR_DUCT: {
            "size": ["4x4", "6x6", "8x8", "10x8", "12x10", "14x10", "16x12", "20x12"],
            "material": ["galvanized_steel", "aluminum", "flexible"],
            "cfm_range": (50, 10000)
        },
        HVACComponentType.RETURN_AIR_DUCT: {
            "size": ["12x10", "14x10", "16x12", "20x12", "24x14"],
            "material": ["galvanized_steel", "aluminum"],
            "cfm_range": (100, 15000)
        },
        HVACComponentType.VAV: {
            "size": ["6\"", "8\"", "10\"", "12\"", "14\""],
            "cfm_range": (200, 4000),
            "designation_pattern": "VAV-[0-9]{3}"
        },
        HVACComponentType.DIFFUSER: {
            "size": ["6x6", "8x8", "10x10", "12x12", "24x24"],
            "cfm_range": (50, 1000)
        },
        HVACComponentType.DAMPER: {
            "size": ["6\"", "8\"", "10\"", "12\"", "14\"", "16\""],
            "types": ["manual", "motorized", "fire", "smoke", "combination"]
        }
    }
    
    # Valid relationships between component types
    VALID_RELATIONSHIPS = {
        (HVACComponentType.AHU, HVACComponentType.SUPPLY_AIR_DUCT): [RelationshipType.SUPPLIES],
        (HVACComponentType.SUPPLY_AIR_DUCT, HVACComponentType.VAV): [RelationshipType.FEEDS],
        (HVACComponentType.VAV, HVACComponentType.DIFFUSER): [RelationshipType.SUPPLIES],
        (HVACComponentType.RETURN_AIR_DUCT, HVACComponentType.AHU): [RelationshipType.RETURNS_FROM],
        (HVACComponentType.DAMPER, HVACComponentType.SUPPLY_AIR_DUCT): [RelationshipType.CONTROLS],
        (HVACComponentType.SENSOR, HVACComponentType.VAV): [RelationshipType.CONTROLS],
    }
    
    # Engineering constraints (ASHRAE/SMACNA standards)
    ENGINEERING_RULES = {
        "supply_exhaust_separation": {
            "rule": "Supply air ducts cannot connect to exhaust systems",
            "severity": "critical"
        },
        "cfm_balance": {
            "rule": "Supply and return CFM must balance within 10%",
            "severity": "warning",
            "tolerance": 0.10
        },
        "minimum_clearances": {
            "rule": "Ductwork must maintain minimum clearances per SMACNA",
            "severity": "warning",
            "clearance": 2.0  # inches
        },
        "maximum_velocity": {
            "rule": "Supply duct velocity should not exceed 2500 FPM",
            "severity": "warning",
            "max_velocity": 2500  # feet per minute
        },
        "pressure_drop": {
            "rule": "Total system pressure drop should not exceed 2.5\" w.g.",
            "severity": "warning",
            "max_pressure": 2.5  # inches water gauge
        }
    }
    
    @staticmethod
    def validate_component_type(component_type: str) -> bool:
        """Validate if component type is recognized"""
        try:
            HVACComponentType(component_type)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_relationship(
        source_type: HVACComponentType,
        target_type: HVACComponentType,
        relationship_type: RelationshipType
    ) -> bool:
        """Validate if relationship between components is valid"""
        valid_rels = HVACDataSchema.VALID_RELATIONSHIPS.get((source_type, target_type), [])
        return relationship_type in valid_rels
    
    @staticmethod
    def get_typical_attributes(component_type: HVACComponentType) -> Dict:
        """Get typical attributes for a component type"""
        return HVACDataSchema.COMPONENT_ATTRIBUTES.get(component_type, {})
    
    @staticmethod
    def validate_cfm_range(component_type: HVACComponentType, cfm: int) -> bool:
        """Validate if CFM is within typical range for component type"""
        attrs = HVACDataSchema.get_typical_attributes(component_type)
        if "cfm_range" in attrs:
            min_cfm, max_cfm = attrs["cfm_range"]
            return min_cfm <= cfm <= max_cfm
        return True  # No validation if range not defined
    
    @staticmethod
    def get_prompt_templates() -> Dict[str, str]:
        """Get VLM prompt templates for HVAC analysis"""
        return {
            "component_detection": """You are an expert HVAC engineer analyzing a blueprint.

Task: Identify all HVAC components in this drawing.

For each component, provide:
1. Component type (e.g., supply duct, VAV box, damper)
2. Location (bounding box coordinates)
3. Engineering specifications (size, CFM, material)
4. Designation/tag (e.g., SD-1, VAV-101)

Output as structured JSON in this format:
{
  "components": [
    {
      "type": "supply_air_duct",
      "bbox": [x1, y1, x2, y2],
      "attributes": {
        "size": "12x10",
        "cfm": 2000,
        "designation": "SD-1"
      }
    }
  ]
}""",
            
            "relationship_analysis": """You are an expert HVAC engineer analyzing system connectivity.

Task: Analyze the airflow relationships in this drawing.

For each connection:
1. Source component and its designation
2. Target component and its designation  
3. Connection type (supply, return, exhaust)
4. Flow direction
5. Validate against ASHRAE standards

Output as structured JSON:
{
  "relationships": [
    {
      "source": "AHU-1",
      "target": "SD-1",
      "type": "supplies",
      "valid": true,
      "notes": "Complies with ASHRAE 62.1"
    }
  ],
  "violations": []
}""",
            
            "specification_extraction": """You are an expert HVAC engineer extracting specifications.

Task: Extract all engineering specifications from this drawing.

Extract:
1. Component sizes and dimensions
2. CFM values and flow rates
3. Material specifications
4. Pressure ratings
5. Temperature ratings
6. Notes and special requirements

Output as structured JSON with component designations as keys.""",
            
            "code_compliance": """You are an expert HVAC engineer performing code compliance review.

Task: Review this HVAC design for compliance with:
- ASHRAE Standard 62.1 (Ventilation)
- ASHRAE Standard 90.1 (Energy)
- SMACNA standards (Ductwork)
- Local building codes

Identify:
1. Code violations (with reference)
2. Design issues
3. Missing components or information
4. Recommendations for improvement

Output as structured JSON with severity levels."""
        }


# Export component type mappings for convenience
DUCTWORK_TYPES = [
    HVACComponentType.SUPPLY_AIR_DUCT,
    HVACComponentType.RETURN_AIR_DUCT,
    HVACComponentType.EXHAUST_DUCT,
    HVACComponentType.FLEXIBLE_DUCT,
]

EQUIPMENT_TYPES = [
    HVACComponentType.AHU,
    HVACComponentType.RTU,
    HVACComponentType.VAV,
    HVACComponentType.FAN,
    HVACComponentType.CHILLER,
    HVACComponentType.BOILER,
]

CONTROL_TYPES = [
    HVACComponentType.DAMPER,
    HVACComponentType.VALVE,
    HVACComponentType.SENSOR,
    HVACComponentType.ACTUATOR,
    HVACComponentType.THERMOSTAT,
]
