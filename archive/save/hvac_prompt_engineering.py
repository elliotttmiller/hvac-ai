"""
HVAC Prompt Engineering Framework

This module implements HVAC-specific prompt engineering patterns based on
ASHRAE and SMACNA standards for improved AI analysis accuracy.

Key features:
- ASHRAE/SMACNA standard prompt templates
- Chain-of-thought prompting for HVAC analysis
- Few-shot learning with HVAC examples
- Contextual prompt management and versioning
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class HVACAnalysisType(Enum):
    """Types of HVAC analysis"""
    COMPONENT_DETECTION = "component_detection"
    DUCT_CONNECTIVITY = "duct_connectivity"
    EQUIPMENT_SIZING = "equipment_sizing"
    CODE_COMPLIANCE = "code_compliance"
    SYSTEM_LAYOUT = "system_layout"
    SYMBOL_RECOGNITION = "symbol_recognition"


class PromptStrategy(Enum):
    """Prompt engineering strategies"""
    DIRECT = "direct"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    ROLE_ASSIGNMENT = "role_assignment"
    CONSTRAINT_SPECIFICATION = "constraint_specification"


@dataclass
class PromptTemplate:
    """Template for HVAC-specific prompts"""
    name: str
    analysis_type: HVACAnalysisType
    strategy: PromptStrategy
    template: str
    variables: List[str]
    version: str = "1.0"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HVACPromptEngineeringFramework:
    """
    Professional HVAC Prompt Engineering Framework
    
    Implements industry-standard prompt patterns for HVAC blueprint analysis
    with ASHRAE and SMACNA compliance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_standard_templates()
    
    def _initialize_standard_templates(self):
        """Initialize ASHRAE/SMACNA standard prompt templates"""
        
        # Component Detection with Chain-of-Thought
        self.register_template(PromptTemplate(
            name="component_detection_cot",
            analysis_type=HVACAnalysisType.COMPONENT_DETECTION,
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            template="""You are an ASHRAE-certified HVAC engineer analyzing this blueprint.

Task: Identify HVAC components in the provided blueprint image.

Analysis Process:
1. First, identify the main ductwork runs and their orientations
2. Then, locate branch connections and terminals (diffusers, grilles, registers)
3. Next, identify major equipment (AHUs, VAV boxes, fans)
4. Finally, note any control devices (dampers, sensors, actuators)

Context: {context}
Blueprint Type: {blueprint_type}

Provide your analysis following the steps above, then list all identified components with their locations and confidence scores.""",
            variables=["context", "blueprint_type"],
            version="1.0"
        ))
        
        # Duct Connectivity Analysis
        self.register_template(PromptTemplate(
            name="duct_connectivity_analysis",
            analysis_type=HVACAnalysisType.DUCT_CONNECTIVITY,
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            template="""You are an SMACNA-certified ductwork installer analyzing this HVAC system layout.

Task: Analyze ductwork connectivity and identify the complete airflow path.

Step-by-step analysis:
1. Identify the primary air handling unit or supply source
2. Trace the main supply ductwork from the source
3. Map all branch takeoffs and their directions
4. Locate terminal devices (diffusers, grilles) connected to branches
5. Verify return air pathways if visible

System Type: {system_type}
Building Type: {building_type}

Provide a detailed connectivity map showing the airflow from source to all terminal devices.""",
            variables=["system_type", "building_type"],
            version="1.0"
        ))
        
        # Few-Shot Symbol Recognition
        self.register_template(PromptTemplate(
            name="symbol_recognition_fewshot",
            analysis_type=HVACAnalysisType.SYMBOL_RECOGNITION,
            strategy=PromptStrategy.FEW_SHOT,
            template="""You are an expert in ASHRAE and SMACNA HVAC symbols.

Task: Identify HVAC symbols in this blueprint.

Examples of standard HVAC symbols:
1. Supply Diffuser: Square or circular symbol with arrows pointing outward
2. Return Grille: Square symbol with arrows pointing inward
3. VAV Box: Rectangular box with "VAV" label and control lines
4. Damper: Line crossing duct with control indicator
5. Fire Damper: Line crossing duct with "FD" designation

Now, analyze the provided blueprint and identify all HVAC symbols you can find.
For each symbol, provide:
- Symbol type
- Location (x, y coordinates)
- Confidence score (0-1)
- Any visible labels or specifications

Blueprint Section: {section_type}""",
            variables=["section_type"],
            version="1.0"
        ))
        
        # Role-Based Equipment Sizing
        self.register_template(PromptTemplate(
            name="equipment_sizing_role",
            analysis_type=HVACAnalysisType.EQUIPMENT_SIZING,
            strategy=PromptStrategy.ROLE_ASSIGNMENT,
            template="""You are a professional HVAC design engineer specializing in equipment selection and sizing.

Task: Analyze equipment sizing and verify appropriateness for the space.

Your expertise includes:
- ASHRAE Standard 62.1 ventilation requirements
- Load calculation methodologies
- Equipment capacity verification
- Duct sizing per SMACNA standards

Space Information:
- Area: {space_area} sq ft
- Occupancy: {occupancy_type}
- Special Requirements: {special_requirements}

Analyze the equipment shown in the blueprint and provide:
1. Equipment type and nominal capacity
2. Verification of sizing adequacy
3. Any concerns or recommendations
4. Code compliance notes""",
            variables=["space_area", "occupancy_type", "special_requirements"],
            version="1.0"
        ))
        
        # Constraint-Based Code Compliance
        self.register_template(PromptTemplate(
            name="code_compliance_constraints",
            analysis_type=HVACAnalysisType.CODE_COMPLIANCE,
            strategy=PromptStrategy.CONSTRAINT_SPECIFICATION,
            template="""You are reviewing HVAC plans for code compliance.

Task: Verify compliance with ASHRAE and local codes.

Constraints and Requirements:
- MUST comply with ASHRAE Standard 62.1 for ventilation
- MUST meet SMACNA installation standards for ductwork
- MUST have proper equipment clearances per manufacturer specs
- MUST have fire dampers at required locations
- MUST have adequate return air pathways

Jurisdiction: {jurisdiction}
Building Code: {building_code}
Special Requirements: {special_requirements}

Review the HVAC layout and report:
1. All code compliance issues found
2. Locations of violations
3. Required corrections
4. Advisory recommendations

Only report issues that clearly violate the stated requirements.""",
            variables=["jurisdiction", "building_code", "special_requirements"],
            version="1.0"
        ))
        
        # System Layout Analysis
        self.register_template(PromptTemplate(
            name="system_layout_comprehensive",
            analysis_type=HVACAnalysisType.SYSTEM_LAYOUT,
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            template="""You are an experienced HVAC system designer reviewing this layout.

Task: Provide comprehensive system layout analysis.

Analysis Framework:
1. System Type Identification
   - Determine if VAV, constant volume, or hybrid system
   - Identify zoning strategy

2. Equipment Assessment
   - List all major equipment (AHUs, chillers, boilers, etc.)
   - Verify equipment placement and accessibility

3. Distribution Analysis
   - Evaluate ductwork layout efficiency
   - Assess terminal device distribution
   - Check for balanced air distribution

4. Controls and Automation
   - Identify control devices and strategies
   - Verify sensor placements

5. Compliance and Best Practices
   - ASHRAE standard compliance
   - Energy efficiency considerations
   - Maintenance accessibility

Project Details:
- Building Type: {building_type}
- Square Footage: {square_footage}
- Number of Zones: {zone_count}

Provide a detailed analysis following the framework above.""",
            variables=["building_type", "square_footage", "zone_count"],
            version="1.0"
        ))
        
        self.logger.info(f"Initialized {len(self.templates)} standard HVAC prompt templates")
    
    def register_template(self, template: PromptTemplate):
        """Register a new prompt template"""
        self.templates[template.name] = template
        self.logger.debug(f"Registered prompt template: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Retrieve a prompt template by name"""
        return self.templates.get(name)
    
    def generate_prompt(
        self,
        template_name: str,
        variables: Dict[str, Any],
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate a complete prompt from template and variables
        
        Args:
            template_name: Name of the template to use
            variables: Dictionary of variable values
            additional_context: Optional additional context to append
            
        Returns:
            Formatted prompt string
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Validate required variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            self.logger.warning(f"Missing variables: {missing_vars}, using defaults")
            for var in missing_vars:
                variables[var] = f"[{var}]"
        
        # Format the prompt
        try:
            prompt = template.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Error formatting template: {e}")
        
        # Add additional context if provided
        if additional_context:
            prompt += f"\n\nAdditional Context:\n{additional_context}"
        
        return prompt
    
    def get_prompt_for_analysis_type(
        self,
        analysis_type: HVACAnalysisType,
        strategy: Optional[PromptStrategy] = None
    ) -> List[PromptTemplate]:
        """
        Get all templates for a specific analysis type
        
        Args:
            analysis_type: Type of HVAC analysis
            strategy: Optional filter by prompt strategy
            
        Returns:
            List of matching templates
        """
        templates = [
            t for t in self.templates.values()
            if t.analysis_type == analysis_type
        ]
        
        if strategy:
            templates = [t for t in templates if t.strategy == strategy]
        
        return templates
    
    def optimize_prompt(
        self,
        template_name: str,
        feedback: Dict[str, Any]
    ) -> PromptTemplate:
        """
        Optimize prompt template based on performance feedback
        
        Args:
            template_name: Name of template to optimize
            feedback: Performance metrics and feedback
            
        Returns:
            Updated template
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Store feedback in metadata
        if "optimization_history" not in template.metadata:
            template.metadata["optimization_history"] = []
        
        template.metadata["optimization_history"].append(feedback)
        
        # In production, this would use ML-based optimization
        # For now, just log the feedback
        self.logger.info(
            f"Recorded feedback for template '{template_name}': "
            f"accuracy={feedback.get('accuracy', 'N/A')}"
        )
        
        return template
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available prompt templates
        
        Returns:
            List of template summaries
        """
        return [
            {
                "name": t.name,
                "analysis_type": t.analysis_type.value,
                "strategy": t.strategy.value,
                "version": t.version,
                "variables": t.variables
            }
            for t in self.templates.values()
        ]
    
    def export_template_library(self) -> Dict[str, Any]:
        """
        Export complete template library for documentation or backup
        
        Returns:
            Dictionary containing all templates and metadata
        """
        return {
            "version": "1.0",
            "templates": {
                name: {
                    "analysis_type": t.analysis_type.value,
                    "strategy": t.strategy.value,
                    "template": t.template,
                    "variables": t.variables,
                    "version": t.version,
                    "metadata": t.metadata
                }
                for name, t in self.templates.items()
            },
            "metadata": {
                "total_templates": len(self.templates),
                "analysis_types": list(set(t.analysis_type.value for t in self.templates.values())),
                "strategies": list(set(t.strategy.value for t in self.templates.values()))
            }
        }


def create_hvac_prompt_framework() -> HVACPromptEngineeringFramework:
    """
    Factory function to create HVAC prompt engineering framework
    
    Returns:
        Configured HVACPromptEngineeringFramework instance
    """
    return HVACPromptEngineeringFramework()
