"""
Regional Code Overrides Manager

Manages jurisdiction-specific code requirements that override or supplement
national standards (ASHRAE, SMACNA, IMC).
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Jurisdiction(Enum):
    """Supported jurisdictions with regional code variations"""
    NATIONAL = "national"  # Default national codes
    CALIFORNIA_TITLE_24 = "california_title_24"
    NYC_BUILDING_CODE = "nyc_building_code"
    FLORIDA_BUILDING_CODE = "florida_building_code"
    CHICAGO_BUILDING_CODE = "chicago_building_code"
    TEXAS_STATE_CODE = "texas_state_code"


@dataclass
class RegionalOverride:
    """Represents a regional code override"""
    jurisdiction: Jurisdiction
    code_section: str
    override_type: str  # "requirement", "exemption", "modification"
    description: str
    multiplier: Optional[float] = None  # For numeric requirements
    additive: Optional[float] = None    # For additive adjustments


class RegionalCodeManager:
    """
    Regional Code Override Manager
    
    Manages jurisdiction-specific code requirements and applies
    appropriate overrides to validation logic.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.overrides: Dict[Jurisdiction, List[RegionalOverride]] = {}
        
        # Load overrides from configuration file if provided
        if config_path:
            self._load_overrides_from_file(config_path)
        else:
            self._load_default_overrides()
    
    def _load_default_overrides(self):
        """Load default regional overrides for major jurisdictions"""
        
        # California Title 24 - More stringent ventilation requirements
        self.overrides[Jurisdiction.CALIFORNIA_TITLE_24] = [
            RegionalOverride(
                jurisdiction=Jurisdiction.CALIFORNIA_TITLE_24,
                code_section="ventilation_rates",
                override_type="modification",
                description="California Title 24 requires 15% higher ventilation rates",
                multiplier=1.15
            ),
            RegionalOverride(
                jurisdiction=Jurisdiction.CALIFORNIA_TITLE_24,
                code_section="equipment_efficiency",
                override_type="requirement",
                description="Higher equipment efficiency requirements per Title 24",
                multiplier=1.0
            )
        ]
        
        # NYC Building Code - Stricter fire protection requirements
        self.overrides[Jurisdiction.NYC_BUILDING_CODE] = [
            RegionalOverride(
                jurisdiction=Jurisdiction.NYC_BUILDING_CODE,
                code_section="fire_damper_spacing",
                override_type="modification",
                description="NYC requires fire dampers at 50 ft max spacing in ducts",
                multiplier=1.0
            ),
            RegionalOverride(
                jurisdiction=Jurisdiction.NYC_BUILDING_CODE,
                code_section="smoke_detection",
                override_type="requirement",
                description="Enhanced smoke detection requirements in HVAC systems",
                multiplier=1.0
            )
        ]
        
        # Florida Building Code - Hurricane resistance requirements
        self.overrides[Jurisdiction.FLORIDA_BUILDING_CODE] = [
            RegionalOverride(
                jurisdiction=Jurisdiction.FLORIDA_BUILDING_CODE,
                code_section="duct_support",
                override_type="modification",
                description="Florida requires enhanced duct support for hurricane resistance",
                multiplier=1.3  # 30% more frequent support spacing
            ),
            RegionalOverride(
                jurisdiction=Jurisdiction.FLORIDA_BUILDING_CODE,
                code_section="outdoor_unit_anchoring",
                override_type="requirement",
                description="Special anchoring requirements for outdoor HVAC equipment",
                multiplier=1.0
            )
        ]
        
        self.logger.info(
            f"Loaded default overrides for {len(self.overrides)} jurisdictions"
        )
    
    def _load_overrides_from_file(self, config_path: str):
        """
        Load regional overrides from JSON configuration file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            for jurisdiction_str, overrides_list in config.items():
                try:
                    jurisdiction = Jurisdiction[jurisdiction_str.upper()]
                except KeyError:
                    self.logger.warning(
                        f"Unknown jurisdiction: {jurisdiction_str}"
                    )
                    continue
                
                self.overrides[jurisdiction] = [
                    RegionalOverride(
                        jurisdiction=jurisdiction,
                        code_section=o["code_section"],
                        override_type=o["override_type"],
                        description=o["description"],
                        multiplier=o.get("multiplier"),
                        additive=o.get("additive")
                    )
                    for o in overrides_list
                ]
            
            self.logger.info(
                f"Loaded overrides from {config_path} for "
                f"{len(self.overrides)} jurisdictions"
            )
        except Exception as e:
            self.logger.error(f"Failed to load overrides from {config_path}: {e}")
            # Fall back to default overrides
            self._load_default_overrides()
    
    def get_overrides(
        self,
        jurisdiction: Jurisdiction,
        code_section: Optional[str] = None
    ) -> List[RegionalOverride]:
        """
        Get regional overrides for a jurisdiction
        
        Args:
            jurisdiction: Jurisdiction to get overrides for
            code_section: Optional filter by code section
            
        Returns:
            List of applicable regional overrides
        """
        overrides = self.overrides.get(jurisdiction, [])
        
        if code_section:
            overrides = [
                o for o in overrides
                if o.code_section == code_section
            ]
        
        return overrides
    
    def apply_ventilation_override(
        self,
        base_requirement: float,
        jurisdiction: Jurisdiction
    ) -> float:
        """
        Apply regional override to ventilation requirement
        
        Args:
            base_requirement: Base ventilation requirement (CFM)
            jurisdiction: Jurisdiction to apply overrides for
            
        Returns:
            Adjusted requirement with regional overrides applied
        """
        overrides = self.get_overrides(jurisdiction, "ventilation_rates")
        
        adjusted_requirement = base_requirement
        
        for override in overrides:
            if override.multiplier:
                adjusted_requirement *= override.multiplier
            if override.additive:
                adjusted_requirement += override.additive
        
        return adjusted_requirement
    
    def apply_support_spacing_override(
        self,
        base_spacing: float,
        jurisdiction: Jurisdiction
    ) -> float:
        """
        Apply regional override to duct support spacing
        
        Args:
            base_spacing: Base support spacing (feet)
            jurisdiction: Jurisdiction to apply overrides for
            
        Returns:
            Adjusted spacing with regional overrides applied
        """
        overrides = self.get_overrides(jurisdiction, "duct_support")
        
        adjusted_spacing = base_spacing
        
        for override in overrides:
            if override.multiplier:
                # Multiplier reduces spacing (more frequent supports)
                adjusted_spacing /= override.multiplier
            if override.additive:
                adjusted_spacing += override.additive
        
        return adjusted_spacing
    
    def get_jurisdiction_notes(
        self,
        jurisdiction: Jurisdiction
    ) -> List[str]:
        """
        Get human-readable notes about jurisdiction-specific requirements
        
        Args:
            jurisdiction: Jurisdiction to get notes for
            
        Returns:
            List of notes about regional requirements
        """
        overrides = self.get_overrides(jurisdiction)
        
        return [
            f"{o.code_section}: {o.description}"
            for o in overrides
        ]
    
    def detect_jurisdiction(
        self,
        location_metadata: Dict[str, Any]
    ) -> Jurisdiction:
        """
        Detect jurisdiction from blueprint location metadata
        
        Args:
            location_metadata: Dictionary with location info (state, city, etc.)
            
        Returns:
            Detected jurisdiction (defaults to NATIONAL)
        """
        state = location_metadata.get("state", "").upper()
        city = location_metadata.get("city", "").upper()
        
        # Simple rule-based detection
        if state == "CA" or state == "CALIFORNIA":
            return Jurisdiction.CALIFORNIA_TITLE_24
        elif state == "NY" and ("NYC" in city or "NEW YORK" in city):
            return Jurisdiction.NYC_BUILDING_CODE
        elif state == "FL" or state == "FLORIDA":
            return Jurisdiction.FLORIDA_BUILDING_CODE
        elif state == "IL" and "CHICAGO" in city:
            return Jurisdiction.CHICAGO_BUILDING_CODE
        elif state == "TX" or state == "TEXAS":
            return Jurisdiction.TEXAS_STATE_CODE
        else:
            return Jurisdiction.NATIONAL
    
    def export_overrides_config(self, output_path: str):
        """
        Export current overrides to JSON configuration file
        
        Args:
            output_path: Path to write configuration file
        """
        config = {}
        
        for jurisdiction, overrides in self.overrides.items():
            config[jurisdiction.value] = [
                {
                    "code_section": o.code_section,
                    "override_type": o.override_type,
                    "description": o.description,
                    "multiplier": o.multiplier,
                    "additive": o.additive
                }
                for o in overrides
            ]
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Exported overrides configuration to {output_path}")
