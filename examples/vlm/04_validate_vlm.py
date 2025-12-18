#!/usr/bin/env python3
"""
Example: Validate HVAC VLM with Engineering Rules

This script demonstrates how to validate VLM predictions using
HVAC engineering rules (ASHRAE/SMACNA standards).
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-services"))

from core.vlm.validation import HVACValidator


def main():
    """Validate HVAC VLM predictions"""
    
    print("=" * 70)
    print("HVAC VLM Validation")
    print("=" * 70)
    print()
    
    # Create validator
    validator = HVACValidator()
    
    # Example prediction data
    # In practice, this would come from VLM model output
    components = [
        {
            "component_id": "ahu_1",
            "component_type": "ahu",
            "bbox": [100, 200, 250, 300],
            "attributes": {
                "designation": "AHU-1",
                "cfm": 5000
            }
        },
        {
            "component_id": "duct_1",
            "component_type": "supply_air_duct",
            "bbox": [250, 240, 650, 260],
            "attributes": {
                "designation": "SD-1",
                "size": "12x10",
                "cfm": 2000
            }
        },
        {
            "component_id": "vav_1",
            "component_type": "vav",
            "bbox": [450, 150, 510, 190],
            "attributes": {
                "designation": "VAV-101",
                "cfm": 800
            }
        }
    ]
    
    relationships = [
        {
            "source": "ahu_1",
            "target": "duct_1",
            "type": "supplies"
        },
        {
            "source": "duct_1",
            "target": "vav_1",
            "type": "feeds"
        }
    ]
    
    print("Validating HVAC system...")
    print(f"  Components: {len(components)}")
    print(f"  Relationships: {len(relationships)}")
    print()
    
    # Run validation
    result = validator.validate_system(components, relationships)
    
    print("-" * 70)
    print("Validation Results")
    print("-" * 70)
    print(f"Is Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Info: {result.info}")
    print()
    
    if result.violations:
        print("Violations:")
        for i, violation in enumerate(result.violations, 1):
            print(f"\n{i}. {violation.rule_name} ({violation.severity.value})")
            print(f"   Message: {violation.message}")
            if violation.expected:
                print(f"   Expected: {violation.expected}")
            if violation.actual:
                print(f"   Actual: {violation.actual}")
            if violation.component_ids:
                print(f"   Components: {', '.join(violation.component_ids)}")
    else:
        print("No violations found! ✓")
    
    print()
    
    # Calculate RKLF reward
    reward = validator.calculate_reward(result)
    print(f"RKLF Reward Score: {reward:.3f}")
    print(f"  (1.0 = perfect, 0.0 = many violations)")
    print()
    
    print("=" * 70)
    print("Validation Complete!")
    print("=" * 70)
    print()
    print("This validator can be used for:")
    print("  1. Post-processing VLM predictions")
    print("  2. RKLF training feedback loop")
    print("  3. Quality assurance in production")
    print()


if __name__ == "__main__":
    main()
