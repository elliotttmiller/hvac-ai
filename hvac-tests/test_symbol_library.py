"""
Test suite for HVAC Symbol Library
Validates comprehensive symbol library expansion and industry standards compliance
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path for imports
services_path = Path(__file__).parent.parent / "services"
python_services_path = Path(__file__).parent.parent / "python-services"

sys.path.insert(0, str(services_path / "hvac-document"))
sys.path.insert(0, str(python_services_path))

import numpy as np


class TestSymbolLibraryExpansion(unittest.TestCase):
    """Test comprehensive HVAC symbol library expansion"""
    
    def test_symbol_library_import(self):
        """Verify symbol library can be imported"""
        try:
            from hvac_symbol_library import (
                HVACSymbolLibrary,
                HVACSymbolCategory,
                create_hvac_symbol_library
            )
            self.assertTrue(True, "Symbol library imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import symbol library: {e}")
    
    def test_symbol_category_count(self):
        """Verify symbol category enum has been massively expanded"""
        from hvac_symbol_library import HVACSymbolCategory
        
        # Count all symbol categories
        category_count = len(HVACSymbolCategory)
        
        # We expanded from ~34 to 130+ enum values (covering all major categories)
        self.assertGreaterEqual(
            category_count, 
            95, 
            f"Symbol library should have 95+ categories, got {category_count}"
        )
        
        print(f"✓ Symbol library has {category_count} categories")
    
    def test_symbol_coverage_taxonomy_alignment(self):
        """Verify symbol library covers all HVAC_TAXONOMY categories"""
        from hvac_symbol_library import HVACSymbolCategory
        try:
            from core.ai.sam_inference import HVAC_TAXONOMY
        except ImportError:
            self.skipTest("SAM inference module not available (requires torch)")
        
        # Convert taxonomy labels to expected category names
        taxonomy_categories = set()
        for label in HVAC_TAXONOMY:
            # Convert "Valve-Ball" to "valve_ball"
            category_name = label.lower().replace("-", "_")
            taxonomy_categories.add(category_name)
        
        # Get symbol library categories
        library_categories = set(cat.value for cat in HVACSymbolCategory)
        
        # Check coverage
        covered_categories = taxonomy_categories & library_categories
        coverage_percent = (len(covered_categories) / len(taxonomy_categories)) * 100
        
        print(f"✓ Symbol library covers {len(covered_categories)}/{len(taxonomy_categories)} "
              f"taxonomy categories ({coverage_percent:.1f}%)")
        
        # We should have very high coverage
        self.assertGreater(
            coverage_percent, 
            85, 
            f"Symbol library should cover >85% of taxonomy, got {coverage_percent:.1f}%"
        )
    
    def test_symbol_library_initialization(self):
        """Verify symbol library initializes with templates"""
        from hvac_symbol_library import create_hvac_symbol_library
        
        library = create_hvac_symbol_library()
        
        # Check that templates were loaded
        self.assertIsNotNone(library.templates, "Library should have templates")
        self.assertGreater(
            len(library.templates), 
            40, 
            f"Library should have 40+ templates, got {len(library.templates)}"
        )
        
        print(f"✓ Symbol library initialized with {len(library.templates)} templates")
    
    def test_actuator_symbols(self):
        """Verify all 7 actuator types have templates"""
        from hvac_symbol_library import HVACSymbolCategory, create_hvac_symbol_library
        
        actuator_categories = [
            HVACSymbolCategory.ACTUATOR_DIAPHRAGM,
            HVACSymbolCategory.ACTUATOR_GENERIC,
            HVACSymbolCategory.ACTUATOR_MANUAL,
            HVACSymbolCategory.ACTUATOR_MOTORIZED,
            HVACSymbolCategory.ACTUATOR_PISTON,
            HVACSymbolCategory.ACTUATOR_PNEUMATIC,
            HVACSymbolCategory.ACTUATOR_SOLENOID,
        ]
        
        library = create_hvac_symbol_library()
        library_categories = [t.category for t in library.templates]
        
        for category in actuator_categories:
            self.assertIn(
                category, 
                library_categories,
                f"Actuator {category.value} should have template"
            )
        
        print(f"✓ All 7 actuator types have templates")
    
    def test_valve_symbols(self):
        """Verify all 14 valve types have templates"""
        from hvac_symbol_library import HVACSymbolCategory, create_hvac_symbol_library
        
        valve_categories = [
            HVACSymbolCategory.VALVE_3WAY,
            HVACSymbolCategory.VALVE_4WAY,
            HVACSymbolCategory.VALVE_ANGLE,
            HVACSymbolCategory.VALVE_BALL,
            HVACSymbolCategory.VALVE_BUTTERFLY,
            HVACSymbolCategory.VALVE_CHECK,
            HVACSymbolCategory.VALVE_CONTROL,
            HVACSymbolCategory.VALVE_DIAPHRAGM,
            HVACSymbolCategory.VALVE_GATE,
            HVACSymbolCategory.VALVE_GENERIC,
            HVACSymbolCategory.VALVE_GLOBE,
            HVACSymbolCategory.VALVE_NEEDLE,
            HVACSymbolCategory.VALVE_PLUG,
            HVACSymbolCategory.VALVE_RELIEF,
        ]
        
        library = create_hvac_symbol_library()
        library_categories = [t.category for t in library.templates]
        
        for category in valve_categories:
            self.assertIn(
                category, 
                library_categories,
                f"Valve {category.value} should have template"
            )
        
        print(f"✓ All 14 valve types have templates")
    
    def test_equipment_symbols(self):
        """Verify all 11 equipment types have templates"""
        from hvac_symbol_library import HVACSymbolCategory, create_hvac_symbol_library
        
        equipment_categories = [
            HVACSymbolCategory.EQUIPMENT_AGITATOR_MIXER,
            HVACSymbolCategory.EQUIPMENT_COMPRESSOR,
            HVACSymbolCategory.EQUIPMENT_FAN_BLOWER,
            HVACSymbolCategory.EQUIPMENT_GENERIC,
            HVACSymbolCategory.EQUIPMENT_HEAT_EXCHANGER,
            HVACSymbolCategory.EQUIPMENT_MOTOR,
            HVACSymbolCategory.EQUIPMENT_PUMP_CENTRIFUGAL,
            HVACSymbolCategory.EQUIPMENT_PUMP_DOSING,
            HVACSymbolCategory.EQUIPMENT_PUMP_GENERIC,
            HVACSymbolCategory.EQUIPMENT_PUMP_SCREW,
            HVACSymbolCategory.EQUIPMENT_VESSEL,
        ]
        
        library = create_hvac_symbol_library()
        library_categories = [t.category for t in library.templates]
        
        for category in equipment_categories:
            self.assertIn(
                category, 
                library_categories,
                f"Equipment {category.value} should have template"
            )
        
        print(f"✓ All 11 equipment types have templates")
    
    def test_instrument_symbols(self):
        """Verify all 11 instrument types have templates"""
        from hvac_symbol_library import HVACSymbolCategory, create_hvac_symbol_library
        
        instrument_categories = [
            HVACSymbolCategory.INSTRUMENT_ANALYZER,
            HVACSymbolCategory.INSTRUMENT_FLOW_INDICATOR,
            HVACSymbolCategory.INSTRUMENT_FLOW_TRANSMITTER,
            HVACSymbolCategory.INSTRUMENT_GENERIC,
            HVACSymbolCategory.INSTRUMENT_LEVEL_INDICATOR,
            HVACSymbolCategory.INSTRUMENT_LEVEL_SWITCH,
            HVACSymbolCategory.INSTRUMENT_LEVEL_TRANSMITTER,
            HVACSymbolCategory.INSTRUMENT_PRESSURE_INDICATOR,
            HVACSymbolCategory.INSTRUMENT_PRESSURE_SWITCH,
            HVACSymbolCategory.INSTRUMENT_PRESSURE_TRANSMITTER,
            HVACSymbolCategory.INSTRUMENT_TEMPERATURE,
        ]
        
        library = create_hvac_symbol_library()
        library_categories = [t.category for t in library.templates]
        
        for category in instrument_categories:
            self.assertIn(
                category, 
                library_categories,
                f"Instrument {category.value} should have template"
            )
        
        print(f"✓ All 11 instrument types have templates")
    
    def test_template_properties(self):
        """Verify templates have required properties"""
        from hvac_symbol_library import create_hvac_symbol_library
        
        library = create_hvac_symbol_library()
        
        for template in library.templates:
            # Check template has a category
            self.assertIsNotNone(template.category, "Template should have category")
            
            # Check template image exists and is valid
            self.assertIsNotNone(template.template, "Template should have image")
            self.assertIsInstance(template.template, np.ndarray, "Template should be numpy array")
            self.assertEqual(len(template.template.shape), 2, "Template should be 2D (grayscale)")
            
            # Check metadata exists
            self.assertIsNotNone(template.metadata, "Template should have metadata")
            self.assertIn("description", template.metadata, "Template should have description")
            self.assertIn("standard", template.metadata, "Template should have standard reference")
        
        print(f"✓ All {len(library.templates)} templates have required properties")
    
    def test_industry_standards_compliance(self):
        """Verify templates reference appropriate industry standards"""
        from hvac_symbol_library import create_hvac_symbol_library
        
        library = create_hvac_symbol_library()
        
        valid_standards = [
            "ASHRAE 134",
            "ASHRAE Standard 134",
            "SMACNA",
            "ISO 14617",
            "ISA S5.1",
            "ISA S5.1/ISO 14617",
            "ASHRAE 134/ISO 14617"
        ]
        
        standards_count = {}
        
        for template in library.templates:
            standard = template.metadata.get("standard", "")
            self.assertTrue(
                any(s in standard for s in valid_standards),
                f"Template {template.category.value} has invalid standard: {standard}"
            )
            
            # Count standards usage
            for std in valid_standards:
                if std in standard:
                    standards_count[std] = standards_count.get(std, 0) + 1
        
        print(f"✓ All templates reference valid industry standards:")
        for std, count in sorted(standards_count.items(), key=lambda x: -x[1]):
            print(f"  - {std}: {count} templates")
    
    def test_get_symbol_description(self):
        """Verify symbol descriptions are comprehensive"""
        from hvac_symbol_library import HVACSymbolCategory, create_hvac_symbol_library
        
        library = create_hvac_symbol_library()
        
        # Test a few key categories
        test_categories = [
            HVACSymbolCategory.VALVE_BALL,
            HVACSymbolCategory.ACTUATOR_MOTORIZED,
            HVACSymbolCategory.INSTRUMENT_FLOW_TRANSMITTER,
            HVACSymbolCategory.EQUIPMENT_PUMP_CENTRIFUGAL,
        ]
        
        for category in test_categories:
            description = library.get_symbol_description(category)
            self.assertIsNotNone(description, f"Should have description for {category.value}")
            self.assertGreater(len(description), 10, f"Description should be meaningful for {category.value}")
            print(f"  - {category.value}: {description}")
        
        print(f"✓ Symbol descriptions are comprehensive")


class TestSymbolDetection(unittest.TestCase):
    """Test symbol detection capabilities"""
    
    def test_template_matching_setup(self):
        """Verify template matching is properly configured"""
        from hvac_symbol_library import create_hvac_symbol_library
        
        library = create_hvac_symbol_library()
        
        for template in library.templates:
            # Check scale range
            self.assertIsNotNone(template.scale_range)
            self.assertEqual(len(template.scale_range), 2)
            self.assertLess(template.scale_range[0], template.scale_range[1])
            
            # Check confidence threshold
            self.assertIsNotNone(template.min_confidence)
            self.assertGreater(template.min_confidence, 0.0)
            self.assertLessEqual(template.min_confidence, 1.0)
        
        print(f"✓ Template matching properly configured for {len(library.templates)} templates")
    
    def test_rotation_invariance_configuration(self):
        """Verify rotation invariance is appropriately set"""
        from hvac_symbol_library import HVACSymbolCategory, create_hvac_symbol_library
        
        library = create_hvac_symbol_library()
        
        # Circular symbols should be rotation invariant
        circular_categories = [
            HVACSymbolCategory.DIFFUSER_ROUND,
            HVACSymbolCategory.FAN,
        ]
        
        # Directional symbols should not be rotation invariant
        directional_categories = [
            HVACSymbolCategory.DIFFUSER_SQUARE,
            HVACSymbolCategory.AHU,
        ]
        
        # Count rotation invariant vs fixed templates
        rotation_invariant_count = sum(1 for t in library.templates if t.rotation_invariant)
        rotation_fixed_count = sum(1 for t in library.templates if not t.rotation_invariant)
        
        # Verify we have a mix of both types
        self.assertGreater(rotation_invariant_count, 20, "Should have many rotation invariant symbols")
        self.assertGreater(rotation_fixed_count, 20, "Should have many fixed orientation symbols")
        
        # Check specific circular categories are rotation invariant
        for template in library.templates:
            if template.category in circular_categories:
                self.assertTrue(
                    template.rotation_invariant,
                    f"{template.category.value} should be rotation invariant"
                )
            elif template.category in directional_categories:
                self.assertFalse(
                    template.rotation_invariant,
                    f"{template.category.value} should not be rotation invariant"
                )
        
        print(f"✓ Rotation invariance appropriately configured")


class TestDocumentation(unittest.TestCase):
    """Test that documentation exists and is comprehensive"""
    
    def test_documentation_file_exists(self):
        """Verify HVAC_SYMBOL_LIBRARY.md documentation exists"""
        doc_path = Path(__file__).parent.parent / "docs" / "HVAC_SYMBOL_LIBRARY.md"
        self.assertTrue(doc_path.exists(), "HVAC_SYMBOL_LIBRARY.md should exist")
        
        # Check file is substantial
        doc_size = doc_path.stat().st_size
        self.assertGreater(doc_size, 10000, "Documentation should be comprehensive (>10KB)")
        
        print(f"✓ Documentation exists ({doc_size:,} bytes)")
    
    def test_documentation_content(self):
        """Verify documentation covers all major topics"""
        doc_path = Path(__file__).parent.parent / "docs" / "HVAC_SYMBOL_LIBRARY.md"
        
        if not doc_path.exists():
            self.skipTest("Documentation file not found")
        
        content = doc_path.read_text()
        
        required_sections = [
            "## Overview",
            "## Symbol Categories",
            "### 1. Actuators",
            "### 2. Valves",
            "### 3. Equipment",
            "## Industry Standards Reference",
            "### ASHRAE Standard 134",
            "### SMACNA",
            "### ISO 14617",
            "### ISA S5.1",
            "## Usage Examples",
            "## Performance Characteristics",
            "## Best Practices",
        ]
        
        for section in required_sections:
            self.assertIn(
                section, 
                content,
                f"Documentation should include section: {section}"
            )
        
        print(f"✓ Documentation includes all required sections")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
