#!/usr/bin/env python3
"""
HVAC Blueprint Analysis Example

This script demonstrates how to use the new HVAC services for
complete blueprint analysis workflow.

Usage:
    python examples/hvac_analysis_example.py <blueprint_path>
"""

import sys
import os
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.hvac_document.hvac_document_processor import (
    create_hvac_document_processor,
    BlueprintFormat
)
from services.hvac_ai.hvac_sahi_engine import create_hvac_sahi_predictor
from services.hvac_domain.hvac_system_engine import (
    HVACSystemEngine,
    HVACComponent,
    HVACComponentType
)
from services.hvac_ai.hvac_prompt_engineering import create_hvac_prompt_framework


def analyze_hvac_blueprint(blueprint_path: str, model_path: str = "models/<your_model_file>.pt"):
    """
    Complete HVAC blueprint analysis workflow
    
    Args:
        blueprint_path: Path to HVAC blueprint file
        model_path: Path to inference model weights (YOLO/Ultralytics)
    """
    
    print("=" * 60)
    print("HVAC Blueprint Analysis Pipeline")
    print("=" * 60)
    print()
    
    # Step 1: Document Processing
    print("Step 1: Processing Blueprint Document")
    print("-" * 60)
    
    processor = create_hvac_document_processor(config={
        "target_dpi": 300,
        "enhance_ductwork_lines": True,
        "enhance_symbols": True
    })
    
    doc_result = processor.process_document(
        file_path=blueprint_path,
        format_hint=BlueprintFormat.PDF
    )
    
    print(f"✓ Processed {doc_result['metadata']['page_count']} page(s)")
    
    for page in doc_result['pages']:
        quality = page['quality_metrics']
        print(f"  Page {page['page_number']}: Quality {quality.overall_quality:.2f}")
        if quality.issues:
            print(f"    Issues: {', '.join(quality.issues)}")
    
    print()
    
    # Step 2: Component Detection with SAHI
    print("Step 2: Detecting HVAC Components (SAHI)")
    print("-" * 60)
    
    # Simulated detections for demonstration
    simulated_detections = [
        {
            "id": "duct_001",
            "type": HVACComponentType.DUCTWORK,
            "bbox": [100.0, 100.0, 200.0, 50.0],
            "confidence": 0.95
        },
        {
            "id": "diffuser_001",
            "type": HVACComponentType.DIFFUSER,
            "bbox": [250.0, 100.0, 30.0, 30.0],
            "confidence": 0.92
        },
        {
            "id": "diffuser_002",
            "type": HVACComponentType.DIFFUSER,
            "bbox": [250.0, 200.0, 30.0, 30.0],
            "confidence": 0.90
        },
        {
            "id": "vav_001",
            "type": HVACComponentType.VAV_BOX,
            "bbox": [50.0, 100.0, 40.0, 40.0],
            "confidence": 0.88
        }
    ]
    
    print(f"✓ Detected {len(simulated_detections)} HVAC components")
    
    for detection in simulated_detections:
        print(f"  {detection['id']}: {detection['type'].value} "
              f"(confidence: {detection['confidence']:.2f})")
    
    print()
    
    # Step 3: System Relationship Analysis
    print("Step 3: Analyzing System Relationships")
    print("-" * 60)
    
    engine = HVACSystemEngine()
    
    # Add detected components to system engine
    for detection in simulated_detections:
        component = HVACComponent(
            id=detection['id'],
            component_type=detection['type'],
            bbox=detection['bbox'],
            confidence=detection['confidence']
        )
        engine.add_component(component)
    
    # Build relationship graph
    graph = engine.build_relationship_graph()
    
    print(f"✓ Built relationship graph with {len(graph)} components")
    print(f"✓ Found {len(engine.relationships)} relationships")
    
    print()
    
    # Step 4: System Validation
    print("Step 4: Validating System Configuration")
    print("-" * 60)
    
    validation = engine.validate_system_configuration()
    
    if validation['is_valid']:
        print("✓ System configuration is VALID")
    else:
        print("✗ System configuration has VIOLATIONS")
    
    print(f"  Components: {validation['summary']['total_components']}")
    print(f"  Violations: {validation['summary']['violation_count']}")
    print(f"  Warnings: {validation['summary']['warning_count']}")
    
    print()
    
    # Step 5: Export Results
    print("Step 5: Exporting Analysis Results")
    print("-" * 60)
    
    graph_export = engine.export_system_graph()
    
    print(f"✓ Exported system graph")
    print(f"  Nodes: {len(graph_export['nodes'])}")
    print(f"  Edges: {len(graph_export['edges'])}")
    
    print()
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return {
        'document': doc_result,
        'detections': simulated_detections,
        'system_graph': graph_export,
        'validation': validation
    }


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python hvac_analysis_example.py <blueprint_path>")
        sys.exit(1)
    
    blueprint_path = sys.argv[1]
    
    if not os.path.exists(blueprint_path):
        print(f"Error: Blueprint file not found: {blueprint_path}")
        sys.exit(1)
    
    try:
        results = analyze_hvac_blueprint(blueprint_path)
        print(f"\n✓ Analysis complete with {len(results['detections'])} components detected")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
