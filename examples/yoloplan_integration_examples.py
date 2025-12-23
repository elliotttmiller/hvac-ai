"""
Complete YOLOplan Integration Examples (Weeks 1-16 Implementation)
Demonstrates all features: symbol detection, BOM generation, connectivity analysis
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_services.core.ai.yoloplan_detector import create_yoloplan_detector
from python_services.core.ai.yoloplan_bom import create_bom_generator, create_connectivity_analyzer
from python_services.core.ai.integrated_detector import create_integrated_detector


def create_sample_blueprint():
    """Create a sample HVAC blueprint for demonstration"""
    image = np.ones((1200, 1600, 3), dtype=np.uint8) * 255
    
    # Title block
    cv2.rectangle(image, (1200, 1000), (1550, 1150), (0, 0, 0), 2)
    cv2.putText(image, "HVAC PLAN", (1220, 1050), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # AHU (primary equipment)
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.putText(image, "AHU-1", (110, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # VAV boxes (distribution)
    cv2.rectangle(image, (350, 100), (420, 170), (0, 255, 0), -1)
    cv2.putText(image, "VAV-1", (360, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    
    cv2.rectangle(image, (550, 100), (620, 170), (0, 255, 0), -1)
    cv2.putText(image, "VAV-2", (560, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    
    # Diffusers (terminals)
    for x in [380, 480, 580, 680]:
        cv2.circle(image, (x, 300), 15, (255, 0, 0), -1)
        cv2.putText(image, "D", (x-5, 305),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Ducts (connections)
    cv2.line(image, (200, 150), (350, 135), (128, 128, 128), 3)
    cv2.line(image, (420, 135), (550, 135), (128, 128, 128), 3)
    cv2.line(image, (385, 170), (385, 300), (128, 128, 128), 2)
    cv2.line(image, (585, 170), (585, 300), (128, 128, 128), 2)
    
    return image


def example_1_symbol_detection():
    """Example 1: Basic symbol detection"""
    print("\n" + "="*60)
    print("Example 1: Symbol Detection (Weeks 3-6)")
    print("="*60)
    
    # Create detector
    detector = create_yoloplan_detector(confidence=0.5)
    
    # Load or create blueprint
    image = create_sample_blueprint()
    
    # Detect symbols
    print("\nDetecting symbols...")
    results = detector.detect_symbols(image)
    
    # Display results
    print(f"\n✓ Detection complete!")
    print(f"  Total symbols detected: {results['total_symbols']}")
    print(f"\nSymbol counts:")
    for symbol_type, count in results['counts'].items():
        print(f"  {symbol_type}: {count}")
    
    # Export results
    print("\nExporting results...")
    json_output = detector.export_results(results, format='json')
    csv_output = detector.export_results(results, format='csv')
    
    print(f"  ✓ JSON export: {len(json_output)} bytes")
    print(f"  ✓ CSV export: {len(csv_output)} bytes")
    
    return results


def example_2_batch_processing():
    """Example 2: Batch processing multiple blueprints"""
    print("\n" + "="*60)
    print("Example 2: Batch Processing (Weeks 3-6)")
    print("="*60)
    
    detector = create_yoloplan_detector()
    
    # Create multiple test blueprints
    print("\nCreating test blueprints...")
    images = [create_sample_blueprint() for _ in range(5)]
    print(f"  Created {len(images)} blueprints")
    
    # Batch process
    print("\nBatch processing...")
    results = detector.batch_detect(images, parallel=True)
    
    # Display results
    print(f"\n✓ Batch processing complete!")
    print(f"  Processed: {len(results)} blueprints")
    
    total_symbols = sum(r['total_symbols'] for r in results)
    print(f"  Total symbols across all blueprints: {total_symbols}")
    print(f"  Average symbols per blueprint: {total_symbols / len(results):.1f}")
    
    return results


def example_3_bom_generation():
    """Example 3: Bill of Materials generation"""
    print("\n" + "="*60)
    print("Example 3: BOM Generation (Weeks 11-14)")
    print("="*60)
    
    # Detect symbols first
    detector = create_yoloplan_detector()
    image = create_sample_blueprint()
    symbol_results = detector.detect_symbols(image)
    
    # Generate BOM
    print("\nGenerating Bill of Materials...")
    bom_generator = create_bom_generator()
    bom = bom_generator.generate_bom(
        symbol_results['counts'],
        symbol_results['detections']
    )
    
    # Display BOM
    print(f"\n✓ BOM generated with {len(bom)} items:")
    print(f"\n{'Item ID':<12} {'Description':<25} {'Qty':<6} {'Unit':<6} {'Est. Cost':<12}")
    print("-" * 70)
    
    total_cost = 0
    for item in bom:
        cost_str = f"${item.estimated_cost:,.2f}" if item.estimated_cost else "N/A"
        print(f"{item.item_id:<12} {item.description:<25} {item.quantity:<6} {item.unit:<6} {cost_str:<12}")
        if item.estimated_cost:
            total_cost += item.estimated_cost
    
    print("-" * 70)
    print(f"{'Total Estimated Cost:':<50} ${total_cost:,.2f}")
    
    # Export BOM
    print("\nExporting BOM...")
    csv_file = bom_generator.export_bom(bom, format='csv', output_path='bom_output.csv')
    json_file = bom_generator.export_bom(bom, format='json', output_path='bom_output.json')
    print(f"  ✓ CSV: {csv_file}")
    print(f"  ✓ JSON: {json_file}")
    
    return bom


def example_4_connectivity_analysis():
    """Example 4: Connectivity and netlist analysis"""
    print("\n" + "="*60)
    print("Example 4: Connectivity Analysis (Weeks 11-14)")
    print("="*60)
    
    # Detect symbols
    detector = create_yoloplan_detector()
    image = create_sample_blueprint()
    symbol_results = detector.detect_symbols(image)
    
    # Analyze connectivity
    print("\nAnalyzing system connectivity...")
    connectivity_analyzer = create_connectivity_analyzer()
    netlist = connectivity_analyzer.generate_netlist(
        symbol_results['detections'],
        image_size=image.shape[:2]
    )
    
    # Display connectivity
    print(f"\n✓ Connectivity analysis complete!")
    print(f"\nGraph Statistics:")
    stats = netlist['graph_stats']
    print(f"  Nodes (symbols): {stats['num_nodes']}")
    print(f"  Edges (connections): {stats['num_edges']}")
    print(f"  Connected components: {stats['connected_components']}")
    
    print(f"\nSystem Hierarchy:")
    hierarchy = netlist['hierarchy']
    print(f"  Primary Equipment: {len(hierarchy['primary_equipment'])}")
    print(f"  Distribution: {len(hierarchy['distribution'])}")
    print(f"  Terminal Units: {len(hierarchy['terminal_units'])}")
    
    print(f"\nIdentified Circuits:")
    for circuit in netlist['circuits']:
        print(f"  Circuit {circuit['circuit_id']}: {circuit['circuit_type']} "
              f"({circuit['num_symbols']} symbols)")
    
    print(f"\nConnections (first 5):")
    for i, conn in enumerate(netlist['connections'][:5], 1):
        print(f"  {i}. {conn['from']['type']} → {conn['to']['type']} "
              f"({conn['connection_type']})")
    
    return netlist


def example_5_integrated_analysis():
    """Example 5: Complete integrated analysis"""
    print("\n" + "="*60)
    print("Example 5: Integrated Analysis (Complete Pipeline)")
    print("="*60)
    
    # Create integrated detector
    print("\nInitializing integrated detector...")
    detector = create_integrated_detector(
        use_sahi=False,  # Can enable if SAHI models available
        use_document_processing=False,  # Can enable if OCR/VLM available
        use_bom_generation=True,
        use_connectivity_analysis=True
    )
    
    # Load blueprint
    image = create_sample_blueprint()
    
    # Run complete analysis
    print("\nRunning complete analysis pipeline...")
    results = detector.analyze_blueprint(image)
    
    # Display comprehensive results
    print(f"\n✓ Complete analysis finished!")
    print(f"\nAnalysis Summary:")
    summary = results['summary']
    print(f"  Stages completed: {', '.join(summary['stages_completed'])}")
    
    if 'total_symbols' in summary:
        print(f"  Total symbols: {summary['total_symbols']}")
    if 'bom_items' in summary:
        print(f"  BOM items: {summary['bom_items']}")
    if 'total_estimated_cost' in summary:
        print(f"  Estimated cost: ${summary['total_estimated_cost']:,.2f}")
    if 'connections' in summary:
        print(f"  Connections: {summary['connections']}")
    if 'circuits' in summary:
        print(f"  Circuits: {summary['circuits']}")
    
    # Export all results
    print("\nExporting results...")
    exported = detector.export_results(results, output_dir='output', formats=['json', 'csv'])
    print(f"  Exported {len(exported)} files:")
    for name, path in exported.items():
        print(f"    - {name}: {path}")
    
    return results


def example_6_batch_integrated():
    """Example 6: Batch integrated analysis"""
    print("\n" + "="*60)
    print("Example 6: Batch Integrated Analysis")
    print("="*60)
    
    detector = create_integrated_detector(
        use_sahi=False,
        use_document_processing=False
    )
    
    # Create multiple blueprints
    print("\nCreating test blueprints...")
    images = [create_sample_blueprint() for _ in range(3)]
    print(f"  Created {len(images)} blueprints")
    
    # Batch process
    print("\nBatch processing...")
    batch_results = detector.batch_analyze(images, parallel=False)
    
    # Display batch summary
    print(f"\n✓ Batch processing complete!")
    print(f"\nBatch Summary:")
    summary = batch_results['batch_summary']
    print(f"  Total blueprints: {summary['total_blueprints']}")
    print(f"  Total symbols detected: {summary['total_symbols_detected']}")
    print(f"  Total BOM items: {summary['total_bom_items']}")
    print(f"  Avg symbols/blueprint: {summary['average_symbols_per_blueprint']:.1f}")
    
    return batch_results


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("YOLOplan Integration Examples")
    print("Complete Implementation (Weeks 1-16)")
    print("="*60)
    
    try:
        # Run all examples
        example_1_symbol_detection()
        example_2_batch_processing()
        example_3_bom_generation()
        example_4_connectivity_analysis()
        example_5_integrated_analysis()
        example_6_batch_integrated()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nFeatures demonstrated:")
        print("  ✓ Symbol detection (Weeks 3-6)")
        print("  ✓ Batch processing (Weeks 3-6)")
        print("  ✓ BOM generation (Weeks 11-14)")
        print("  ✓ Connectivity analysis (Weeks 11-14)")
        print("  ✓ Integrated pipeline (Weeks 15-16)")
        print("  ✓ Export capabilities (JSON, CSV)")
        print("\nSee docs/YOLOPLAN_INTEGRATION.md for more details")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
