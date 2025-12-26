"""
Example script demonstrating the HVAC Drawing Analysis Pipeline.
Shows basic usage, batch processing, and error handling.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import pipeline components (may not be available without dependencies)
try:
    from core.ai.hvac_pipeline import create_hvac_analyzer
    from core.ai.pipeline_models import PipelineConfig, PipelineStage
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Pipeline not available: {e}")
    print("   This is a demonstration script showing API usage")
    print("   Install dependencies to run actual pipeline:")
    print("   pip install ultralytics easyocr")
    print()
    PIPELINE_AVAILABLE = False
    PipelineConfig = None
    PipelineStage = None


def create_sample_image(width=1200, height=800):
    """Create a sample HVAC drawing for demonstration."""
    # Create white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some colored rectangles to simulate components
    # Valve (blue box)
    img[200:300, 300:400] = [100, 100, 200]
    
    # Damper (green box)
    img[200:300, 500:600] = [100, 200, 100]
    
    # Text regions (gray boxes)
    img[180:195, 305:380] = [150, 150, 150]  # Above valve
    img[180:195, 505:580] = [150, 150, 150]  # Above damper
    
    return img


def example_basic_usage():
    """Example 1: Basic pipeline usage."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("=" * 80)
    
    # Note: This example requires a real YOLO model file
    # For demonstration, we'll show the API usage
    model_path = os.getenv("MODEL_PATH", "./models/yolo11m-obb-hvac.pt")
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print("   Set MODEL_PATH environment variable to your YOLO model file")
        print("   Example usage shown below:\n")
    
    # Create pipeline configuration
    if PIPELINE_AVAILABLE:
        config = PipelineConfig(
            confidence_threshold=0.7,
            max_processing_time_ms=25.0,
            enable_gpu=True
        )
        
        print("Configuration:")
        print(f"  - Confidence threshold: {config.confidence_threshold}")
        print(f"  - Max processing time: {config.max_processing_time_ms}ms")
        print(f"  - GPU enabled: {config.enable_gpu}")
        print()
    
    # This would create the analyzer (requires model file)
    print("Code to create analyzer:")
    print("  from core.ai.hvac_pipeline import create_hvac_analyzer")
    print("  from core.ai.pipeline_models import PipelineConfig")
    print()
    print("  config = PipelineConfig(")
    print("      confidence_threshold=0.7,")
    print("      max_processing_time_ms=25.0,")
    print("      enable_gpu=True")
    print("  )")
    print()
    print("  analyzer = create_hvac_analyzer(")
    print(f"      model_path='{model_path}',")
    print("      config=config")
    print("  )")
    print()
    
    print("Code to analyze drawing:")
    print("  result = analyzer.analyze_drawing('path/to/drawing.png')")
    print()
    
    print("Expected result structure:")
    print("  - request_id: Unique identifier")
    print("  - stage: Pipeline stage (complete/failed)")
    print("  - detection_result: Stage 1 results")
    print("  - text_results: Stage 2 results")
    print("  - interpretation_result: Stage 3 results")
    print("  - total_processing_time_ms: Total time")
    print("  - stage_timings: Per-stage timing breakdown")
    print()


def example_result_processing():
    """Example 2: Processing pipeline results."""
    print("=" * 80)
    print("EXAMPLE 2: Processing Pipeline Results")
    print("=" * 80)
    
    print("Code to check result status:")
    print("""
    if result.success:
        print("✅ Analysis completed successfully!")
        
        # Access detections
        detections = result.detection_result.detections
        print(f"Found {len(detections)} components")
        
        # Access text results
        for text_result in result.text_results:
            print(f"Text: {text_result.text}")
            print(f"Confidence: {text_result.confidence:.2f}")
        
        # Access interpretations
        if result.interpretation_result:
            for interp in result.interpretation_result.interpretations:
                print(f"Equipment: {interp.equipment_type}")
                print(f"Zone: {interp.zone_number}")
                print(f"System ID: {interp.system_id}")
                
                # Check for associated component
                if interp.associated_component:
                    print(f"Associated with: {interp.associated_component.class_name}")
        
        # Check timing
        print(f"Total time: {result.total_processing_time_ms:.2f}ms")
        for stage, time_ms in result.stage_timings.items():
            print(f"  {stage}: {time_ms:.2f}ms")
    
    elif result.partial_success:
        print("⚠️  Partial success - some stages completed")
        print(f"Stage reached: {result.stage}")
        
        # Check what succeeded
        if result.detection_result:
            print("✅ Detection completed")
        if result.text_results:
            print("✅ Text recognition completed")
        if result.interpretation_result:
            print("✅ Interpretation completed")
        
        # Check warnings
        for warning in result.warnings:
            print(f"⚠️  {warning}")
    
    else:
        print("❌ Analysis failed")
        
        # Check errors
        for error in result.errors:
            print(f"Error in {error.stage}: {error.message}")
            print(f"Severity: {error.severity}")
    """)


def example_batch_processing():
    """Example 3: Batch processing multiple drawings."""
    print("=" * 80)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 80)
    
    print("Code for batch processing:")
    print("""
    # Create analyzer once
    analyzer = create_hvac_analyzer(model_path="./models/yolo11m-obb-hvac.pt")
    
    # Process multiple images
    image_paths = [
        "drawing1.png",
        "drawing2.png",
        "drawing3.png"
    ]
    
    results = []
    for path in image_paths:
        result = analyzer.analyze_drawing(path)
        results.append(result)
        
        # Print summary
        print(f"{path}: {len(result.detection_result.detections)} detections "
              f"in {result.total_processing_time_ms:.2f}ms")
    
    # Calculate statistics
    avg_time = sum(r.total_processing_time_ms for r in results) / len(results)
    success_rate = sum(1 for r in results if r.success) / len(results) * 100
    
    print(f"Average processing time: {avg_time:.2f}ms")
    print(f"Success rate: {success_rate:.1f}%")
    """)


def example_api_usage():
    """Example 4: Using the REST API."""
    print("=" * 80)
    print("EXAMPLE 4: REST API Usage")
    print("=" * 80)
    
    print("Start the server:")
    print("  cd services/hvac-analysis")
    print("  python hvac_analysis_service.py")
    print()
    
    print("Single image analysis (curl):")
    print("""
    curl -X POST "http://localhost:8000/api/v1/pipeline/analyze" \\
      -F "image=@drawing.png" \\
      -F "confidence_threshold=0.7" \\
      -F "max_processing_time_ms=25.0"
    """)
    
    print("Batch analysis (curl):")
    print("""
    curl -X POST "http://localhost:8000/api/v1/pipeline/analyze/batch" \\
      -F "images=@drawing1.png" \\
      -F "images=@drawing2.png" \\
      -F "confidence_threshold=0.7"
    """)
    
    print("Health check:")
    print("  curl http://localhost:8000/api/v1/pipeline/health")
    print()
    
    print("Statistics:")
    print("  curl http://localhost:8000/api/v1/pipeline/stats")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "HVAC Pipeline Examples" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    examples = [
        example_basic_usage,
        example_result_processing,
        example_batch_processing,
        example_api_usage
    ]
    
    for example in examples:
        example()
        print()
    
    print("=" * 80)
    print("For more information, see:")
    print("  - PIPELINE_README.md - Complete documentation")
    print("  - http://localhost:8000/docs - API documentation (when server running)")
    print("  - tests/test_hvac_pipeline.py - Test examples")
    print("=" * 80)


if __name__ == "__main__":
    main()
