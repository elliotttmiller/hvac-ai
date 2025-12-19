#!/usr/bin/env python3
"""
Example: Test HVAC VLM on Blueprint Images

This script demonstrates how to use a trained HVAC VLM to analyze
blueprint images and extract components, relationships, and specifications.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-services"))

from core.vlm.model_interface import create_hvac_vlm


def main():
    """Test HVAC VLM"""
    
    print("=" * 70)
    print("HVAC VLM Testing")
    print("=" * 70)
    print()
    
    # Configuration
    model_type = "qwen2-vl"
    model_path = "checkpoints/hvac_vlm_sft/final"  # Path to trained model
    test_image = "datasets/synthetic_hvac_v1/images/synthetic_supply_1234.png"
    
    print(f"Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Model Path: {model_path}")
    print(f"  Test Image: {test_image}")
    print()
    
    # Check if model exists
    if not Path(model_path).exists():
        print("WARNING: Trained model not found!")
        print(f"Using base model instead...")
        model_path = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Check if test image exists
    if not Path(test_image).exists():
        print("WARNING: Test image not found!")
        print("Please generate synthetic data first or provide a valid image path")
        return
    
    print("Loading model...")
    try:
        vlm = create_hvac_vlm(
            model_type=model_type,
            model_path=model_path,
            device="cuda"  # Use "cpu" if no GPU available
        )
        vlm.load_model()
        print("Model loaded successfully!")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Test 1: Component Detection
    print("-" * 70)
    print("Test 1: Component Detection")
    print("-" * 70)
    try:
        result = vlm.analyze_components(test_image)
        print(json.dumps(result, indent=2))
        print()
    except Exception as e:
        print(f"ERROR: {e}")
        print()
    
    # Test 2: Relationship Analysis
    print("-" * 70)
    print("Test 2: Relationship Analysis")
    print("-" * 70)
    try:
        result = vlm.analyze_relationships(test_image)
        print(json.dumps(result, indent=2))
        print()
    except Exception as e:
        print(f"ERROR: {e}")
        print()
    
    # Test 3: Specification Extraction
    print("-" * 70)
    print("Test 3: Specification Extraction")
    print("-" * 70)
    try:
        result = vlm.extract_specifications(test_image)
        print(json.dumps(result, indent=2))
        print()
    except Exception as e:
        print(f"ERROR: {e}")
        print()
    
    # Test 4: Code Compliance Check
    print("-" * 70)
    print("Test 4: Code Compliance Check")
    print("-" * 70)
    try:
        result = vlm.check_code_compliance(test_image)
        print(json.dumps(result, indent=2))
        print()
    except Exception as e:
        print(f"ERROR: {e}")
        print()
    
    print("=" * 70)
    print("Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
