#!/usr/bin/env python3
"""
Example: Generate Synthetic HVAC Training Data

This script demonstrates how to generate synthetic HVAC drawings
with automatic annotations for VLM training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-services"))

from core.vlm.synthetic_generator import generate_training_dataset


def main():
    """Generate synthetic training dataset"""
    
    print("=" * 70)
    print("HVAC VLM Synthetic Data Generation")
    print("=" * 70)
    print()
    
    # Configuration
    output_dir = "datasets/synthetic_hvac_v1"
    num_samples = 100  # Start with 100 samples for testing
    image_size = (2048, 2048)
    
    print(f"Configuration:")
    print(f"  Output Directory: {output_dir}")
    print(f"  Number of Samples: {num_samples}")
    print(f"  Image Size: {image_size}")
    print()
    
    # Generate dataset
    examples = generate_training_dataset(
        output_dir=output_dir,
        num_samples=num_samples,
        image_size=image_size
    )
    
    print()
    print("=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print(f"Generated {len(examples)} training examples")
    print(f"Images saved to: {output_dir}/images/")
    print(f"Annotations saved to: {output_dir}/annotations/")
    print(f"Manifest saved to: {output_dir}/dataset_manifest.json")
    print()
    print("Next steps:")
    print("  1. Review generated images and annotations")
    print("  2. Run 02_train_vlm.py to start training")
    print()


if __name__ == "__main__":
    main()
