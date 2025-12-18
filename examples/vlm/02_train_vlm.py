#!/usr/bin/env python3
"""
Example: Train HVAC VLM with Supervised Fine-Tuning

This script demonstrates how to train a HVAC-specific VLM using
supervised fine-tuning on synthetic data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-services"))

from core.vlm.training import SupervisedFinetuner, create_sft_trainer


def main():
    """Train HVAC VLM"""
    
    print("=" * 70)
    print("HVAC VLM Supervised Fine-Tuning")
    print("=" * 70)
    print()
    
    # Configuration
    model_type = "qwen2-vl"
    base_model = "Qwen/Qwen2-VL-7B-Instruct"
    train_data_path = "datasets/synthetic_hvac_v1"
    output_dir = "checkpoints/hvac_vlm_sft"
    use_lora = True  # Use LoRA for efficient training
    
    print(f"Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Base Model: {base_model}")
    print(f"  Training Data: {train_data_path}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Use LoRA: {use_lora}")
    print()
    
    # Check if training data exists
    if not Path(train_data_path).exists():
        print("ERROR: Training data not found!")
        print(f"Please run 01_generate_synthetic_data.py first")
        return
    
    print("Creating SFT trainer...")
    try:
        trainer = create_sft_trainer(
            model_type=model_type,
            base_model_path=base_model,
            output_dir=output_dir,
            use_lora=use_lora
        )
    except Exception as e:
        print(f"ERROR: Failed to create trainer: {e}")
        print()
        print("Note: This requires:")
        print("  1. GPU with sufficient VRAM (16GB+ recommended)")
        print("  2. transformers>=4.35.0 installed")
        print("  3. Base model downloaded or accessible")
        return
    
    print("Starting training...")
    print()
    
    # Training configuration
    training_config = {
        "train_data_path": train_data_path,
        "num_epochs": 3,
        "batch_size": 2,  # Small batch size for memory efficiency
        "learning_rate": 2e-5,
        "warmup_steps": 500,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "eval_steps": 500,
        "logging_steps": 50
    }
    
    print("Training Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        metrics = trainer.train(**training_config)
        
        print()
        print("=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Final Metrics: {metrics}")
        print(f"Model saved to: {output_dir}")
        print()
        print("Next steps:")
        print("  1. Test the trained model with 03_test_vlm.py")
        print("  2. Run validation with 04_validate_vlm.py")
        print()
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
