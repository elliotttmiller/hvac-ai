# HVAC VLM Examples

This directory contains example scripts demonstrating the HVAC Vision-Language Model (VLM) system.

## Overview

The HVAC VLM system transforms a general-purpose Vision-Language Model into a domain-specific expert for HVAC blueprint analysis. These examples guide you through the complete workflow from data generation to model deployment.

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 64GB
- Storage: 1TB NVMe SSD

**Recommended:**
- GPU: NVIDIA A100 (40GB+ VRAM)
- RAM: 128GB+
- Storage: 2TB+ NVMe SSD

### Software Requirements

```bash
# Install required packages
cd python-services
pip install -r requirements.txt

# Additional VLM-specific packages
pip install transformers>=4.35.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0
```

## Examples

### 1. Generate Synthetic Data

**File:** `01_generate_synthetic_data.py`

Generate synthetic HVAC drawings with automatic annotations for training.

```bash
python examples/vlm/01_generate_synthetic_data.py
```

**Output:**
- Synthetic HVAC drawings in `datasets/synthetic_hvac_v1/images/`
- JSON annotations in `datasets/synthetic_hvac_v1/annotations/`
- Dataset manifest

**What it does:**
- Creates programmatic HVAC drawings (AHU, ducts, VAVs, diffusers)
- Generates perfect labels automatically
- Creates training prompts and responses
- Exports in VLM-compatible format

### 2. Train VLM

**File:** `02_train_vlm.py`

Train HVAC VLM using Supervised Fine-Tuning (SFT) with LoRA.

```bash
python examples/vlm/02_train_vlm.py
```

**Requirements:**
- Synthetic dataset from step 1
- GPU with 16GB+ VRAM
- 4-8 hours training time (depends on dataset size)

**What it does:**
- Loads base VLM (Qwen2-VL or InternVL)
- Applies LoRA for efficient fine-tuning
- Trains on synthetic HVAC data
- Saves checkpoints and metrics

**Configuration:**
```python
model_type = "qwen2-vl"
base_model = "Qwen/Qwen2-VL-7B-Instruct"
num_epochs = 3
batch_size = 2
learning_rate = 2e-5
use_lora = True  # Efficient training
```

### 3. Test VLM

**File:** `03_test_vlm.py`

Test trained VLM on HVAC blueprints.

```bash
python examples/vlm/03_test_vlm.py
```

**What it tests:**
1. **Component Detection** - Identify all HVAC components
2. **Relationship Analysis** - Understand system connectivity
3. **Specification Extraction** - Extract CFM, sizes, materials
4. **Code Compliance** - Check ASHRAE/SMACNA standards

**Sample Output:**
```json
{
  "components": [
    {
      "type": "supply_air_duct",
      "bbox": [250, 240, 650, 260],
      "attributes": {
        "size": "12x10",
        "cfm": 2000,
        "designation": "SD-1"
      }
    }
  ]
}
```

### 4. Validate with Engineering Rules

**File:** `04_validate_vlm.py`

Validate VLM predictions using HVAC engineering rules.

```bash
python examples/vlm/04_validate_vlm.py
```

**What it validates:**
- Supply/exhaust separation
- CFM balance (±10%)
- Valid component relationships
- Attribute ranges (CFM, sizes)
- ASHRAE/SMACNA compliance

**Sample Output:**
```
Validation Results
------------------
Is Valid: True
Errors: 0
Warnings: 1
Info: 0

RKLF Reward Score: 0.900
```

## Workflow

### Phase 1: Foundation (Months 1-4)

1. **Generate Training Data**
   ```bash
   python examples/vlm/01_generate_synthetic_data.py
   ```
   - Start with 1,000 samples
   - Scale to 10,000+ samples
   - Add real drawings incrementally

2. **Set Up Model Architecture**
   - Choose base model (Qwen2-VL or InternVL)
   - Configure LoRA for efficient training
   - Set up monitoring (TensorBoard, W&B)

### Phase 2: Training (Months 5-8)

3. **Supervised Fine-Tuning**
   ```bash
   python examples/vlm/02_train_vlm.py
   ```
   - Train on synthetic data (3-5 epochs)
   - Monitor loss and metrics
   - Validate on held-out set

4. **Test and Iterate**
   ```bash
   python examples/vlm/03_test_vlm.py
   ```
   - Test on diverse blueprints
   - Identify failure modes
   - Collect more training data

### Phase 3: Validation (Months 9-12+)

5. **Validate and Refine**
   ```bash
   python examples/vlm/04_validate_vlm.py
   ```
   - Run on real-world test set
   - Achieve >95% accuracy targets
   - Deploy to production

## Advanced Usage

### Custom Prompts

```python
from core.vlm.model_interface import create_hvac_vlm

vlm = create_hvac_vlm(model_type="qwen2-vl")
vlm.load_model()

custom_prompt = """
Analyze this HVAC drawing and identify all VAV boxes.
For each VAV, extract:
- Designation (e.g., VAV-101)
- CFM rating
- Location coordinates
"""

result = vlm.custom_analysis(
    image_path="blueprint.png",
    custom_prompt=custom_prompt,
    return_json=True
)
```

### Batch Processing

```python
from pathlib import Path

blueprint_dir = Path("blueprints/")
for blueprint in blueprint_dir.glob("*.pdf"):
    result = vlm.analyze_components(str(blueprint))
    # Process result...
```

### Model Export

```python
# Export to ONNX for production
from core.vlm.model_interface import create_hvac_vlm

vlm = create_hvac_vlm(model_path="checkpoints/hvac_vlm_sft/final")
vlm.load_model()

# Export (implementation depends on model type)
# vlm.export(format="onnx", path="models/hvac_vlm.onnx")
```

## Performance Targets

### Production-Ready Metrics

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Symbol Detection F1 | 0.90 | 0.95 | 0.98+ |
| Text Extraction Accuracy | 0.85 | 0.92 | 0.97+ |
| Relationship Inference | 0.80 | 0.90 | 0.95+ |
| CFM/Size Extraction | 0.85 | 0.92 | 0.97+ |
| Rule Validation Recall | 0.90 | 0.95 | 0.98+ |

### Inference Performance

- **Latency:** <2 seconds per drawing
- **Throughput:** 30+ images/minute on A100
- **GPU Memory:** ~8-12GB for inference (with quantization)

## Troubleshooting

### Out of Memory (OOM)

**Problem:** GPU runs out of memory during training

**Solutions:**
```python
# Reduce batch size
batch_size = 1
gradient_accumulation_steps = 16

# Use smaller model
base_model = "Qwen/Qwen2-VL-2B-Instruct"

# Enable CPU offloading
device_map = "auto"

# Use int8 quantization
quantization = "int8"
```

### Low Accuracy

**Problem:** Model predictions are inaccurate

**Solutions:**
1. Generate more training data (10,000+ samples)
2. Increase training epochs (5-10 epochs)
3. Add real annotated drawings to training set
4. Fine-tune learning rate (try 1e-5 to 5e-5)
5. Use larger base model (7B → 72B parameters)

### Model Not Loading

**Problem:** Model fails to load

**Solutions:**
```bash
# Update transformers
pip install --upgrade transformers>=4.35.0

# Check model path
ls checkpoints/hvac_vlm_sft/final/

# Try base model first
model_path = "Qwen/Qwen2-VL-7B-Instruct"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

1. **Scale Training Data**
   - Generate 10,000+ synthetic samples
   - Collect and annotate real HVAC drawings
   - Implement active learning pipeline

2. **Domain Pre-Training**
   - Collect unlabeled HVAC drawings
   - Pre-train vision encoder
   - Requires significant GPU resources

3. **RKLF Training**
   - Implement reinforcement learning loop
   - Use validator rewards
   - Iteratively improve predictions

4. **Production Deployment**
   - Export to ONNX/TensorRT
   - Set up API service
   - Implement monitoring and logging

## Resources

### Documentation
- [VLM Implementation Guide](../../docs/VLM_IMPLEMENTATION_GUIDE.md)
- [Data Schema](../../python-services/core/vlm/data_schema.py)
- [Model Interface](../../python-services/core/vlm/model_interface.py)

### External Links
- [Qwen2-VL Documentation](https://github.com/QwenLM/Qwen2-VL)
- [InternVL Documentation](https://github.com/OpenGVLab/InternVL)
- [PEFT (LoRA) Guide](https://github.com/huggingface/peft)

### Research Papers
- "Qwen2-VL: Enhancing Vision-Language Model's Perception"
- "InternVL: Scaling up Vision Foundation Models"
- "LoRA: Low-Rank Adaptation of Large Language Models"

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [VLM Implementation Guide](../../docs/VLM_IMPLEMENTATION_GUIDE.md)
3. Create GitHub issue with:
   - System specifications
   - Error message
   - Steps to reproduce

---

**Version:** 1.0  
**Last Updated:** 2024-12-18  
**Status:** Active Development
