# HVAC VLM Quick Start Guide

## What is the HVAC VLM System?

The HVAC Vision-Language Model (VLM) system is a specialized AI designed to understand and analyze HVAC blueprints with engineering-level precision. Unlike general-purpose AI, it's specifically trained on HVAC components, relationships, and industry standards (ASHRAE/SMACNA).

## Key Capabilities

1. **Component Detection** - Identify all HVAC components (ducts, VAVs, dampers, sensors)
2. **Relationship Analysis** - Understand how components connect and interact
3. **Specification Extraction** - Extract CFM ratings, sizes, materials, designations
4. **Code Compliance** - Validate designs against ASHRAE/SMACNA standards
5. **Engineering Intelligence** - Understand context and design intent

## Quick Example

```python
from core.vlm.model_interface import create_hvac_vlm

# Load the model
vlm = create_hvac_vlm(model_type="qwen2-vl")
vlm.load_model()

# Analyze a blueprint
result = vlm.analyze_components("blueprint.pdf")

# Result includes:
# - All detected components with locations
# - Engineering specifications (CFM, sizes)
# - Component designations (SD-1, VAV-101)
# - Structured JSON output
```

## Installation

### Prerequisites

**Hardware:**
- GPU: NVIDIA RTX 3090 (24GB) minimum
- RAM: 64GB
- Storage: 500GB+

**Software:**
```bash
# Python 3.10+
python --version

# CUDA for GPU support
nvidia-smi
```

### Setup

```bash
# 1. Navigate to python-services
cd python-services

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install VLM-specific packages (if not in requirements)
pip install transformers>=4.35.0 peft>=0.7.0 bitsandbytes>=0.41.0

# 4. Test installation
python -c "from core.vlm import HVACVLMInterface; print('✓ VLM installed')"
```

## Usage Patterns

### 1. Generate Training Data

```bash
python examples/vlm/01_generate_synthetic_data.py
```

Generates synthetic HVAC drawings with automatic annotations.

**Output:**
- 100 synthetic blueprints
- Perfect JSON annotations
- Training-ready dataset

**Time:** ~5 minutes

### 2. Train Your Model

```bash
python examples/vlm/02_train_vlm.py
```

Trains a domain-specific VLM on HVAC data.

**Requirements:**
- GPU with 16GB+ VRAM
- 4-8 hours training time

**What it does:**
- Fine-tunes Qwen2-VL on HVAC data
- Uses LoRA for efficient training
- Saves checkpoints automatically

### 3. Test the Model

```bash
python examples/vlm/03_test_vlm.py
```

Tests trained model on blueprints.

**Performs:**
- Component detection
- Relationship analysis
- Specification extraction
- Code compliance check

### 4. Validate Results

```bash
python examples/vlm/04_validate_vlm.py
```

Validates predictions using HVAC engineering rules.

**Checks:**
- Supply/exhaust separation
- CFM balance
- Valid relationships
- Engineering constraints

## Common Use Cases

### Use Case 1: Component Inventory

```python
# Extract all components from a blueprint
components = vlm.analyze_components("blueprint.pdf")

# Filter by type
vavs = [c for c in components['components'] 
        if c['type'] == 'vav']

# Generate inventory report
print(f"Found {len(vavs)} VAV boxes")
for vav in vavs:
    print(f"  {vav['attributes']['designation']}: "
          f"{vav['attributes']['cfm']} CFM")
```

### Use Case 2: System Validation

```python
# Analyze system relationships
relationships = vlm.analyze_relationships("blueprint.pdf")

# Check for violations
if relationships['violations']:
    print("Design issues found:")
    for violation in relationships['violations']:
        print(f"  - {violation['message']}")
else:
    print("✓ System design valid")
```

### Use Case 3: Specification Extraction

```python
# Extract all specifications
specs = vlm.extract_specifications("blueprint.pdf")

# Generate material takeoff
total_cfm = sum(s.get('cfm', 0) for s in specs['components'])
print(f"Total system CFM: {total_cfm}")
```

### Use Case 4: Code Compliance

```python
# Check code compliance
compliance = vlm.check_code_compliance("blueprint.pdf")

# Generate report
print(f"ASHRAE Compliance: {compliance['ashrae_compliant']}")
print(f"SMACNA Compliance: {compliance['smacna_compliant']}")

for issue in compliance['issues']:
    print(f"  [{issue['code']}] {issue['description']}")
```

## Performance Expectations

### Accuracy (After Training)

| Metric | Expected Performance |
|--------|---------------------|
| Component Detection | 85-95% F1 |
| Text Extraction | 80-92% accuracy |
| Relationship Inference | 75-90% accuracy |
| Code Violations | 90-95% recall |

### Speed

| Operation | Time (A100 GPU) |
|-----------|----------------|
| Single blueprint analysis | 1-2 seconds |
| Batch processing (10 drawings) | 10-15 seconds |
| Model loading (first time) | 5-10 seconds |

### Resource Usage

| Resource | Usage |
|----------|-------|
| GPU Memory | 8-12GB (with quantization) |
| System RAM | 16-32GB |
| Disk Space | 50GB (model + cache) |

## Troubleshooting

### Issue: Model won't load

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Use smaller model
vlm = create_hvac_vlm(
    model_type="qwen2-vl",
    model_path="Qwen/Qwen2-VL-2B-Instruct",  # Smaller
    quantization="int8"  # Reduce memory
)
```

### Issue: Poor accuracy

**Problem:** Model predictions are inaccurate

**Solution:**
1. Generate more training data (1,000+ samples)
2. Train for more epochs (5-10)
3. Add real annotated drawings
4. Use larger base model

### Issue: Slow inference

**Problem:** Analysis takes too long

**Solution:**
```python
# Export to ONNX for faster inference
# Reduce image resolution
# Use quantization
# Batch process multiple drawings
```

## Next Steps

1. **Learn More:**
   - [Full Implementation Guide](VLM_IMPLEMENTATION_GUIDE.md)
   - [Development Roadmap](VLM_ROADMAP.md)
   - [Example Scripts](../examples/vlm/README.md)

2. **Start Building:**
   - Generate synthetic data (100 samples)
   - Train a small model (3 epochs)
   - Test on your blueprints
   - Iterate and improve

3. **Scale Up:**
   - Generate 10,000+ samples
   - Collect real drawings
   - Train production model
   - Deploy to API service

## Best Practices

### Data Preparation
✅ Start with synthetic data for fast iteration  
✅ Gradually add real drawings  
✅ Maintain train/val/test split  
✅ Version your datasets  

### Training
✅ Use LoRA for efficient fine-tuning  
✅ Monitor training with TensorBoard  
✅ Save checkpoints frequently  
✅ Evaluate on held-out test set  

### Deployment
✅ Quantize models for production  
✅ Implement caching for repeated analyses  
✅ Set up monitoring and logging  
✅ Version your models  

### Quality Assurance
✅ Validate all predictions with HVAC rules  
✅ Human review for critical applications  
✅ Track accuracy metrics over time  
✅ Collect failure cases for retraining  

## Support & Community

### Getting Help
- **Documentation:** [VLM Implementation Guide](VLM_IMPLEMENTATION_GUIDE.md)
- **Examples:** [examples/vlm/](../examples/vlm/)
- **Issues:** GitHub Issues

### Contributing
- Report bugs and issues
- Share your trained models
- Contribute training data
- Improve documentation

### Resources
- [Qwen2-VL GitHub](https://github.com/QwenLM/Qwen2-VL)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [PEFT (LoRA) Documentation](https://github.com/huggingface/peft)

## FAQ

**Q: Do I need to train from scratch?**  
A: No! Start with a pre-trained base model (Qwen2-VL) and fine-tune on HVAC data.

**Q: How much training data do I need?**  
A: Start with 1,000 synthetic samples, scale to 10,000+. Add real drawings incrementally.

**Q: What GPU do I need?**  
A: Minimum: RTX 3090 (24GB). Recommended: A100 (40GB+). Can use CPU but very slow.

**Q: How long does training take?**  
A: 4-8 hours for 1,000 samples on A100. Scales linearly with data size.

**Q: Can I use this in production?**  
A: Yes! After training and validation. Expect 90-95% accuracy with proper training.

**Q: Does it work with DWG files?**  
A: Yes! Convert DWG to images first using existing converters in the platform.

**Q: What about 3D models?**  
A: Current system is 2D-focused. 3D support planned for future releases.

**Q: Can I customize for my specific needs?**  
A: Absolutely! Modify data_schema.py to add custom component types and rules.

---

**Version:** 1.0  
**Last Updated:** 2024-12-18  
**Next Review:** After Phase 2 completion

Ready to build world-class HVAC AI? Start with `examples/vlm/01_generate_synthetic_data.py`! 🚀
