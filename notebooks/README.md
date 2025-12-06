# SAM HVAC Blueprint Analysis Pipeline

## Overview

This directory contains a comprehensive Jupyter notebook for training and deploying Segment Anything Model (SAM) for HVAC blueprint component detection and segmentation.

## Notebook: `sam_hvac_pipeline.ipynb`

### Purpose
End-to-end pipeline for:
- Dataset quality auditing and image selection
- Class consolidation (70+ categories → 6 simplified classes)
- Comprehensive prompt engineering for HVAC blueprints
- SAM model fine-tuning with frozen ViT backbone
- Inference pipeline with visualization
- Model export and documentation

### Simplified Categories (6 Classes)

1. **Equipment**: Pumps, coils, fans, motors, compressors, tanks, heat exchangers
2. **Ductwork**: Ducts, bends, reducers
3. **Piping**: Insulated pipes, traps
4. **Valves**: All valve types consolidated (ball, gate, check, globe, control, etc.)
5. **Air Devices**: Dampers, fire dampers, filters, detectors
6. **Controls**: Sensors, switches, instrumentation

### Key Features

✅ **Industry Standards Compliance**
- PEP 8 style guidelines
- Comprehensive type hints
- Detailed docstrings (Google/NumPy format)
- Professional logging (no print statements)
- Proper error handling

✅ **Dataset Quality System**
- Image quality scoring (resolution, contrast, sharpness)
- Annotation quality metrics (completeness, diversity, density)
- Automated selection of top-quality images
- Statistical reporting and analysis

✅ **Advanced Prompt Engineering**
- Category-specific prompt templates
- Contextual spatial relationship prompts
- Hierarchical prompts based on confidence
- Fallback mechanisms for difficult cases
- Blueprint-optimized prompt strategies

✅ **Memory-Optimized Training**
- Google Colab T4 GPU optimization
- Mixed precision training (FP16)
- Gradient accumulation
- Small batch sizes (1-2 images)
- Efficient data loading

✅ **Professional Architecture**
- Class-based design patterns
- Dataclass configurations
- Modular components
- Comprehensive documentation

### System Requirements

**Recommended Environment:**
- Google Colab (Free or Pro)
- T4 GPU (minimum)
- 12+ GB GPU RAM
- Python 3.10+

**Key Dependencies:**
```
segment-anything
torch >= 2.0.0
torchvision >= 0.15.0
opencv-python-headless
pycocotools
albumentations
numpy, pandas, scipy
matplotlib, pillow
```

### Usage Instructions

#### 1. Open in Google Colab

```python
# Upload notebook to Colab or open from GitHub
# Install dependencies (first cell)
!pip install -q git+https://github.com/facebookresearch/segment-anything.git
# ... (other dependencies installed automatically)
```

#### 2. Upload Dataset

```python
# Upload hvac-dataset.zip to Colab
from google.colab import files
uploaded = files.upload()

# Extract dataset
import zipfile
with zipfile.ZipFile('hvac-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/datasets/')
```

#### 3. Run All Cells

Execute cells sequentially to:
1. ✅ Setup environment
2. ✅ Load and analyze dataset
3. ✅ Audit image quality
4. ✅ Initialize prompt engineering
5. ✅ Prepare training pipeline
6. ✅ Train model (uncomment training lines)
7. ✅ Run inference
8. ✅ Export model

#### 4. Start Training

In the final cell, uncomment training lines:
```python
# Uncomment these lines to start training:
print("\nStarting training...")
trainer.train(train_loader, num_epochs=config.num_epochs)
```

### Configuration

Modify `TrainingConfig` dataclass to adjust hyperparameters:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 1  # Increase if you have more GPU memory
    num_epochs: int = 20  # Adjust based on dataset size
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    checkpoint_freq: int = 5
```

### Output Files

After training completion:
- `checkpoint_epoch_N.pth` - Model checkpoints
- `sam_hvac_final.pth` - Final trained model
- `MODEL_CARD.md` - Model documentation
- `sam_hvac_pipeline.log` - Training logs

### Dataset Structure

Expected directory structure:
```
/content/datasets/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    └── ...
```

### Performance Tips

**Memory Optimization:**
- Keep batch_size = 1 or 2
- Use gradient accumulation for larger effective batch size
- Enable mixed precision training
- Monitor GPU memory: `torch.cuda.memory_summary()`

**Training Speed:**
- Use smaller subset for initial testing
- Adjust num_workers in DataLoader (0 for Colab)
- Enable pin_memory for faster transfers
- Use quality audit to select best images

**Quality Improvement:**
- Increase quality audit sample size
- Adjust quality threshold for stricter selection
- Add custom prompt templates
- Fine-tune prompt engineering weights

### Troubleshooting

**Out of Memory (OOM) Errors:**
```python
# Reduce batch size
config.batch_size = 1

# Or reduce image size
config.image_size = 512  # Default is 1024
```

**Slow Training:**
```python
# Check GPU utilization
!nvidia-smi

# Reduce quality audit samples
auditor.audit_images(max_images=50)
```

**Poor Segmentation Results:**
```python
# Increase training data quality
selected_images = auditor.select_top_quality(percentile=20.0)  # Top 20% only

# Or add more training epochs
config.num_epochs = 50
```

### Customization

**Add Custom Categories:**
```python
# Modify CATEGORY_MAPPING dictionary
CATEGORY_MAPPING = {
    "Equipment": [...],
    "YourCategory": ["cat1", "cat2", ...],
    # ...
}
```

**Custom Prompt Templates:**
```python
# In PromptEngineer class
def _build_category_prompts(self):
    return {
        "YourCategory": [
            "Your custom prompt",
            "Another prompt variant",
            # ...
        ]
    }
```

**Quality Metrics Weights:**
```python
# Adjust in ImageQualityMetrics.compute_overall_score()
weights = {
    'resolution': 0.20,  # Increase resolution importance
    'contrast': 0.15,
    # ...
}
```

### Best Practices

1. **Start Small**: Test with 50-100 images before full training
2. **Quality First**: Use quality audit to select best samples
3. **Monitor Training**: Watch loss curves and GPU utilization
4. **Save Checkpoints**: Enable frequent checkpointing
5. **Document Changes**: Keep notes on hyperparameter changes
6. **Version Control**: Save notebook versions after successful runs

### Citation

If you use this pipeline in your research or production systems:

```bibtex
@software{sam_hvac_pipeline,
  title = {SAM Model Pipeline for HVAC Blueprint Analysis},
  author = {HVAC AI Development Team},
  year = {2025},
  url = {https://github.com/elliotttmiller/hvac-ai}
}
```

### License

MIT License - See repository root for full license text

### Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Review existing issues and discussions
- Check troubleshooting section above

---

**Last Updated**: 2025-12-06  
**Version**: 1.0.0  
**Maintainer**: HVAC AI Development Team
