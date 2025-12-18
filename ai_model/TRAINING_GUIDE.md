# HVAC AI Training Pipeline Guide

## Overview

This guide provides comprehensive documentation for training YOLO11 segmentation models for HVAC blueprint analysis. The training pipeline has been optimized for Google Colab with T4 GPU and includes advanced features for production-ready model training.

## Table of Contents

- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Training Configuration](#training-configuration)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Data Augmentation Strategy](#data-augmentation-strategy)
- [Monitoring and Evaluation](#monitoring-and-evaluation)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

1. **Google Colab Account** with GPU runtime (T4 recommended)
2. **Roboflow Account** with HVAC dataset
3. **Google Drive** for checkpoint storage

### Running the Pipeline

1. Open `ai_model/notebooks/YOLOplan_pipeline_optimized.ipynb` in Google Colab
2. Set runtime to GPU: Runtime → Change runtime type → GPU (T4)
3. Add Roboflow credentials to Colab secrets:
   - `ROBOFLOW_API_KEY`: Your API key
   - `RF_WORKSPACE`: Workspace ID
   - `RF_PROJECT`: Project ID
   - `RF_VERSION`: Dataset version number
4. Run all cells sequentially

## Pipeline Architecture

### Stage 1: Environment Setup
- Clones YOLOplan repository
- Installs dependencies (ultralytics, roboflow, albumentations, tensorboard)
- Verifies GPU availability

### Stage 2: Data Download & Validation
- Downloads dataset from Roboflow in COCO format
- Validates dataset structure
- Generates statistics and visualizations
- Identifies potential issues (class imbalance, missing annotations)

### Stage 3: Data Preparation
- Converts COCO annotations to YOLO segmentation format
- Repairs file paths and validates integrity
- Creates YAML configuration file
- Organizes images and labels

### Stage 4: Training Configuration
- Generates comprehensive training config
- Sets hardware parameters optimized for T4 GPU
- Configures augmentation strategies
- Sets up learning rate scheduling

### Stage 5: Model Training
- Smart checkpoint resuming
- Progressive learning rate scheduling
- Advanced augmentation
- Real-time validation

### Stage 6: Monitoring
- TensorBoard integration
- Real-time metrics visualization
- Training curve analysis

### Stage 7: Evaluation
- Comprehensive model evaluation
- Per-class performance analysis
- Confusion matrix generation
- Results export

### Stage 8: Model Export
- ONNX export for production
- TorchScript export
- Model optimization

## Training Configuration

### Hardware Configuration (T4 GPU Optimized)

```yaml
hardware:
  imgsz: 1024      # Critical for small objects (valves, instruments)
  batch: 4         # Optimized for ~15GB T4 memory
  workers: 2       # Colab CPU constraint (2 vCPU)
  cache: False     # Prevent RAM overflow
  amp: True        # Mixed precision (2x speed boost)
  device: 0        # GPU device ID
```

**Rationale:**
- **1024px images**: Essential for detecting small HVAC symbols (needle valves, sensors)
- **Batch size 4**: Fills T4 VRAM without OOM crashes
- **No cache**: Prevents system RAM bottlenecks on Colab
- **AMP enabled**: Provides 2x training speed with minimal accuracy loss

### Training Parameters

```yaml
training:
  epochs: 100
  patience: 20         # Early stopping
  save_period: 5       # Checkpoint frequency
  close_mosaic: 15     # Disable mosaic in last 15 epochs
  optimizer: 'AdamW'   # Better than SGD for small datasets
  lr0: 0.001          # Initial learning rate
  lrf: 0.01           # Final LR multiplier (0.001 * 0.01 = 0.00001)
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3.0   # Gradual LR ramp-up
```

**Learning Rate Schedule:**
- Epochs 0-3: Linear warmup (0 → 0.001)
- Epochs 3-85: Cosine annealing (0.001 → 0.00001)
- Epochs 85-100: Fine-tuning at minimum LR

### Loss Weights

```yaml
loss_weights:
  box: 7.5      # Bounding box loss
  cls: 0.5      # Classification loss
  dfl: 1.5      # Distribution focal loss
  seg: 1.0      # Segmentation mask loss
```

**Tuning Guide:**
- Increase `box` if localization is poor
- Increase `cls` if classification accuracy is low
- Increase `seg` if mask quality is poor
- Keep total weight sum around 10.5

## Hyperparameter Optimization

### Critical Parameters by Use Case

#### Small Object Detection (Valves, Sensors)
```yaml
imgsz: 1024          # Higher resolution
copy_paste: 0.3      # Increase small object density
degrees: 10.0        # Handle scan skew
close_mosaic: 15     # Keep mosaic longer
```

#### Class Imbalance
```yaml
copy_paste: 0.5      # Aggressive augmentation
cls: 1.0             # Increase classification weight
# Consider weighted sampling in future versions
```

#### Large Dataset (>2000 images)
```yaml
batch: 8             # If GPU memory allows
epochs: 150          # More data needs more epochs
patience: 30         # Longer patience
cache: 'ram'         # If sufficient RAM (>16GB)
```

#### Limited Training Time
```yaml
epochs: 50
imgsz: 640           # Faster training
batch: 8
close_mosaic: 5
```

### Hyperparameter Search Strategy

For systematic optimization:

1. **Baseline Run**: Use default config
2. **Learning Rate**: Test [0.0001, 0.001, 0.01]
3. **Augmentation**: Vary mosaic, copy_paste, degrees
4. **Architecture**: Try yolo11n, yolo11s, yolo11m, yolo11l
5. **Batch Size**: Max out GPU memory

## Data Augmentation Strategy

### Geometric Augmentations

```yaml
augmentation:
  mosaic: 1.0          # Always enabled - teaches context
  copy_paste: 0.3      # Increases rare object density
  degrees: 10.0        # Handles scan skew (not rotation)
  translate: 0.1       # Small positional shifts
  scale: 0.5           # Scale variation
  fliplr: 0.5          # Horizontal flip (common in blueprints)
  flipud: 0.5          # Vertical flip (common in blueprints)
  shear: 0.0           # Disabled - destroys technical drawings
  perspective: 0.0     # Disabled - blueprints are 2D
  mixup: 0.0           # Disabled - destroys sharp edges
```

### Color Augmentations (Technical Drawing Specific)

```yaml
  hsv_h: 0.015         # Minimal hue variation
  hsv_s: 0.7           # Simulate faded ink
  hsv_v: 0.4           # Simulate dark scans/overexposed photos
```

**Why These Values:**
- Technical drawings have consistent line styles
- Color augmentation simulates real-world scan variations
- Aggressive color aug would destroy line clarity

### Advanced Augmentation with Albumentations

The pipeline supports Albumentations for:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Blur/Sharpen
- ISO Noise (simulates scanner artifacts)
- Grid Distortion (minimal)

Enable in config:
```yaml
augmentation:
  use_albumentations: True
  albumentations_p: 0.5
```

### Augmentation Best Practices

1. **Always keep Mosaic enabled** - It's the most effective for small objects
2. **Disable Mixup** - Destroys sharp edges in technical drawings
3. **Use Copy-Paste** - Essential for rare small symbols
4. **Moderate color augmentation** - Too much destroys line contrast
5. **Close Mosaic late** - Last 10-15 epochs for fine-tuning

## Monitoring and Evaluation

### TensorBoard Metrics

Key metrics to monitor:

1. **Training Loss**: Should decrease steadily
   - If plateaus early → increase LR
   - If unstable → decrease LR or batch size

2. **Validation mAP50**: Primary metric
   - Target: >0.85 for production
   - If validation >> training → underfitting
   - If training >> validation → overfitting

3. **Precision vs Recall**:
   - High precision, low recall → too conservative (increase IoU)
   - Low precision, high recall → too many FPs (decrease conf)

### Model Evaluation Checklist

After training:

- [ ] mAP50 > 0.80 on validation set
- [ ] mAP50-95 > 0.60
- [ ] No catastrophic class failures (mAP < 0.30)
- [ ] Precision/Recall balanced (within 0.1)
- [ ] Visual inspection of predictions
- [ ] Test on out-of-distribution samples

### Per-Class Analysis

Identify weak classes:
```python
# Classes with mAP50 < 0.70 need attention:
# 1. Collect more training examples
# 2. Increase copy_paste augmentation
# 3. Verify annotation quality
```

## Best Practices

### 1. Dataset Quality

- **Annotation Consistency**: Use same annotator or strict guidelines
- **Polygon Precision**: Tight polygons improve segmentation
- **Class Balance**: Aim for >50 examples per class minimum
- **Data Diversity**: Multiple blueprint styles, scan qualities

### 2. Training Strategy

- **Start Small**: Begin with yolo11s, scale up if needed
- **Progressive Training**: Train 50 epochs, analyze, continue if improving
- **Checkpoint Management**: Save best, last, and epoch-specific checkpoints
- **Version Control**: Tag dataset version + config in run name

### 3. Colab-Specific Tips

- **Save to Drive**: Always use Drive for PROJECT_DIR
- **Smart Resume**: Check for last.pt before starting
- **Session Management**: Training >6 hours? Use Colab Pro
- **Memory Management**: Clear outputs to prevent RAM issues

### 4. Production Deployment

- **Export Format**: Use ONNX for cross-platform compatibility
- **Quantization**: Consider INT8 quantization for edge devices
- **Batch Inference**: Use batch=16 for throughput
- **Confidence Threshold**: Start at 0.25, adjust based on precision/recall

## Troubleshooting

### Common Issues

#### 1. OOM (Out of Memory) Errors

**Symptoms**: `CUDA out of memory` error during training

**Solutions**:
```yaml
# Try in order:
batch: 2          # Reduce batch size
imgsz: 640        # Reduce image size
workers: 1        # Reduce data loading workers
amp: True         # Ensure AMP is enabled
```

#### 2. Low mAP

**Symptoms**: mAP50 < 0.70 after 50+ epochs

**Diagnosis**:
- Check training loss - if high, training not converged
- Check val loss vs train loss - if val >> train, overfitting
- Visualize predictions - are annotations correct?

**Solutions**:
```yaml
# If underfitting:
epochs: 150
lr0: 0.002
batch: 8

# If overfitting:
augment: True
mosaic: 1.0
copy_paste: 0.5
patience: 10
```

#### 3. Class Imbalance

**Symptoms**: Some classes perform well (mAP>0.9), others poorly (mAP<0.5)

**Solutions**:
1. Collect more examples of weak classes (target 100+ per class)
2. Increase copy_paste augmentation
3. Use focal loss (future enhancement)
4. Manually oversample weak classes in dataset

#### 4. Training Instability

**Symptoms**: Loss spikes, NaN values, unstable validation

**Solutions**:
```yaml
lr0: 0.0001        # Lower learning rate
warmup_epochs: 5   # Longer warmup
batch: 2           # Smaller batch
amp: False         # Disable if causing issues
```

#### 5. Slow Training

**Symptoms**: <30 it/s on T4 GPU

**Solutions**:
- Check GPU utilization: Should be >90%
- Reduce workers if CPU bottleneck
- Enable cache if RAM allows
- Ensure AMP is enabled
- Close unnecessary browser tabs

### Error Messages

#### "No labels found in dataset"
- Check YAML paths are correct
- Verify labels/ directory exists
- Ensure .txt files match image names

#### "Dataset not found"
- Verify DATASET_ROOT path
- Check YAML file paths are absolute
- Ensure train/valid splits exist

#### "Colab disconnected"
- Use smart resume (checks for last.pt)
- Save checkpoints to Drive frequently
- Consider Colab Pro for longer runtimes

## Advanced Topics

### Multi-GPU Training

For DGX or multi-GPU systems:

```python
model.train(
    device=[0,1,2,3],  # Use 4 GPUs
    batch=16,          # 4 per GPU
    # ... other args
)
```

### Transfer Learning from Custom Checkpoint

```python
# Instead of yolo11m-seg.pt:
model = YOLO('path/to/previous_best.pt')
model.train(
    # Reduce LR for fine-tuning
    lr0=0.0001,
    epochs=50
)
```

### Ensemble Models

Train multiple models with different configs, then ensemble:

```python
from ultralytics import YOLO

models = [
    YOLO('run1/best.pt'),
    YOLO('run2/best.pt'),
    YOLO('run3/best.pt')
]

# Average predictions (requires custom code)
```

## Validation Strategies

### K-Fold Cross-Validation

For small datasets (<500 images):

1. Split data into 5 folds
2. Train 5 models (each with different validation fold)
3. Average metrics across folds
4. Use best-performing fold for production

### Temporal Validation

For time-series data:

1. Train on older blueprints
2. Validate on newer blueprints
3. Ensures model generalizes to future data

## Performance Targets

### Production-Ready Thresholds

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| mAP50 | 0.75 | 0.85 | 0.92+ |
| mAP50-95 | 0.50 | 0.65 | 0.75+ |
| Precision | 0.75 | 0.85 | 0.90+ |
| Recall | 0.70 | 0.80 | 0.90+ |
| Inference Speed (T4) | 20 FPS | 30 FPS | 50+ FPS |

### Class-Specific Targets

- **Critical Classes** (valves, sensors): mAP50 > 0.85
- **Common Classes** (pipes, ducts): mAP50 > 0.80
- **Rare Classes**: mAP50 > 0.70 acceptable

## Next Steps

After completing this guide:

1. Run baseline training with default config
2. Analyze results and identify bottlenecks
3. Iterate on hyperparameters
4. Document learnings for team
5. Deploy best model to production

## Resources

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [YOLOplan Repository](https://github.com/DynMEP/YOLOplan)
- [HVAC Symbol Standards](../docs/HVAC_REFACTORING_GUIDE.md)
- [Roboflow Documentation](https://docs.roboflow.com/)

## Contributing

To improve this training pipeline:

1. Test new augmentation strategies
2. Benchmark different architectures
3. Optimize hyperparameters
4. Document findings
5. Submit PR with improvements

---

**Last Updated**: 2024-12-18  
**Version**: 1.0  
**Maintainer**: HVAC-AI Team
