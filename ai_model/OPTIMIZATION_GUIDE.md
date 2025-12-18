# HVAC AI Model Optimization Guide

## Executive Summary

This guide provides advanced optimization techniques for improving YOLO11 segmentation model performance on HVAC blueprint analysis. It covers data-centric, model-centric, and inference optimization strategies.

## Table of Contents

1. [Performance Optimization Workflow](#performance-optimization-workflow)
2. [Data-Centric Optimization](#data-centric-optimization)
3. [Model-Centric Optimization](#model-centric-optimization)
4. [Training Optimization](#training-optimization)
5. [Inference Optimization](#inference-optimization)
6. [Hardware-Specific Tuning](#hardware-specific-tuning)
7. [Advanced Techniques](#advanced-techniques)

## Performance Optimization Workflow

### Step-by-Step Optimization Process

```
1. Baseline Measurement
   ↓
2. Data Quality Analysis
   ↓
3. Augmentation Tuning
   ↓
4. Architecture Selection
   ↓
5. Hyperparameter Optimization
   ↓
6. Post-Processing Tuning
   ↓
7. Deployment Optimization
```

### Measurement Framework

Before optimizing, establish baseline metrics:

```python
# Baseline Metrics Template
baseline = {
    'training': {
        'time_per_epoch': None,  # seconds
        'final_train_loss': None,
        'GPU_memory_peak': None,  # GB
    },
    'validation': {
        'mAP50': None,
        'mAP50-95': None,
        'precision': None,
        'recall': None,
        'per_class_mAP': {},
    },
    'inference': {
        'fps': None,  # frames per second
        'latency_ms': None,  # milliseconds
        'GPU_memory': None,  # GB
    }
}
```

## Data-Centric Optimization

### 1. Data Quality Assessment

**Pre-Training Checklist:**

- [ ] All images have annotations
- [ ] Polygon annotations are precise (not just bounding boxes)
- [ ] No duplicate images in train/val splits
- [ ] Class distribution analyzed
- [ ] Annotation consistency verified

**Quality Metrics:**

```python
# Calculate annotation quality metrics
def assess_annotation_quality(dataset):
    metrics = {
        'avg_polygon_points': 0,  # More points = better precision
        'bbox_to_polygon_ratio': 0,  # Should be < 0.8
        'annotation_density': 0,  # Annotations per image
        'class_balance_ratio': 0,  # max_class / min_class
    }
    return metrics
```

**Thresholds:**
- Average polygon points: >8 (good), >12 (excellent)
- Bbox/polygon ratio: <0.8 (tight fit)
- Annotation density: 5-15 per image (optimal)
- Class balance: <10x (good), <5x (excellent)

### 2. Data Augmentation Optimization

**Augmentation Testing Protocol:**

1. **Baseline** (minimal augmentation):
```yaml
mosaic: 0.0
copy_paste: 0.0
degrees: 0.0
hsv_v: 0.0
```

2. **Geometric Only**:
```yaml
mosaic: 1.0
copy_paste: 0.3
degrees: 10.0
fliplr: 0.5
flipud: 0.5
```

3. **Full Augmentation**:
```yaml
# Add color augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

4. **Aggressive** (for small datasets <500 images):
```yaml
mosaic: 1.0
copy_paste: 0.5
degrees: 15.0
scale: 0.7
translate: 0.2
```

**Expected Results:**
- Minimal: Fast training, may overfit
- Geometric: Good generalization, slower training
- Full: Best generalization for varied data
- Aggressive: Best for limited data, slowest training

### 3. Class Imbalance Solutions

**Problem Detection:**
```python
# If max_class_count / min_class_count > 10:
# Apply solutions below
```

**Solution Hierarchy:**

1. **Data Collection** (Best):
   - Collect 50-100 more examples of rare classes
   - Use auto-labeling for similar images

2. **Copy-Paste Augmentation** (Good):
```yaml
copy_paste: 0.5  # Increase to 0.7 for severe imbalance
```

3. **Manual Oversampling** (Acceptable):
```python
# Duplicate images of rare classes in dataset
# Target: Balance to within 5x ratio
```

4. **Weighted Loss** (Future Enhancement):
```python
# Calculate class weights
weights = total_samples / (num_classes * class_samples)
# Apply in custom training loop
```

### 4. Data Cleaning Pipeline

**Automated Cleaning Script:**

```python
def clean_dataset(dataset_path):
    """
    Automated data cleaning pipeline
    """
    issues_found = []
    
    # 1. Check for corrupted images
    for img_path in get_all_images(dataset_path):
        if not verify_image(img_path):
            issues_found.append(f"Corrupted: {img_path}")
    
    # 2. Check for missing labels
    for img_path in get_all_images(dataset_path):
        label_path = img_to_label_path(img_path)
        if not os.path.exists(label_path):
            issues_found.append(f"Missing label: {img_path}")
    
    # 3. Check for empty labels
    for label_path in get_all_labels(dataset_path):
        if file_is_empty(label_path):
            issues_found.append(f"Empty label: {label_path}")
    
    # 4. Validate polygon coordinates
    for label_path in get_all_labels(dataset_path):
        if not validate_polygons(label_path):
            issues_found.append(f"Invalid polygons: {label_path}")
    
    return issues_found
```

## Model-Centric Optimization

### 1. Architecture Selection

**Performance vs Speed Trade-off:**

| Model | mAP50 | Speed (FPS) | Memory (GB) | Use Case |
|-------|-------|-------------|-------------|----------|
| yolo11n-seg | 0.75 | 60 | 2.5 | Edge devices, real-time |
| yolo11s-seg | 0.82 | 45 | 4.0 | Balanced |
| yolo11m-seg | 0.88 | 30 | 8.0 | **Recommended** |
| yolo11l-seg | 0.90 | 20 | 12.0 | High accuracy |
| yolo11x-seg | 0.92 | 12 | 16.0 | Research |

**Selection Criteria:**

- **Small Dataset** (<500 images): Start with yolo11s to avoid overfitting
- **Medium Dataset** (500-2000): Use yolo11m (optimal)
- **Large Dataset** (>2000): Try yolo11l for maximum accuracy
- **Real-time Required**: Use yolo11s or yolo11n
- **Highest Accuracy**: Use yolo11l or yolo11x

### 2. Transfer Learning Strategy

**Pre-training Options:**

1. **COCO Pre-trained** (Default):
   - Best for general object detection
   - May not recognize HVAC-specific symbols

2. **Custom Pre-training**:
   - Train on larger synthetic HVAC dataset first
   - Fine-tune on real blueprints
   - Requires 5000+ synthetic images

3. **Progressive Training**:
   ```
   Stage 1: Train on easy examples (50 epochs)
   Stage 2: Add hard examples (30 epochs)
   Stage 3: Fine-tune on full dataset (20 epochs)
   ```

### 3. Model Pruning

**For Deployment Optimization:**

```python
# Prune model to reduce size (future enhancement)
# Target: 30% size reduction with <2% mAP loss

# Steps:
# 1. Identify less important filters
# 2. Prune by importance score
# 3. Fine-tune pruned model
```

## Training Optimization

### 1. Learning Rate Optimization

**Learning Rate Finder:**

```python
# Test learning rates from 1e-6 to 1e-1
# Plot loss vs LR
# Select LR where loss decreases fastest
# Typically: 1e-3 to 1e-2 for YOLO11
```

**Optimal Schedules:**

1. **Cosine Annealing** (Recommended):
```yaml
lr0: 0.001
lrf: 0.01
# LR: 0.001 → 0.00001 over training
```

2. **Step Decay**:
```yaml
# Reduce LR by 10x at epochs 60, 80
# Requires custom callback
```

3. **OneCycle**:
```yaml
# Rapid warmup, long annealing
# Best for short training (<50 epochs)
```

### 2. Batch Size Optimization

**Rule of Thumb:**
```python
# Maximum batch size that fits in GPU memory
# With gradient accumulation if needed

optimal_batch = {
    'T4 (15GB)': 4,   # for imgsz=1024
    'A100 (40GB)': 12,
    'H100 (80GB)': 24,
}
```

**Gradient Accumulation:**

```python
# Simulate larger batch size
effective_batch = batch_size * accumulation_steps

# Example: batch=2, accumulate=4
# Effective batch = 8
# Use when GPU memory is limited
```

### 3. Mixed Precision Training

**Always Enable AMP:**

```yaml
amp: True  # 2x speed, minimal accuracy loss
```

**Benefits:**
- 50-100% faster training
- 30% lower memory usage
- <1% mAP degradation

**When to Disable:**
- Numerical instability (rare)
- NaN losses
- Debugging

### 4. Multi-Scale Training

**Implemented in YOLO by Default:**

```yaml
# Images randomly resized between:
# imgsz * 0.5 to imgsz * 1.5
# Example: 1024 → [512, 768, 1024, 1280, 1536]

# Improves scale invariance
# Adds 10-20% training time
```

### 5. Early Stopping Strategy

**Patience Tuning:**

```yaml
patience: 20  # Stop if no improvement in 20 epochs
```

**Guidelines:**
- Small dataset: patience=10-15
- Medium dataset: patience=20-25
- Large dataset: patience=30-40

**Save Best Model:**
```yaml
save_period: 5  # Save checkpoint every 5 epochs
# Always saves best.pt and last.pt
```

## Inference Optimization

### 1. Post-Processing Optimization

**NMS (Non-Maximum Suppression) Tuning:**

```python
# Adjust based on use case
inference_params = {
    # Conservative (fewer false positives)
    'conf': 0.35,
    'iou': 0.50,
    
    # Balanced (recommended)
    'conf': 0.25,
    'iou': 0.45,
    
    # Aggressive (more detections)
    'conf': 0.15,
    'iou': 0.40,
}
```

**Trade-offs:**
- Higher conf → Fewer FPs, lower recall
- Lower conf → More FPs, higher recall
- Higher IoU → More overlapping boxes
- Lower IoU → Fewer overlapping boxes

### 2. Batch Inference

**For Production:**

```python
# Process multiple images at once
results = model.predict(
    images_list,
    batch=16,  # Process 16 images simultaneously
    stream=True  # Memory efficient
)
```

**Benefits:**
- 2-3x throughput improvement
- Better GPU utilization
- Amortize overhead

### 3. Model Export Optimization

**ONNX Export (Recommended):**

```python
model.export(
    format='onnx',
    imgsz=1024,
    optimize=True,   # Apply graph optimizations
    simplify=True,   # Simplify compute graph
    dynamic=True,    # Support dynamic batch sizes
    half=False       # Use FP16 (test carefully)
)
```

**TensorRT Export (Fastest):**

```python
# Requires NVIDIA GPU + TensorRT
model.export(
    format='engine',
    imgsz=1024,
    half=True,  # FP16 precision
    workspace=4  # GB
)
# 2-3x faster than ONNX
```

### 4. Quantization

**INT8 Quantization:**

```python
# For edge deployment
# Reduces model size by 4x
# ~10% speed improvement
# ~2-5% mAP loss

# Requires calibration dataset
model.export(
    format='onnx',
    int8=True,
    data='calibration.yaml'
)
```

## Hardware-Specific Tuning

### Google Colab T4 (15GB)

**Optimal Configuration:**

```yaml
hardware:
  imgsz: 1024
  batch: 4
  workers: 2
  cache: False
  amp: True

training:
  epochs: 100
  close_mosaic: 15
```

**Expected Performance:**
- Training: ~80 seconds/epoch (1000 images)
- Inference: 30-35 FPS

### NVIDIA A100 (40GB)

**Optimal Configuration:**

```yaml
hardware:
  imgsz: 1280  # Higher resolution
  batch: 12
  workers: 8
  cache: 'ram'  # If sufficient system RAM
  amp: True

training:
  epochs: 150
  close_mosaic: 20
```

**Expected Performance:**
- Training: ~40 seconds/epoch (1000 images)
- Inference: 50-60 FPS

### DGX Station (8x A100)

**Multi-GPU Configuration:**

```python
model.train(
    device=[0,1,2,3,4,5,6,7],  # All 8 GPUs
    batch=96,  # 12 per GPU
    workers=32,
    # ... other args
)
```

**Expected Performance:**
- Training: ~8 seconds/epoch (1000 images)
- Near-linear scaling up to 8 GPUs

## Advanced Techniques

### 1. Pseudo-Labeling

**For Semi-Supervised Learning:**

```python
# 1. Train model on labeled data
model = YOLO('yolo11m-seg.pt')
model.train(data='labeled_data.yaml', epochs=100)

# 2. Generate predictions on unlabeled data
results = model.predict('unlabeled_images/', conf=0.8)

# 3. Convert high-confidence predictions to labels
for result in results:
    if result.boxes.conf.min() > 0.8:
        save_as_label(result)

# 4. Retrain on labeled + pseudo-labeled data
model.train(data='combined_data.yaml', epochs=50)
```

**Best Practices:**
- Use high confidence threshold (>0.8)
- Manually review samples
- Iterate 2-3 times maximum

### 2. Knowledge Distillation

**Train Smaller Model from Larger:**

```python
# 1. Train large teacher model
teacher = YOLO('yolo11l-seg.pt')
teacher.train(data='data.yaml', epochs=100)

# 2. Train small student model with teacher guidance
# (Requires custom implementation)
student = YOLO('yolo11s-seg.pt')
# Use teacher predictions as soft labels
# Combine with ground truth labels
```

**Benefits:**
- Student model: 80-90% of teacher accuracy
- 3-4x faster inference
- Useful for edge deployment

### 3. Test-Time Augmentation (TTA)

**For Maximum Accuracy:**

```python
# Apply multiple augmentations at inference
# Average predictions

results = model.predict(
    image,
    augment=True  # Enables TTA
)

# Typically: +1-2% mAP
# 3-5x slower inference
```

**Use Cases:**
- Competition submissions
- Critical quality checks
- Not for production (too slow)

### 4. Active Learning

**Iterative Dataset Improvement:**

```
1. Train initial model on small dataset
2. Predict on large unlabeled pool
3. Select most uncertain predictions
4. Manually label selected images
5. Add to training set
6. Retrain and repeat
```

**Uncertainty Metrics:**
- Low confidence predictions
- High variance across ensemble
- Near-threshold detections

### 5. Ensemble Methods

**Multiple Model Fusion:**

```python
# Train 3 models with different configs
models = [
    YOLO('run1_yolo11m/best.pt'),
    YOLO('run2_yolo11l/best.pt'),
    YOLO('run3_yolo11m_aug/best.pt'),
]

# Combine predictions (requires custom code)
# Methods: NMS, weighted averaging, voting
```

**Expected Gains:**
- +2-5% mAP improvement
- 3x slower inference
- Use for competitions or critical applications

## Optimization Checklist

### Before Training
- [ ] Dataset cleaned and validated
- [ ] Class distribution analyzed
- [ ] Baseline metrics recorded
- [ ] Configuration saved with version control
- [ ] GPU availability confirmed

### During Training
- [ ] Monitor training loss (should decrease)
- [ ] Monitor validation mAP (should increase)
- [ ] Check for overfitting (train vs val gap)
- [ ] Verify augmentation is working (visualize)
- [ ] Save checkpoints regularly

### After Training
- [ ] Compare against baseline
- [ ] Analyze per-class performance
- [ ] Test on unseen data
- [ ] Optimize post-processing
- [ ] Export optimized model
- [ ] Document results

## Troubleshooting Performance Issues

### Issue: Low mAP (<0.70)

**Diagnosis:**
1. Check training loss - is it still decreasing?
2. Visualize predictions - are annotations correct?
3. Check class balance - is it severe (>10x)?
4. Verify augmentation is enabled

**Solutions:**
```yaml
# Try in order:
epochs: 150          # More training
lr0: 0.002          # Higher learning rate
copy_paste: 0.5     # More augmentation
imgsz: 1280         # Higher resolution
architecture: yolo11l-seg.pt  # Bigger model
```

### Issue: Overfitting

**Symptoms:**
- Training mAP >> Validation mAP (>0.1 difference)
- Training loss much lower than validation loss

**Solutions:**
```yaml
augment: True
mosaic: 1.0
copy_paste: 0.5
degrees: 15.0
hsv_v: 0.5
patience: 10        # Earlier stopping
```

### Issue: Slow Training

**Diagnosis:**
1. Check GPU utilization: `nvidia-smi`
2. Check CPU: Should not be bottleneck
3. Check I/O: Is data loading slow?

**Solutions:**
```yaml
workers: 8          # More data loading threads
cache: 'ram'        # If sufficient RAM
batch: 8            # Larger batch (if GPU allows)
amp: True           # Ensure enabled
```

### Issue: Poor Performance on Specific Classes

**Diagnosis:**
1. Check number of training examples
2. Check annotation quality for that class
3. Check if class is visually similar to others

**Solutions:**
1. Collect 50-100 more examples
2. Review and fix annotations
3. Increase copy_paste augmentation
4. Consider merging similar classes

## Performance Benchmarks

### Expected Performance by Dataset Size

| Dataset Size | mAP50 (yolo11m) | Training Time (T4) | Epochs |
|--------------|-----------------|-------------------|--------|
| 100 images | 0.65-0.75 | 30 min | 100 |
| 500 images | 0.75-0.85 | 2 hours | 100 |
| 1000 images | 0.82-0.90 | 4 hours | 100 |
| 2000 images | 0.87-0.93 | 8 hours | 150 |
| 5000+ images | 0.90-0.95 | 20+ hours | 200 |

### Inference Speed by Hardware

| Hardware | FPS (1024px) | Batch=1 | Batch=16 |
|----------|--------------|---------|----------|
| T4 GPU | 30 | 35 | 80 |
| A100 GPU | 60 | 70 | 180 |
| RTX 4090 | 75 | 85 | 220 |
| CPU (16 core) | 2 | 2 | 4 |

## Next Steps

1. Implement baseline training with default config
2. Measure and document performance
3. Apply optimization techniques systematically
4. Document improvements
5. Share findings with team

---

**Last Updated**: 2024-12-18  
**Version**: 1.0  
**Maintainer**: HVAC-AI Team
