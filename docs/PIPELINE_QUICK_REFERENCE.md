# YOLO Pipeline Comparison Quick Reference

## Available Pipelines

| Pipeline | File Name | Purpose | Augmentation | Best For |
|----------|-----------|---------|--------------|----------|
| **Optimized** | `YOLOplan_pipeline_optimized.ipynb` | Full-featured training with augmentation | ✅ Enabled (10+ techniques) | General training, small-medium datasets |
| **No-Augmentation** | `YOLOplan_pipeline_no_augmentation.ipynb` | Pure training on exact dataset | ❌ Completely disabled | Pre-processed datasets, controlled experiments |
| **DGX Spark** | `YOLOplan_pipeline_DGX_Spark.ipynb` | High-performance multi-GPU | ✅ Enabled | DGX systems, large-scale training |

## Quick Decision Tree

```
Do you have a pre-augmented dataset?
├─ YES → Use YOLOplan_pipeline_no_augmentation.ipynb
└─ NO
   └─ Is your dataset < 1000 images?
      ├─ YES → Use YOLOplan_pipeline_optimized.ipynb
      └─ NO
         └─ Do you have DGX/multi-GPU?
            ├─ YES → Use YOLOplan_pipeline_DGX_Spark.ipynb
            └─ NO → Use YOLOplan_pipeline_optimized.ipynb
```

## Feature Comparison

| Feature | Optimized | No-Aug | DGX Spark |
|---------|-----------|--------|-----------|
| Data Augmentation | ✅ Full | ❌ None | ✅ Full |
| Google Colab T4 | ✅ Yes | ✅ Yes | ❌ No |
| Multi-GPU | ❌ No | ❌ No | ✅ Yes |
| TensorBoard | ✅ Yes | ✅ Yes | ✅ Yes |
| Model Export | ✅ ONNX/TS | ✅ ONNX/TS | ✅ ONNX/TS |
| Learning Rate Schedule | ✅ Yes | ✅ Yes | ✅ Yes |
| Mixed Precision (AMP) | ✅ Yes | ✅ Yes | ✅ Yes |

## Augmentation Details

### Optimized Pipeline
Enabled augmentations:
- **Geometric:** Mosaic (1.0), Copy-paste (0.3), Rotation (10°), Translation (0.1), Scale (0.5), Flip H/V (0.5)
- **Color:** HSV variation (minimal)
- **Advanced:** Albumentations support

### No-Augmentation Pipeline
All augmentations disabled:
- **augment:** False (master switch)
- **All geometric transforms:** 0.0
- **All color transforms:** 0.0
- **Albumentations:** Not installed

### DGX Spark Pipeline
Enhanced augmentations for large-scale training:
- All optimized pipeline augmentations
- Multi-GPU distributed training
- Advanced memory management

## When to Switch Pipelines

### From Optimized to No-Augmentation
✅ You've pre-augmented your dataset externally  
✅ You need exact reproducibility  
✅ You're conducting controlled experiments  
✅ Your augmented dataset is large enough  

### From No-Augmentation to Optimized
✅ Validation performance is poor  
✅ Model is overfitting (train/val gap)  
✅ Dataset lacks diversity  
✅ You want automatic generalization  

### From Colab to DGX Spark
✅ You have access to DGX/multi-GPU system  
✅ Dataset is very large (>10K images)  
✅ You need faster training  
✅ You're doing production-scale training  

## Runtime Requirements

### Optimized & No-Augmentation
- **Platform:** Google Colab
- **GPU:** T4 (free tier compatible)
- **RAM:** 12.7 GB (standard)
- **Setup Time:** ~3 minutes
- **Training Speed:** ~4 hours per 100 epochs (1K images)

### DGX Spark
- **Platform:** DGX Station/Server
- **GPU:** Multiple A100/V100
- **RAM:** 256+ GB
- **Setup Time:** ~10 minutes
- **Training Speed:** ~1 hour per 100 epochs (1K images)

## Getting Started

### 1. Choose Your Pipeline
Use the decision tree above to select the right pipeline.

### 2. Open in Google Colab (for T4 pipelines)
```
File > Upload Notebook > Select your chosen .ipynb file
Runtime > Change runtime type > T4 GPU
```

### 3. Set Credentials
```python
# Add to Colab Secrets
RF_API_KEY = "your_roboflow_api_key"
RF_WORKSPACE = "your_workspace"
RF_PROJECT = "your_project"  
RF_VERSION = "1"
```

### 4. Run Sequential Cells
Execute cells from top to bottom, monitoring progress in TensorBoard.

## Troubleshooting

### Issue: Overfitting
**Symptoms:** Train loss << Val loss  
**Solutions:**
- Switch to optimized pipeline (if using no-aug)
- Add more training data
- Increase patience parameter

### Issue: Poor Validation Performance
**Symptoms:** Low mAP, poor generalization  
**Solutions:**
- If using no-aug: Check dataset quality and diversity
- If using optimized: Increase augmentation parameters
- Verify annotations are correct

### Issue: OOM Errors
**Symptoms:** CUDA out of memory  
**Solutions:**
- Reduce batch size (4 → 2)
- Reduce image size (1024 → 640)
- Disable caching (set cache: False)

### Issue: Slow Training
**Symptoms:** Low GPU utilization  
**Solutions:**
- Enable AMP if not already (amp: True)
- Increase batch size if memory allows
- Consider DGX pipeline for large datasets

## Performance Expectations

### With No-Augmentation
- **Training:** Faster per epoch (~10-15% faster)
- **Convergence:** Earlier plateau (expected)
- **Final mAP:** Depends on dataset quality
- **Overfitting Risk:** Higher
- **Best for:** Large, diverse pre-processed datasets

### With Augmentation (Optimized)
- **Training:** Slightly slower per epoch
- **Convergence:** Gradual improvement
- **Final mAP:** +2-3% improvement typically
- **Overfitting Risk:** Lower
- **Best for:** Small-medium datasets, general use

## Related Documentation

- `NO_AUGMENTATION_PIPELINE.md` - Detailed no-augmentation guide
- `PIPELINE_COMPARISON.md` - In-depth technical comparison
- `NOTEBOOK_COMPARISON.md` - Feature-by-feature analysis
- `DGX_SPARK_SETUP.md` - DGX pipeline setup guide

## Support

For issues or questions:
1. ✅ Check this quick reference first
2. ✅ Review detailed documentation for your pipeline
3. ✅ Check inline comments in the notebook
4. ✅ Review YOLO11 documentation

---

**Version:** 1.0  
**Last Updated:** 2024-12-19  
**Maintained by:** HVAC-AI Team
