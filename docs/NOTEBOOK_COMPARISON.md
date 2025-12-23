# YOLOplan Pipeline Notebook Comparison

## Overview

This document compares the two YOLOplan training notebooks in this directory:

1. **YOLOplan_pipeline_optimized.ipynb** - Optimized for Google Colab
2. **YOLOplan_pipeline_DGX_Spark.ipynb** - Optimized for NVIDIA DGX Spark (NEW)

## Key Differences

### Platform & Infrastructure

| Feature | Google Colab | DGX Spark |
|---------|--------------|-----------|
| **Target Platform** | Google Colab (Cloud) | NVIDIA DGX (On-premise) |
| **GPU Type** | T4, V100, A100 (shared) | A100, H100 (dedicated) |
| **GPU Count** | 1 GPU | 1-8 GPUs (configurable) |
| **Storage** | Google Drive + /content | Local NVMe storage |
| **RAM** | 12-32GB (shared) | 256GB+ (dedicated) |
| **Cluster Management** | N/A | Spark cluster aware |

### Configuration & Setup

#### Google Colab Version
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Get credentials from Colab secrets
from google.colab import userdata
RF_API_KEY = userdata.get('RF_API_KEY')

# Paths
PROJECT_DIR = '/content/drive/MyDrive/hvac_detection_project'
DATA_PATH = '/content/dataset'
```

#### DGX Spark Version
```python
# Environment verification (no mounting)
import torch
gpu_count = torch.cuda.device_count()

# Get credentials from environment or prompt
RF_API_KEY = os.environ.get('ROBOFLOW_API_KEY')
if not RF_API_KEY:
    RF_API_KEY = getpass.getpass("Roboflow API Key: ")

# Paths
workspace_dir = os.path.expanduser('~/hvac_workspace')
PROJECT_DIR = os.path.join(workspace_dir, 'runs', 'segment')
```

### Hardware Configuration

#### Google Colab Version
```yaml
hardware:
  imgsz: 1024
  batch: 4           # Limited by T4 GPU
  workers: 2         # Limited CPU cores
  cache: False       # Prevent RAM overflow
  amp: True
  device: 0          # Single GPU only
```

#### DGX Spark Version
```yaml
hardware:
  imgsz: 1024-1280   # Auto-scaled based on GPU
  batch: 48          # 12 per GPU x 4 GPUs (example)
  workers: 16        # Scaled with CPU cores
  cache: False       # Spark-friendly (disk-based)
  amp: True
  device: [0,1,2,3]  # Multi-GPU support (limited to 4 for Spark)
  cudnn_benchmark: True  # Additional optimization
```

### Training Configuration

#### Google Colab Version
```yaml
training:
  epochs: 100
  patience: 20
  save_period: 5
  optimizer: 'AdamW'
  lr0: 0.001
  warmup_epochs: 3.0
```

#### DGX Spark Version
```yaml
training:
  epochs: 150        # More epochs (faster training)
  patience: 30       # Increased patience
  save_period: 10    # Less frequent saves
  optimizer: 'AdamW'
  lr0: 0.001
  warmup_epochs: 5.0 # Longer warmup for stability
  cos_lr: True       # Cosine learning rate schedule
```

### Resource Management

#### Google Colab
- **Goal**: Maximize use of limited free/paid resources
- **Strategy**: Conservative settings to avoid disconnection
- **Monitoring**: Basic (Colab runtime warnings)

#### DGX Spark
- **Goal**: Balance performance with cluster fairness
- **Strategy**: Auto-detect hardware, set Spark-friendly limits
- **Monitoring**: TensorBoard + nvidia-smi + Spark dashboards

### Augmentation Differences

Both notebooks use similar augmentation strategies optimized for HVAC blueprints, but DGX version has slight adjustments:

| Parameter | Colab | DGX Spark |
|-----------|-------|-----------|
| copy_paste | 0.3 | 0.4 (higher for small objects) |
| degrees | 10.0 | 15.0 (more rotation) |
| albumentations_p | 0.5 | 0.6 (higher probability) |

### Export & Deployment

#### Google Colab
```python
# Export for download
onnx_path = model.export(format='onnx')
# Download manually from Colab
```

#### DGX Spark
```python
# Export with TensorRT option
onnx_path = model.export(format='onnx')
trt_path = model.export(format='engine', half=True)  # TensorRT
```

## When to Use Each Notebook

### Use Google Colab Version When:
- ✅ You don't have access to dedicated GPU infrastructure
- ✅ You want quick experimentation with free/paid GPUs
- ✅ Dataset size is small-medium (<10K images)
- ✅ You need to share work easily (Google Drive integration)
- ✅ You're doing proof-of-concept or learning

### Use DGX Spark Version When:
- ✅ You have access to NVIDIA DGX infrastructure
- ✅ You need multi-GPU training for large datasets
- ✅ You want maximum training speed (10-100x faster)
- ✅ You're working in a Spark cluster environment
- ✅ You need enterprise-grade model training
- ✅ You have large datasets (>10K images)
- ✅ You need to avoid cloud storage/transfer costs

## Performance Comparison

### Training Speed (Example: 1000 images, 1024px)

| Platform | GPUs | Batch | Time/Epoch | Speedup |
|----------|------|-------|------------|---------|
| Colab (T4) | 1 | 4 | ~120s | 1x |
| Colab (V100) | 1 | 8 | ~80s | 1.5x |
| Colab (A100) | 1 | 12 | ~50s | 2.4x |
| DGX (A100) | 1 | 12 | ~20s | 6x |
| DGX (4x A100) | 4 | 48 | ~5s | 24x |
| DGX (8x A100) | 8 | 96 | ~3s | 40x |

*Note: Actual performance varies based on dataset, model, and augmentation*

### Cost Comparison (100 epoch training)

| Platform | GPU Time | Cloud Cost | Local Cost |
|----------|----------|------------|------------|
| Colab Free | 3-5 hours | $0 (with limits) | N/A |
| Colab Pro+ | 2-3 hours | ~$50/month | N/A |
| DGX (1 GPU) | 30-60 min | N/A | Amortized HW cost |
| DGX (4 GPUs) | 10-15 min | N/A | Amortized HW cost |

## Migration Guide

### From Colab to DGX Spark

1. **Environment Setup**:
   ```bash
   # On DGX, create environment
   python3 -m venv ~/hvac_venv
   source ~/hvac_venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export ROBOFLOW_API_KEY="your-key"
   export ROBOFLOW_WORKSPACE="your-workspace"
   export ROBOFLOW_PROJECT="your-project"
   ```

3. **Update Paths**:
   - Replace Google Drive paths with local paths
   - Use `~/hvac_workspace` instead of `/content/drive/...`

4. **Adjust Configuration**:
   - Increase batch size based on GPU memory
   - Enable multi-GPU if available
   - Adjust workers based on CPU cores

5. **Launch Notebook**:
   ```bash
   jupyter lab YOLOplan_pipeline_DGX_Spark.ipynb
   ```

### From DGX Spark to Colab

1. **Copy Model Weights**:
   ```bash
   # Download from DGX
   scp dgx-server:~/hvac_workspace/runs/segment/*/weights/best.pt .
   ```

2. **Upload to Google Drive**:
   - Upload `best.pt` to your Google Drive
   - Reference in Colab notebook

3. **Use for Inference**:
   ```python
   # In Colab
   model = YOLO('/content/drive/MyDrive/models/best.pt')
   results = model.predict('image.jpg')
   ```

## Best Practices

### For Both Notebooks
- ✅ Always validate dataset before training
- ✅ Monitor training metrics in TensorBoard
- ✅ Start with lower epochs for testing
- ✅ Back up best model weights
- ✅ Use version control for configurations

### Colab-Specific
- ✅ Save checkpoints frequently to Drive
- ✅ Monitor runtime disconnections
- ✅ Use Pro/Pro+ for serious work
- ✅ Be aware of usage limits

### DGX Spark-Specific
- ✅ Coordinate with cluster admin
- ✅ Monitor GPU usage: `nvidia-smi`
- ✅ Respect Spark resource limits
- ✅ Use checkpointing for long training
- ✅ Clean up old runs regularly

## Support

- **Colab Issues**: Check [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **DGX Issues**: Consult `ai_model/notebooks/DGX_SPARK_SETUP.md`
- **General Training**: See `ai_model/OPTIMIZATION_GUIDE.md`

## Conclusion

Both notebooks serve important purposes:

- **Colab version** is ideal for accessibility, learning, and rapid prototyping
- **DGX Spark version** is designed for production-scale training with enterprise infrastructure

Choose the version that best fits your hardware access, dataset size, and performance requirements.
