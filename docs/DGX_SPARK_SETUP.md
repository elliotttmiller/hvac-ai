# NVIDIA DGX Spark Setup Guide for YOLOplan Pipeline

## Overview

This guide provides instructions for running the `YOLOplan_pipeline_DGX_Spark.ipynb` notebook on NVIDIA DGX infrastructure with Spark cluster management.

## Prerequisites

### Hardware Requirements
- **Platform**: NVIDIA DGX Station, DGX Server, or DGX SuperPOD with Spark
- **GPU**: 1-8x NVIDIA A100 (40/80GB) or H100 (80GB) recommended
- **RAM**: 256GB+ system RAM recommended
- **Storage**: High-speed NVMe storage for datasets (100GB+ free space)

### Software Requirements
- **OS**: Ubuntu 20.04 or 22.04 LTS
- **CUDA**: 11.8+ or 12.x
- **Python**: 3.8, 3.9, or 3.10
- **Jupyter**: JupyterLab or Jupyter Notebook

## Installation

### 1. Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv ~/hvac_venv
source ~/hvac_venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics YOLO
pip install ultralytics

# Install other dependencies
pip install roboflow pyyaml pandas matplotlib seaborn tensorboard tqdm jupyter
```

### 3. Verify GPU Setup

```bash
# Check NVIDIA driver and GPUs
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 4. Set Environment Variables

```bash
# Create ~/.bashrc additions or use a .env file
export ROBOFLOW_API_KEY="your-api-key-here"
export ROBOFLOW_WORKSPACE="hvac-detection"
export ROBOFLOW_PROJECT="hvac-blueprint-analysis"
export ROBOFLOW_VERSION="1"

# Optional: Set CUDA device visibility (to limit GPU usage)
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

## Usage

### Starting the Notebook

```bash
# Activate environment
source ~/hvac_venv/bin/activate

# Navigate to notebooks directory
cd /path/to/hvac-ai/ai_model/notebooks

# Launch Jupyter
jupyter notebook YOLOplan_pipeline_DGX_Spark.ipynb

# Or for JupyterLab
jupyter lab YOLOplan_pipeline_DGX_Spark.ipynb
```

### Running the Pipeline

Execute cells in order:

1. **Environment Verification** - Checks GPU and system resources
2. **Environment Setup** - Installs/updates packages
3. **Data Download** - Downloads dataset from Roboflow
4. **Dataset Repair** - Validates and repairs annotations
5. **COCO to YOLO Conversion** - Converts annotations to YOLO format
6. **Training Configuration** - Creates DGX-optimized config
7. **Training Execution** - Runs model training
8. **TensorBoard Monitoring** - Launches real-time monitoring
9. **Model Evaluation** - Evaluates trained model
10. **Model Export** - Exports to ONNX/TorchScript
11. **Summary** - Final instructions and best practices

## Configuration

### GPU Resource Management

The notebook automatically detects your GPU configuration and sets conservative defaults to avoid overloading Spark:

- **A100 40GB**: 12 batch per GPU, 1024 image size
- **A100 80GB**: 16 batch per GPU, 1280 image size
- **H100 80GB**: 20 batch per GPU, 1280 image size
- **Other GPUs**: 4 batch per GPU, 1024 image size

**Default Spark-friendly limits:**
- Maximum 4 GPUs used (even if more available)
- Conservative worker count based on CPU cores
- Disk-based caching (no RAM caching)

### Customizing Configuration

Edit the generated `training_config_dgx.yaml` file to customize:

```yaml
hardware:
  device: [0, 1, 2, 3]  # GPUs to use
  batch: 48             # Total batch size
  workers: 16           # Data loading workers
  imgsz: 1024          # Image size
  cache: false         # Set to true if enough RAM

training:
  epochs: 150          # Training epochs
  patience: 30         # Early stopping patience
  lr0: 0.001          # Initial learning rate
```

## Monitoring

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir ~/hvac_workspace/runs/segment --port 6006

# Access at http://localhost:6006
# Or via SSH tunnel:
ssh -L 6006:localhost:6006 user@dgx-server
```

### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvidia-smi in loop
nvidia-smi -l 1
```

### Spark Resource Monitoring

Check with your cluster administrator for Spark monitoring tools specific to your installation.

## Optimization Tips

### For Faster Training

1. **Increase batch size** (if GPU memory allows):
   ```yaml
   batch: 64  # or higher
   ```

2. **Use more GPUs**:
   ```yaml
   device: [0, 1, 2, 3, 4, 5, 6, 7]
   ```

3. **Enable caching** (if enough RAM):
   ```yaml
   cache: true
   ```

4. **Reduce image size** for faster iteration:
   ```yaml
   imgsz: 640  # instead of 1024
   ```

### For Better Accuracy

1. **Increase image size**:
   ```yaml
   imgsz: 1280  # or 1536
   ```

2. **Train longer**:
   ```yaml
   epochs: 300
   patience: 50
   ```

3. **Adjust augmentation**:
   ```yaml
   copy_paste: 0.5  # increase for small objects
   mosaic: 1.0      # keep enabled
   ```

### For Spark Cluster Compatibility

1. **Limit GPU usage**:
   ```yaml
   device: [0, 1]  # use only 2 GPUs
   ```

2. **Reduce worker count**:
   ```yaml
   workers: 8  # free up CPU cores
   ```

3. **Use smaller batch size**:
   ```yaml
   batch: 24  # total across GPUs
   ```

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `CUDA out of memory` error

**Solutions**:
1. Reduce batch size: `batch: 16`
2. Reduce image size: `imgsz: 640`
3. Use fewer GPUs: `device: [0]`
4. Disable cache: `cache: false`

### Slow Training

**Symptom**: Training is slower than expected

**Solutions**:
1. Verify AMP is enabled: `amp: true`
2. Enable cuDNN benchmark (already enabled)
3. Increase workers: `workers: 32`
4. Check data loading: `cache: true` if RAM available
5. Verify NVMe storage is being used

### Spark Resource Conflicts

**Symptom**: Cluster admin reports resource overuse

**Solutions**:
1. Reduce GPU count: `device: [0, 1, 2]`
2. Lower worker count: `workers: 8`
3. Add delays between training runs
4. Coordinate with cluster scheduler

### Dataset Issues

**Symptom**: No data or missing annotations

**Solutions**:
1. Verify Roboflow credentials are set
2. Check dataset downloaded to `~/hvac_workspace/dataset`
3. Verify COCO annotations exist
4. Re-run dataset repair cell

## Performance Benchmarks

Expected training performance on DGX infrastructure:

| GPU Model | Batch Size | Images/sec | Epoch Time (1000 imgs) |
|-----------|------------|------------|------------------------|
| A100 40GB | 12         | 50-60      | ~20 seconds           |
| A100 80GB | 16         | 70-80      | ~15 seconds           |
| H100 80GB | 20         | 100-120    | ~10 seconds           |
| 4x A100   | 48         | 200-240    | ~5 seconds            |

*Note: Times vary based on image size, augmentation, and model architecture*

## Best Practices

### Before Training
- [ ] Verify GPU availability: `nvidia-smi`
- [ ] Check disk space: `df -h ~/hvac_workspace`
- [ ] Set environment variables
- [ ] Coordinate with Spark cluster admin

### During Training
- [ ] Monitor TensorBoard for convergence
- [ ] Watch GPU memory usage
- [ ] Check for data loading bottlenecks
- [ ] Verify Spark resources are not exceeded

### After Training
- [ ] Evaluate model on test set
- [ ] Export to ONNX/TensorRT
- [ ] Back up best model weights
- [ ] Clean up old checkpoints
- [ ] Archive training logs

## Additional Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com/
- **NVIDIA DGX Documentation**: https://docs.nvidia.com/dgx/
- **YOLO11 Paper**: https://arxiv.org/abs/2304.00501
- **TensorRT Guide**: https://docs.nvidia.com/deeplearning/tensorrt/

## Support

For issues specific to this notebook:
1. Check the troubleshooting section above
2. Review `ai_model/OPTIMIZATION_GUIDE.md` in the repository
3. Consult with your DGX cluster administrator
4. Open an issue in the repository

## License

This notebook and guide are part of the HVAC AI project. See the main repository LICENSE file for details.
