# Autodistill HVAC Pipeline Execution Guide

## Overview

This directory contains the production-grade HVAC auto-labeling pipeline using Grounded-SAM-2 and YOLOv8. The pipeline has been fully implemented and tested for structure, but requires specific hardware and environment to execute.

## Files

- **`run_autodistill_pipeline.py`**: Complete end-to-end pipeline script
- **`requirements_autodistill.txt`**: Python dependencies
- **`notebooks/autodistill_hvac_grounded_sam2.ipynb`**: Original Jupyter notebook

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (required)
  - Minimum 8GB VRAM recommended
  - The pipeline uses Florence-2, SAM-2, and YOLOv8 which require GPU acceleration
- **Disk Space**: Minimum 30GB free space
  - Model weights: ~15-20GB (Florence-2, SAM-2, YOLOv8)
  - Dependencies: ~10GB (PyTorch, CUDA libraries)
  - Working data: ~5GB (datasets, outputs)
- **RAM**: Minimum 16GB recommended

### Software Requirements
- **Python**: 3.8+ (tested with 3.12)
- **CUDA**: 11.8+ or 12.x
- **Operating System**: Linux (Ubuntu 20.04+ recommended)

## Installation

### 1. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Autodistill Dependencies

Following the [official autodistill-grounded-sam-2 repository](https://github.com/autodistill/autodistill-grounded-sam-2):

```bash
pip3 install -r ai_model/requirements_autodistill.txt
```

This installs:
- `autodistill` - Core framework
- `autodistill-grounded-sam-2` - Grounded SAM-2 base model
- `autodistill-yolov8` - YOLOv8 target model
- `supervision` - Detection handling
- Supporting libraries (transformers, opencv, matplotlib, etc.)

### 3. Install Additional Dependencies

```bash
pip3 install hydra-core omegaconf iopath scikit-learn
```

### 4. Fix Compatibility Issue (if needed)

If you encounter `AdamW` import error from transformers, patch the autodistill-florence-2 module:

```python
# Edit: ~/.local/lib/python3.12/site-packages/autodistill_florence_2/model.py
# Change line 15 from:
#   from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler)
# To:
from transformers import (AutoModelForCausalLM, AutoProcessor, get_scheduler)
from torch.optim import AdamW
```

## Dataset Structure

The pipeline expects the following directory structure:

```
ai_model/
├── datasets/
│   ├── hvac_templates/
│   │   └── hvac_templates/
│   │       ├── template_instrument_*.PNG
│   │       ├── template_signal_*.PNG
│   │       └── template_valve_*.PNG
│   └── hvac_example_images/
│       └── hvac_example_images/
│           ├── src1_train_*.jpg
│           └── ...
└── outputs/  (created automatically)
    ├── autodistill_dataset/
    ├── yolov8_training/
    └── inference_results/
```

### Current Dataset
- **Templates**: 18 HVAC symbol templates in `hvac_templates/`
- **Examples**: 5 HVAC blueprint images in `hvac_example_images/`

## Execution

### Quick Start

```bash
cd /path/to/hvac-ai
python3 ai_model/run_autodistill_pipeline.py
```

### Pipeline Phases

The script executes 7 phases automatically:

1. **Configuration** - Set paths, parameters, logging
2. **Ontology Generation** - Build class ontology from templates
3. **Auto-Labeling** - Generate annotations with Grounded-SAM-2
4. **Quality Review** - Compute dataset statistics
5. **Training** - Train YOLOv8 model (1 epoch for testing)
6. **Inference** - Test trained model on example images
7. **Summary** - Report metrics and timings

### Configuration

Key parameters in `run_autodistill_pipeline.py`:

```python
# Training Configuration
TRAINING_EPOCHS = 1  # Set to 1 for testing (increase for production)
YOLO_MODEL_SIZE = "yolov8n.pt"  # Nano model for faster training

# Detection Parameters (optimized for HVAC blueprints)
BOX_THRESHOLD = 0.27  # Bounding box confidence threshold
TEXT_THRESHOLD = 0.22  # Text prompt matching threshold

# Paths (automatically configured)
TEMPLATE_FOLDER_PATH = "ai_model/datasets/hvac_templates/hvac_templates"
UNLABELED_IMAGES_PATH = "ai_model/datasets/hvac_example_images/hvac_example_images"
DATASET_OUTPUT_PATH = "ai_model/outputs/autodistill_dataset"
TRAINING_OUTPUT_PATH = "ai_model/outputs/yolov8_training"
INFERENCE_OUTPUT_PATH = "ai_model/outputs/inference_results"
```

## Expected Output

### Console Output

The pipeline provides detailed progress reporting:

```
======================================================================
🚀 HVAC AUTO-LABELING PIPELINE - AUTOMATED EXECUTION
======================================================================
ℹ️  Running in local environment
📂 Working Directory: /path/to/hvac-ai

⚙️  Pipeline Configuration:
   • Auto-proceed to training: True
   • Training epochs: 1

======================================================================
📝 SETTING UP LOGGING SYSTEM
======================================================================
✅ Logging system initialized
   • Log file: /path/to/hvac-ai/pipeline_logs/autodistill_pipeline_YYYYMMDD_HHMMSS.log

...
```

### Generated Files

#### 1. Autodistill Dataset (`ai_model/outputs/autodistill_dataset/`)
```
autodistill_dataset/
├── data.yaml           # YOLOv8 dataset configuration
├── train/
│   ├── images/         # Training images
│   └── labels/         # YOLO format annotations
└── valid/
    ├── images/         # Validation images
    └── labels/         # YOLO format annotations
```

#### 2. Training Output (`ai_model/outputs/yolov8_training/train*/`)
```
train/
├── weights/
│   ├── best.pt         # Best model checkpoint
│   └── last.pt         # Last epoch checkpoint
├── results.png         # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── PR_curve.png
```

#### 3. Inference Results (`ai_model/outputs/inference_results/`)
```
inference_results/
└── inference_result_*.jpg  # Annotated inference images
```

#### 4. Logs (`pipeline_logs/`)
```
pipeline_logs/
└── autodistill_pipeline_YYYYMMDD_HHMMSS.log
```

### Metrics Reported

The pipeline tracks and reports:

- **Configuration**: Paths, thresholds, epochs
- **Ontology**: Template count, class count
- **Labeling**: Processing time, detection count, confidence scores
- **Dataset**: Images processed, detections found, class distribution
- **Training**: Training time, model checkpoints
- **Inference**: Detection count, inference time

Example summary:

```
======================================================================
📊 PIPELINE EXECUTION SUMMARY
======================================================================

⏱️  Total Pipeline Time: 15.42 minutes

🔄 Phase Breakdown:
   • Configuration               2.15s
   • Ontology Generation         3.28s
   • Auto-Labeling             450.67s
   • Quality Review              5.43s
   • Training                  375.12s
   • Inference                   8.92s

📈 Key Metrics:
   • Template Files Found        18
   • Ontology Classes            18
   • Images to Label             5
   • Total Detections            87
   • Training Epochs             1
   • Inference Detections        23
```

## Troubleshooting

### Common Issues

#### 1. No NVIDIA GPU Found
```
RuntimeError: Found no NVIDIA driver on your system
```
**Solution**: The pipeline requires an NVIDIA GPU with CUDA support. Run on a machine with proper GPU hardware.

#### 2. Out of Disk Space
```
OSError: [Errno 28] No space left on device
```
**Solution**: Ensure at least 30GB free disk space before running.

#### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size in training
- Use smaller YOLOv8 model (yolov8n.pt instead of yolov8m.pt)
- Close other GPU-intensive applications

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'X'
```
**Solution**: Install missing dependencies:
```bash
pip3 install hydra-core omegaconf iopath scikit-learn
```

### Performance Optimization

For faster execution:
1. **Use GPU**: Essential for reasonable performance
2. **Reduce epochs**: Use `TRAINING_EPOCHS = 1` for testing
3. **Smaller model**: Use `yolov8n.pt` instead of larger variants
4. **Fewer images**: Test with a subset of images first

For better accuracy:
1. **Increase epochs**: Set `TRAINING_EPOCHS = 50-100` for production
2. **More training data**: Add more example images
3. **Tune thresholds**: Adjust `BOX_THRESHOLD` and `TEXT_THRESHOLD`
4. **Larger model**: Use `yolov8m.pt` or `yolov8l.pt` if resources allow

## Alternative: Google Colab

For systems without GPU, consider running in Google Colab:

1. Upload the notebook: `ai_model/notebooks/autodistill_hvac_grounded_sam2.ipynb`
2. Upload datasets to Google Drive
3. Mount Google Drive in Colab
4. Run cells sequentially

## Reference Documentation

- [Autodistill Official Docs](https://autodistill.github.io/autodistill/)
- [Grounded-SAM-2 Docs](https://autodistill.github.io/autodistill/base_models/grounded-sam-2/)
- [Autodistill GitHub](https://github.com/autodistill/autodistill)
- [Grounded-SAM-2 GitHub](https://github.com/autodistill/autodistill-grounded-sam-2)
- [Florence-2 GitHub](https://github.com/autodistill/autodistill-florence-2)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

## Support

For issues related to:
- **Pipeline script**: Check this repository's issues
- **Autodistill**: https://github.com/autodistill/autodistill/issues
- **YOLOv8**: https://github.com/ultralytics/ultralytics/issues
- **CUDA/GPU**: NVIDIA documentation and forums

## License

This pipeline implementation follows the licenses of its components:
- Autodistill: Apache 2.0
- Florence-2: MIT
- SAM-2: Apache 2.0
- YOLOv8: AGPL-3.0
