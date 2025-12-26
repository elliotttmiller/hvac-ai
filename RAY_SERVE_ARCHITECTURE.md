# HVAC Cortex - Ray Serve Infrastructure

## Overview

The HVAC Cortex platform now features a distributed inference architecture built on Ray Serve. This architecture implements a Directed Acyclic Graph (DAG) of independent AI services that work together to provide multi-modal analysis of HVAC blueprints.

## Architecture

### The Inference Graph

```
┌─────────────────────────────────────────────────────────────┐
│                     Ingress (API Gateway)                    │
│                    - Receives HTTP POST                      │
│                    - Decodes images                          │
│                    - Orchestrates pipeline                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   ObjectDetector Node    │    │     Geometry Engine      │
│   - YOLOv11 detection    │───▶│  - Coordinate mapping     │
│   - OBB geometry         │    │  - Perspective transforms │
│   - 40% GPU allocation   │    │  - Un-rotate regions     │
└──────────────────────────┘    └──────────────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────┐
                              │  TextExtractor Node      │
                              │  - PaddleOCR engine      │
                              │  - Text recognition      │
                              │  - 30% GPU allocation    │
                              └──────────────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────┐
                              │     Fusion Layer         │
                              │  - Merges spatial + text │
                              │  - Unified JSON response │
                              └──────────────────────────┘
```

## Key Components

### 1. Universal Service Classes (Domain-Driven Design)

Following DDD principles, we use universal, tool-agnostic naming:

- **`ObjectDetector`** - Generic object detection (currently YOLOv11)
  - Easy to swap YOLO for EfficientDet, DETR, etc.
  - Location: `python-services/core/services/object_detector.py`

- **`TextExtractor`** - Generic text recognition (currently PaddleOCR)
  - Easy to swap PaddleOCR for EasyOCR, Tesseract, or GPT-4V
  - Location: `python-services/core/services/text_extractor.py`

- **`GeometryUtils`** - Geometric transformations
  - OBB corner calculation
  - Perspective transformation to rectify rotated crops
  - OCR preprocessing (grayscale, thresholding)
  - Location: `python-services/core/utils/geometry.py`

### 2. Ray Serve Deployments

Each service is wrapped as a Ray Serve deployment with:
- **Fractional GPU allocation** for efficient resource usage on GTX 1070 (8GB)
- **Async request handling** for high throughput
- **Independent scaling** per service

### 3. Intelligent Handshake

The pipeline implements selective inference:
- Only detections with `TEXT_RICH_CLASSES` trigger the TextExtractor
- TEXT_RICH_CLASSES = `{'id_letters', 'tag_number', 'text_label', 'label', 'text', 'tag'}`
- Geometric correction is applied before OCR for better accuracy

## Installation

### Prerequisites

```bash
# Python dependencies
cd python-services
pip install -r requirements.txt

# This includes:
# - ray[serve]>=2.9.0
# - paddlepaddle>=2.5.0
# - paddleocr>=2.7.0
# - ultralytics>=8.0.0
# - torch>=2.0.0
```

### Environment Setup

Create a `.env.local` file in the repository root:

```bash
# Model configuration
YOLO_MODEL_PATH=/path/to/your/best.pt
CONF_THRESHOLD=0.5

# Optional: Skip model loading for dev mode
SKIP_MODEL=0

# Optional: Force CPU mode
FORCE_CPU=0
```

## Usage

### Option 1: Ray Serve Mode (Recommended)

```bash
# Start the distributed inference platform
python scripts/start_unified.py --mode ray-serve

# Or just the AI engine (no frontend)
python scripts/start_unified.py --mode ray-serve --no-frontend

# Or directly
python scripts/start_ray_serve.py
```

### Option 2: Legacy Mode (FastAPI)

```bash
# Start with the original FastAPI backend
python scripts/start_unified.py --mode legacy

# Or use the original script
python scripts/start.py
```

## API Usage

### Request Format

```bash
# Using base64-encoded image
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "...",
    "conf_threshold": 0.5
  }'
```

### Response Format

```json
{
  "status": "success",
  "total_detections": 15,
  "image_shape": [4096, 3072],
  "detections": [
    {
      "label": "tag_number",
      "score": 0.95,
      "class_id": 3,
      "obb": {
        "x_center": 1024.5,
        "y_center": 768.3,
        "width": 150.0,
        "height": 50.0,
        "rotation": 0.523
      },
      "textContent": "AHU-1",
      "textConfidence": 0.98
    }
  ]
}
```

Note the **`textContent`** and **`textConfidence`** fields - these are populated by the TextExtractor for text-rich detections.

## Frontend Integration

The frontend automatically displays extracted text:

- **High-contrast green background** for extracted text
- **Monospace font** to signify "Read Data"
- **Text displayed preferentially** over class labels
- Format: `"AHU-1 (98%)"` instead of `"tag_number 95%"`

See: `src/components/viewer/DeepZoomViewer.tsx`

## Performance Characteristics

### GPU Memory Allocation

On GTX 1070 (8GB VRAM):
- ObjectDetector: ~3.2 GB (40%)
- TextExtractor: ~2.4 GB (30%)
- Remaining: ~2.4 GB buffer

### Throughput

- **Concurrent requests**: 10 per service (configurable)
- **Async processing**: Non-blocking I/O
- **Batch text extraction**: Multiple crops processed together

## Development

### Adding a New Service

1. Create a universal service class in `python-services/core/services/`
2. Follow the naming convention (e.g., `LayoutAnalyzer`, not `DocLayoutNetWrapper`)
3. Wrap it as a Ray Serve deployment in `core/inference_graph.py`
4. Update the ingress to call your service

### Testing Individual Services

```python
from core.services.object_detector import ObjectDetector
from core.services.text_extractor import TextExtractor

# Test object detection
detector = ObjectDetector(model_path="ai_model/best.pt")
detections = detector.detect(image)

# Test text extraction
extractor = TextExtractor(lang='en')
text_result = extractor.extract_single_text(crop_image)
```

## Troubleshooting

### Ray Serve not starting

```bash
# Check Ray installation
pip install ray[serve]

# Check model path
export YOLO_MODEL_PATH=/path/to/best.pt
ls -lh $YOLO_MODEL_PATH
```

### GPU memory errors

```bash
# Reduce GPU allocations in core/inference_graph.py
# ObjectDetectorDeployment: num_gpus=0.3 (instead of 0.4)
# TextExtractorDeployment: num_gpus=0.2 (instead of 0.3)
```

### PaddleOCR not loading

```bash
# Install PaddlePaddle and PaddleOCR
pip install paddlepaddle paddleocr

# For GPU support
pip install paddlepaddle-gpu
```

## Global Coding Standards

Per the PR specification, all code follows these standards:

### Universal Terminology (DDD)

✅ **Use:** `ObjectDetector`, `TextExtractor`, `BlueprintViewer`, `GeometryUtils`

❌ **Avoid:** `YoloService`, `PaddleOCRWrapper`, `DeepZoomInferenceAnalysis`

**Why?** If we switch from YOLO to EfficientDet, or Paddle to GPT-4V, we shouldn't have to rename our entire codebase.

## Next Steps

1. **Cloud Deployment**: Scale to Ray cluster on AWS/GCP
2. **Additional Services**: Layout analysis, component classification
3. **Performance Optimization**: Model quantization, TensorRT
4. **Monitoring**: Ray dashboard, metrics collection

## References

- Ray Serve Documentation: https://docs.ray.io/en/latest/serve/
- Domain-Driven Design: https://martinfowler.com/bliki/DomainDrivenDesign.html
- Project Task Document: `pr-task.md`
