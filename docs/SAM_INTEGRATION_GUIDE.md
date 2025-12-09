# SAM Model Integration - Implementation Guide

## Overview

This implementation integrates the fine-tuned Segment Anything Model (SAM) into the HVAC AI Platform, providing powerful AI-driven tools for analyzing P&ID and HVAC diagrams.

**🆕 Recent Enhancements**: See [AI Inference Enhancements](./AI_INFERENCE_ENHANCEMENTS.md) for details on performance improvements, caching, and advanced features.

## Features

### 1. Interactive Segmentation (`/api/v1/segment`)
- **Purpose**: Click-to-segment tool for precise, pixel-perfect masks of individual components
- **Method**: POST with multipart/form-data
- **Inputs**:
  - `image`: Uploaded diagram file
  - `prompt`: JSON string with interaction details (e.g., point click coordinates)
  - `return_top_k` (optional): Number of top predictions to return (default: 1)
  - `enable_refinement` (optional): Enable prompt refinement (default: true)
- **Output**: Segmentation mask (RLE encoded), component label, confidence score, bounding box, and detailed confidence breakdown

### 2. Automated Component Counting (`/api/v1/count`)
- **Purpose**: One-click analysis to identify, classify, and count all recognized components
- **Method**: POST with multipart/form-data
- **Inputs**:
  - `image`: Uploaded diagram file
  - `grid_size` (optional): Grid spacing in pixels (default: 32)
  - `confidence_threshold` (optional): Minimum confidence score (default: 0.85)
  - `use_adaptive_grid` (optional): Auto-adjust grid size (default: true)
- **Output**: Total object count, breakdown by category, processing time, and confidence statistics

### 3. Performance Monitoring (`/api/v1/metrics`)
- **Purpose**: Track inference performance and cache utilization
- **Method**: GET
- **Output**: Metrics including cache hit rate, inference times, and cache size

### 4. Cache Management (`/api/v1/cache/clear`)
- **Purpose**: Clear the inference cache to free memory
- **Method**: POST
- **Output**: Status message

## Architecture

### Backend Service (Python/FastAPI)

**Location**: `python-services/`

Key components:
- `hvac_analysis_service.py`: Main FastAPI application with new SAM endpoints
- `core/ai/sam_inference.py`: SAM inference engine with model loading, segmentation, and counting logic
- `Dockerfile`: GPU-enabled container configuration
- `docker-compose.yml`: Deployment configuration with GPU support

**Technology Stack**:
- FastAPI for API endpoints
- PyTorch for model inference
- Segment Anything Model (SAM) for segmentation
- OpenCV and NumPy for image processing
- pycocotools for RLE mask encoding

### Frontend Application (Next.js/React)

**Location**: `src/`

Key components:
- `components/sam/SAMAnalysis.tsx`: Main SAM analysis component with interactive canvas
- `app/sam-analysis/page.tsx`: SAM analysis page

**Features**:
- Image upload with drag-and-drop
- Interactive canvas with click-to-segment
- Real-time mask visualization
- Automated counting with progress indicator
- Results table with filtering
- CSV export functionality

## Component Taxonomy

The model recognizes 70 HVAC/P&ID component types organized into categories:

### Valves & Actuators (21 types)
- Actuator types: Diaphragm, Generic, Manual, Motorized, Piston, Pneumatic, Solenoid
- Valve types: 3Way, 4Way, Angle, Ball, Butterfly, Check, Control, Diaphragm, Gate, Generic, Globe, Needle, Plug, Relief

### Equipment (11 types)
- AgitatorMixer, Compressor, FanBlower, Generic, HeatExchanger, Motor
- Pump types: Centrifugal, Dosing, Generic, Screw
- Vessel

### Instrumentation & Controls (14 types)
- Components: DiaphragmSeal, Switch
- Controllers: DCS, Generic, PLC
- Instruments: Analyzer, Flow-Indicator, Flow-Transmitter, Generic, Level-Indicator, Level-Switch, Level-Transmitter, Pressure-Indicator, Pressure-Switch, Pressure-Transmitter, Temperature

### Piping, Ductwork & In-line Components (24 types)
- Accessories: Drain, Generic, SightGlass, Vent
- Damper, Duct, Filter
- Fittings: Bend, Blind, Flange, Generic, Reducer
- Pipes: Insulated, Jacketed
- Strainers: Basket, Generic, YType
- Trap

## Setup Instructions

### Backend Setup

1. **Install Dependencies**:
   ```bash
   cd python-services
   pip install -r requirements.txt
   ```

2. **Place Model File**:
   - Place your fine-tuned SAM model at: `python-services/models/sam_hvac_finetuned.pth`
   - Or set `SAM_MODEL_PATH` environment variable

3. **Run Development Server**:
   ```bash
   python hvac_analysis_service.py
   ```
   Service will be available at `http://localhost:8000`

4. **Docker Deployment** (Recommended for Production):
   ```bash
   cd python-services
   docker-compose up -d
   ```

### Frontend Setup

1. **Install Dependencies**:
   ```bash
   bun install
   # or npm install
   ```

2. **Configure API URL**:
   Set `NEXT_PUBLIC_API_URL` environment variable or it defaults to `http://localhost:8000`

3. **Run Development Server**:
   ```bash
   bun dev
   # or npm run dev
   ```
   Access at `http://localhost:3000/sam-analysis`

## API Usage Examples

### Interactive Segmentation

```bash
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@diagram.png" \
  -F 'prompt={"type":"point","data":{"coords":[452,312],"label":1}}'
```

Response:
```json
{
  "status": "success",
  "segments": [
    {
      "label": "Valve-Ball",
      "score": 0.967,
      "mask": "eNq1k0tqAjEQhU...",
      "bbox": [430, 298, 55, 60]
    }
  ]
}
```

### Automated Counting

```bash
curl -X POST http://localhost:8000/api/v1/count \
  -F "image=@diagram.png"
```

Response:
```json
{
  "status": "success",
  "total_objects_found": 87,
  "counts_by_category": {
    "Valve-Ball": 23,
    "Valve-Gate": 12,
    "Fitting-Bend": 31,
    "Equipment-Pump-Centrifugal": 2,
    "Instrument-Pressure-Indicator": 19
  }
}
```

## Performance Considerations

### Hardware Requirements
- **GPU**: NVIDIA T4 or better (12+ GB VRAM recommended)
- **CPU**: 8+ cores for optimal performance
- **RAM**: 16+ GB system memory
- **Storage**: 10+ GB for model weights and dependencies

### Optimization Strategies
1. **Model Loading**: Model is loaded once at startup, not per-request
2. **Image Encoding**: For counting, image is encoded once and reused for all grid points
3. **NMS De-duplication**: High IoU threshold (0.9) efficiently removes duplicates
4. **Grid Size**: Adjustable grid spacing (default 32px) balances speed vs coverage
5. **Batch Processing**: Grid points processed efficiently with shared image embedding

### Expected Performance
- **Interactive Segmentation**: <1 second per click
- **Automated Counting**: 2-5 seconds for typical diagram (depending on size and complexity)
- **Memory Usage**: ~8-10 GB GPU RAM during inference

## Configuration Options

### Backend Configuration

Environment variables:
- `SAM_MODEL_PATH`: Path to model checkpoint (default: `models/sam_hvac_finetuned.pth`)
- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: `0`)

Inference parameters (in `sam_inference.py`):
- `grid_size`: Grid spacing for counting (default: 32 pixels)
- `confidence_threshold`: Minimum score to keep detection (default: 0.85)
- `nms_iou_threshold`: IoU threshold for NMS (default: 0.9)

### Frontend Configuration

Environment variables:
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: `http://localhost:8000`)

## Troubleshooting

### Model Not Loading
- Ensure model file exists at specified path
- Check GPU availability: `torch.cuda.is_available()`
- Verify CUDA version compatibility

### Slow Performance
- Enable GPU: Check `nvidia-smi` output
- Reduce grid size for counting
- Use smaller images or resize before upload

### Memory Issues
- Reduce batch size or grid density
- Use mixed precision training (FP16)
- Close other GPU processes

### API Connection Issues
- Verify backend is running: `curl http://localhost:8000/health`
- Check CORS configuration if accessing from different origin
- Ensure `NEXT_PUBLIC_API_URL` is set correctly

## Development Notes

### Mock Mode
Both backend and frontend support mock/demo modes for development without a trained model:
- Backend automatically falls back to mock mode if model file not found
- Mock mode generates realistic sample responses for testing UI

### Testing
Test endpoints with sample images:
```bash
# Health check
curl http://localhost:8000/health

# Test segmentation (mock mode)
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@test.png" \
  -F 'prompt={"type":"point","data":{"coords":[100,100],"label":1}}'

# Test counting (mock mode)
curl -X POST http://localhost:8000/api/v1/count -F "image=@test.png"
```

## Future Enhancements

Potential improvements for future versions:
- [ ] Real-time RLE mask visualization on canvas
- [ ] Multi-component selection and batch operations
- [ ] Advanced filtering and search in count results
- [ ] Export results to PDF reports
- [ ] Integration with existing HVAC analysis workflows
- [ ] Model versioning and A/B testing
- [ ] Confidence threshold adjustment in UI
- [ ] Component relationship detection

**✅ Recently Implemented**:
- [x] Intelligent caching system with LRU eviction
- [x] Advanced prompt engineering with multi-point sampling
- [x] Multi-stage classification pipeline (geometric + visual features)
- [x] Adaptive grid processing for optimized counting
- [x] Performance monitoring and metrics API
- [x] Enhanced API responses with confidence breakdowns
- [x] Model warm-up for optimized first inference

See [AI Inference Enhancements](./AI_INFERENCE_ENHANCEMENTS.md) for complete details.

## Documentation

- **Implementation Guide**: This document
- **Enhancement Details**: [AI_INFERENCE_ENHANCEMENTS.md](./AI_INFERENCE_ENHANCEMENTS.md)
- **Usage Examples**: [INFERENCE_USAGE_EXAMPLES.md](./INFERENCE_USAGE_EXAMPLES.md)
- **API Reference**: See FastAPI docs at `http://localhost:8000/docs`

## Support

For issues or questions:
- GitHub Issues: [github.com/elliotttmiller/hvac-ai/issues](https://github.com/elliotttmiller/hvac-ai/issues)
- Documentation: See project README and inline code documentation

## License

This implementation follows the project's license terms.
