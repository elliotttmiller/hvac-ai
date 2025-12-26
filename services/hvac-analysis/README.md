# HVAC AI Platform - Python Services

This directory contains the backend AI services for the HVAC AI Platform.

## Structure

```
services/hvac-analysis/
├── core/                          # Core business logic
│   ├── ai/                       # AI models and inference
│   │   ├── yolo_inference.py    # YOLO/Ultralytics model inference engine
│   │   └── detector.py          # HVAC component detection
│   ├── document/                 # Document processing
│   │   └── processor.py         # Blueprint/CAD processing
│   ├── estimation/               # Cost estimation
│   │   └── calculator.py        # Cost calculation engine
│   └── location/                 # Location intelligence
│       └── intelligence.py      # Building codes & compliance
├── hvac_analysis_service.py      # Main FastAPI application
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image configuration
├── docker-compose.yml            # Docker Compose setup
└── start.sh                      # Service startup script
```

## Setup

### Local Development

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the service**
   ```bash
   python hvac_analysis_service.py
   ```

   The API will be available at:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Check logs**
   ```bash
   docker-compose logs -f
   ```

3. **Stop services**
   ```bash
   docker-compose down
   ```

## API Endpoints

-### Core Endpoints

- `GET /` - Service information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

### AI Analysis

- `POST /api/analyze` - Interactive inference segmentation (polygon masks, YOLO/Ultralytics)
- `POST /api/count` - Automated component counting
- `POST /api/v1/segment` - Legacy interactive segmentation (backward compatible)
- `POST /api/v1/count` - Legacy counting endpoint
- `GET /api/v1/metrics` - Performance metrics
- `POST /api/v1/cache/clear` - Clear inference cache

### Document Processing

- `POST /api/analyze` - Analyze HVAC blueprint
- `POST /api/estimate` - Generate cost estimate

### Configuration

### Environment Variables

- `MODEL_PATH` - Path to inference model file (YOLO/Ultralytics) (required)
- `NGROK_AUTHTOKEN` - ngrok auth token for secure tunneling (required for development)
- `SAM_MODEL_PATH` - Backward-compatible alias for `MODEL_PATH` (legacy)
- `CUDA_VISIBLE_DEVICES` - GPU device ID (default: `0`)
- `PORT` - Service port (default: `8000`)
- `HOST` - Service host (default: `0.0.0.0`)

### Model Setup

Place your trained inference model at (YOLO/Ultralytics recommended):
```bash
services/hvac-analysis/models/<your_model_file>.pt
```

If no model is found, the service runs in mock mode for development.

## Core Modules

### AI Module (`core/ai/`)

**YOLO Inference Engine** (`yolo_inference.py`)
- YOLO/Ultralytics model for HVAC/P&ID diagrams
- Features: adaptive slicing, polygon/segmentation support (model-dependent), multi-stage classification
- Recognizes 70+ HVAC component types

**Component Detector** (`detector.py`)
- YOLO-based object detection
- Spatial relationship analysis
- Component classification

### Document Module (`core/document/`)

**Document Processor** (`processor.py`)
- Multi-format support (PDF, DWG, DXF, PNG, JPG)
- Text extraction with OCR
- Blueprint classification

### Estimation Module (`core/estimation/`)

**Cost Calculator** (`calculator.py`)
- Material cost estimation
- Labor calculation
- Regional pricing adjustments

### Location Module (`core/location/`)

**Location Intelligence** (`intelligence.py`)
- Building code compliance
- Climate zone analysis
- Regional requirements

## Development

### Adding New Endpoints

1. Add route to `hvac_analysis_service.py`
2. Implement business logic in appropriate `core/` module
3. Update documentation
4. Test with FastAPI's interactive docs

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest
```

## Performance

- Image embedding caching (configurable size)
- GPU acceleration with CUDA
- Adaptive grid processing
- Efficient NMS de-duplication

## Troubleshooting

### Common Issues

1. **Model not found**
   - Place model at `models/sam_hvac_finetuned.pth`
   - Service runs in mock mode without model

2. **CUDA errors**
   - Check GPU availability
   - Verify CUDA version compatibility
   - Set `CUDA_VISIBLE_DEVICES` appropriately

3. **Memory issues**
   - Reduce cache size in `sam_inference.py`
   - Lower grid resolution for counting
   - Use smaller batch sizes

## Documentation

- See the main [documentation directory](../docs/) for:
- [Inference Integration Guide](../docs/SAM_INTEGRATION_GUIDE.md)
- [API Usage Examples](../docs/INFERENCE_USAGE_EXAMPLES.md)
- [Deployment Guide](../docs/SAM_DEPLOYMENT.md)
