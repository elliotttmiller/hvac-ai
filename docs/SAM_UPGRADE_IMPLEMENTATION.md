# SAM Inference Pipeline Upgrade - Implementation Guide

## Overview

This document describes the upgrade of the HVAC AI Platform to use a fine-tuned, production-ready SAM (Segment Anything Model) for HVAC and P&ID component analysis. The implementation provides interactive segmentation and automated component counting with RLE-encoded masks for efficient data transfer.

## Architecture

### Backend (Python/FastAPI)

The backend service provides two main API endpoints:

#### 1. Interactive Segmentation Endpoint
**Endpoint:** `POST /api/v1/segment`

Segments individual components based on user interaction (point clicks).

**Request:**
- `image`: Image file (multipart/form-data)
- `prompt`: JSON string with prompt details
  ```json
  {
    "type": "point",
    "data": {
      "coords": [x, y],
      "label": 1
    }
  }
  ```
- `return_top_k`: Number of top predictions (optional, default: 1)
- `enable_refinement`: Enable prompt refinement (optional, default: true)

**Response:**
```json
{
  "status": "success",
  "segments": [
    {
      "label": "Valve-Ball",
      "score": 0.967,
      "mask": "base64_encoded_rle_string",
      "bbox": [x, y, width, height],
      "confidence_breakdown": {
        "geometric": 0.92,
        "visual": 0.88,
        "combined": 0.90
      },
      "alternative_labels": [
        ["Valve-Gate", 0.85],
        ["Valve-Control", 0.78]
      ]
    }
  ],
  "processing_time_ms": 234.5
}
```

#### 2. Automated Counting Endpoint
**Endpoint:** `POST /api/v1/count`

Analyzes entire diagram to identify, classify, and count all components.

**Request:**
- `image`: Image file (multipart/form-data)
- `grid_size`: Grid spacing in pixels (optional, default: 32)
- `confidence_threshold`: Minimum confidence (optional, default: 0.85)
- `use_adaptive_grid`: Auto-adjust grid size (optional, default: true)

**Response:**
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
  },
  "processing_time_ms": 2340.5,
  "confidence_stats": {
    "mean": 0.87,
    "std": 0.12,
    "min": 0.65,
    "max": 0.98,
    "above_threshold": 112,
    "after_nms": 87
  }
}
```

### Frontend (React/TypeScript/Next.js)

The frontend provides an interactive UI for diagram analysis:

**Features:**
- Drag-and-drop image upload
- Interactive click-to-segment mode
- Multi-segment visualization with color coding
- Automated component counting
- Results table with CSV export
- Real-time RLE mask decoding and rendering

**Key Components:**
- `src/components/sam/SAMAnalysis.tsx` - Main UI component
- `src/lib/rle-decoder.ts` - RLE mask decoder utility

## RLE Mask Format

The backend encodes masks using COCO RLE (Run-Length Encoding) format for efficient data transfer:

**Encoding (Backend):**
```python
# 1. Convert binary mask to COCO RLE
rle = mask_utils.encode(np.asfortranarray(mask))
# 2. Create string format: "HxW:counts"
rle_string = f"{rle['size'][0]}x{rle['size'][1]}:{rle['counts']}"
# 3. Base64 encode for JSON transfer
mask_encoded = base64.b64encode(rle_string.encode()).decode('utf-8')
```

**Decoding (Frontend):**
```typescript
// 1. Decode base64
const rleString = atob(encodedMask);
// 2. Parse format: "HxW:counts"
const [sizeStr, countsStr] = rleString.split(':');
const [height, width] = sizeStr.split('x').map(Number);
// 3. Decode RLE counts (variable-length integers)
const counts = decodeRLECounts(countsStr);
// 4. Reconstruct binary mask from runs
const mask = rleToBinaryMask(counts, height, width);
```

**Benefits:**
- Significantly smaller payload than PNG encoding
- Lossless compression
- Industry-standard format (COCO dataset)
- Fast encoding/decoding

## Model Configuration

### Environment Variables

```bash
# Backend Configuration
SAM_MODEL_PATH=./models/sam_hvac_finetuned.pth  # Path to fine-tuned model
CUDA_VISIBLE_DEVICES=0  # GPU device (if available)

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000  # Backend API URL
```

### Model Loading

The backend automatically loads the model on startup:

1. Checks `SAM_MODEL_PATH` environment variable
2. Falls back to default: `./models/sam_hvac_finetuned.pth`
3. Loads model weights into GPU memory
4. Sets model to evaluation mode
5. Warms up with dummy forward pass

**Mock Mode:**
If the model file is not found or dependencies are missing, the service runs in mock mode for development/testing.

## Inference Pipeline

### Direct Encoder/Decoder Usage

The implementation uses direct model component access instead of `SamPredictor`:

```python
# 1. Encode image once (cached for reuse)
image_embedding = self.image_encoder(input_tensor)

# 2. For each prompt:
# Encode prompt
sparse_embeddings, dense_embeddings = self.prompt_encoder(
    points=(point_coords, point_labels),
    boxes=None,
    masks=None
)

# 3. Decode mask
low_res_masks, iou_predictions = self.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=self.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False  # Single best mask
)
```

**Key Features:**
- Image embedding is computed once and cached
- Multiple prompts reuse the same embedding
- LRU cache for repeated image analysis
- Configurable cache size (default: 50 embeddings)

### Component Classification

Multi-stage classification pipeline:

1. **Geometric Classification** (60% weight)
   - Shape analysis (circularity, aspect ratio)
   - Size-based heuristics
   - Vertex counting

2. **Visual Classification** (40% weight)
   - Color intensity features
   - Texture analysis (placeholder for learned features)

3. **Combined Scoring**
   - Weighted average of geometric and visual scores
   - Returns top prediction with confidence breakdown
   - Includes alternative predictions (top 3)

### Automated Counting with NMS

Grid-based detection with de-duplication:

1. **Grid Generation**
   - Adaptive grid sizing based on image dimensions
   - Dense sampling (default: 32px spacing)

2. **Detection**
   - Process each grid point as a prompt
   - Filter by confidence threshold (default: 0.85)

3. **Non-Maximum Suppression**
   - Sort detections by confidence
   - Remove duplicates based on IoU threshold (default: 0.9)
   - Keep only highest-confidence detection per object

4. **Category Counting**
   - Tally unique objects by classification label
   - Return statistics and per-category counts

## HVAC Component Taxonomy

The model recognizes 70 component types across 4 categories:

### Valves & Actuators (21 types)
- Actuator variants: Diaphragm, Manual, Motorized, Pneumatic, Solenoid, etc.
- Valve types: Ball, Gate, Butterfly, Check, Control, Globe, Relief, etc.

### Equipment (11 types)
- Pumps: Centrifugal, Dosing, Screw
- Motors, Compressors, Fans/Blowers
- Heat Exchangers, Vessels, Mixers

### Instrumentation & Controls (14 types)
- Flow: Indicators, Transmitters
- Level: Indicators, Switches, Transmitters
- Pressure: Indicators, Switches, Transmitters
- Temperature sensors
- Controllers: DCS, PLC, Generic
- Analyzers

### Piping/Ductwork/In-line Components (24 types)
- Piping: Standard, Insulated, Jacketed
- Fittings: Bends, Flanges, Reducers
- Filters, Strainers (Basket, Y-Type)
- Dampers, Ducts
- Accessories: Drains, Vents, Sight Glass
- Traps

## Performance Optimization

### Caching Strategy
- **Embedding Cache**: LRU cache for image embeddings
- **Cache Size**: Configurable (default: 50)
- **Cache Hits**: Significantly reduces computation for repeated operations
- **Metrics**: Track cache hit rate and performance

### Adaptive Grid Processing
- **Small Images** (<1000x1000): 24px grid spacing
- **Large Images** (>2000x2000): 48px grid spacing
- **Custom**: User-specified grid size

### Batch Operations
- Grid prompts processed efficiently
- Reuses single image embedding
- Minimal memory overhead

## Installation & Setup

### Backend Dependencies

```bash
cd python-services

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Key dependencies:
# - fastapi>=0.115.6
# - uvicorn[standard]>=0.34.0
# - torch>=2.0.0
# - opencv-python>=4.8.0
# - pycocotools>=2.0.0
# - segment-anything (from GitHub)
```

### Frontend Dependencies

```bash
# Install Node.js dependencies
npm install --legacy-peer-deps

# Key dependencies:
# - react
# - next.js
# - react-dropzone
# - shadcn/ui components
```

### Model Placement

```bash
# Create models directory
mkdir -p python-services/models

# Place fine-tuned model
cp /path/to/your/best_model_expert_v1.pth python-services/models/sam_hvac_finetuned.pth

# Or set custom path
export SAM_MODEL_PATH=/custom/path/to/model.pth
```

## Running the Services

### Backend

```bash
cd python-services
source venv/bin/activate
python -m uvicorn hvac_analysis_service:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
npm run dev
# Opens at http://localhost:3000
```

### Production Build

```bash
# Frontend
npm run build
npm start

# Backend
uvicorn hvac_analysis_service:app --host 0.0.0.0 --port 8000 --workers 4
```

## Testing

### Backend API Testing

```bash
# Health check
curl http://localhost:8000/health

# Test segmentation
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@test_diagram.png" \
  -F 'prompt={"type": "point", "data": {"coords": [150, 200], "label": 1}}'

# Test counting
curl -X POST http://localhost:8000/api/v1/count \
  -F "image=@test_diagram.png"
```

### Frontend Testing

1. Open http://localhost:3000/sam-analysis
2. Upload a P&ID or HVAC diagram
3. Enable click-to-segment mode
4. Click on components to segment them
5. Use "Analyze and Count All Components" for full analysis

## Troubleshooting

### Backend Issues

**Model Not Loading:**
```
2025-12-09 11:15:18,687 - WARNING - SAM model file not found. Using mock mode.
```
- Check `SAM_MODEL_PATH` is set correctly
- Verify model file exists at the path
- Ensure file has .pth extension

**GPU Not Available:**
```
INFO - Initializing Enhanced SAM Inference Engine on cpu
```
- Install CUDA-compatible PyTorch: `pip install torch --extra-index-url https://download.pytorch.org/whl/cu118`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

**Dependencies Missing:**
```
ModuleNotFoundError: No module named 'torch'
```
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Frontend Issues

**API Connection Failed:**
```
Error: API URL not configured
```
- Set `NEXT_PUBLIC_API_URL` in `.env.local`
- Ensure backend is running on the configured URL

**Build Errors:**
```
npm error peer dependency conflict
```
- Use `npm install --legacy-peer-deps`
- Check Node.js version (requires Node 18+)

**RLE Decoding Errors:**
- Check browser console for detailed error messages
- Verify mask format matches expected structure
- Ensure backend is returning valid RLE-encoded masks

## Security Considerations

1. **Input Validation**
   - File type validation (PNG, JPG, JPEG only)
   - File size limits
   - Prompt parameter validation

2. **CORS Configuration**
   - Production: Restrict to specific origins
   - Development: Allow all origins

3. **Rate Limiting**
   - Consider adding rate limiting for production
   - Protect against denial of service

4. **Model Security**
   - Keep model weights in secure location
   - Use environment variables for paths
   - Don't commit model files to git

## Future Enhancements

1. **Model Improvements**
   - Fine-tune on larger HVAC dataset
   - Add more component types
   - Improve classification accuracy

2. **API Enhancements**
   - Batch processing endpoint
   - Async job queue for large images
   - WebSocket for real-time updates

3. **UI Improvements**
   - 3D visualization
   - Component relationship graph
   - Export to CAD formats

4. **Performance**
   - Model quantization for faster inference
   - Distributed processing for scale
   - Edge deployment options

## References

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [COCO RLE Format](https://github.com/cocodataset/cocoapi)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)

## Support

For issues or questions, please refer to:
- Project repository: `elliotttmiller/hvac-ai`
- Documentation: `/docs` directory
- API docs: http://localhost:8000/docs (when backend running)
