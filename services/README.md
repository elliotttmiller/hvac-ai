# HVAC-AI Services

This directory contains modular HVAC-specialized services that power the HVAC-AI platform.

## Service Architecture

The platform is organized into four primary service domains:

### 1. Gateway (`gateway/`)
API Gateway for routing and orchestration:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- API versioning

### 2. HVAC AI (`hvac-ai/`)
Core AI services for HVAC analysis:
- **YOLO Inference:** YOLOv11-obb for component detection
- **SAHI Engine:** Slice-aided inference for large blueprints
- **Prompt Engineering:** Domain-specific prompting framework
- **Integrated Detector:** Complete YOLOplan integration
- **VLM (Vision Language Models):** Advanced vision-language capabilities
- **Pipeline:** End-to-end drawing analysis pipeline

**Key Modules:**
- `hvac_sahi_engine.py` - SAHI-powered component detection
- `hvac_prompt_engineering.py` - Professional prompt templates
- `yolo_inference.py` - YOLO model inference engine
- `hvac_pipeline.py` - End-to-end HVAC drawing analyzer
- `pipeline_models.py` - Data models for pipeline
- `integrated_detector.py` - Integrated detection system
- `yoloplan_detector.py` - YOLOplan MEP symbol detection
- `yoloplan_bom.py` - BOM generation and connectivity
- `vlm/` - Vision Language Model components

### 3. HVAC Document (`hvac-document/`)
Document processing and enhancement:
- Multi-format support (PDF, DWG, DXF, raster images)
- Quality assessment and adaptive enhancement
- Multi-page blueprint handling
- HVAC symbol extraction
- Enhanced document processing with OCR
- Hybrid processing capabilities
- Table extraction

**Key Modules:**
- `hvac_document_processor.py` - Document processing pipeline
- `hvac_symbol_library.py` - Symbol library and recognition
- `enhanced_document_processor.py` - Enhanced processing with caching
- `hybrid_document_processor.py` - Hybrid OCR processing
- `table_extractor.py` - Table detection and extraction
- `document_processor.py` - Base document processor

### 4. HVAC Domain (`hvac-domain/`)
Business logic and domain rules:
- System relationship analysis
- ASHRAE/SMACNA compliance validation
- HVAC engineering rules engine
- Component connectivity graphs
- Cost estimation
- Pricing services
- Location intelligence

**Key Modules:**
- `hvac_system_engine.py` - Relationship and validation engine
- `hvac_compliance_analyzer.py` - Compliance checking
- `relationship_graph.py` - System relationship analysis
- `compliance/` - Compliance standards (ASHRAE, IMC, SMACNA)
- `system_analysis/` - System analysis tools
- `estimation/` - Cost estimation
- `pricing/` - Pricing services
- `location/` - Location intelligence

## Unified Service

The `hvac_unified_service.py` provides a consolidated FastAPI application that integrates all services into a single deployment unit.

```bash
# Start the unified service
python services/hvac_unified_service.py
```

## Import Compatibility

The repository uses shim packages (`hvac_ai`, `hvac_document`, `hvac_domain`) to enable Python imports from the hyphenated directories (`hvac-ai`, `hvac-document`, `hvac-domain`). You can import using either style:

```python
# Using shim packages (recommended for compatibility)
from services.hvac_ai.hvac_pipeline import create_hvac_analyzer
from services.hvac_document.enhanced_document_processor import create_enhanced_processor
from services.hvac_domain.hvac_system_engine import HVACSystemEngine

# Direct imports also work when the package is properly configured
from services.hvac_ai import create_hvac_analyzer
```

## Usage Examples

### HVAC Pipeline (End-to-End Analysis)

```python
from services.hvac_ai.hvac_pipeline import create_hvac_analyzer
from services.hvac_ai.pipeline_models import PipelineConfig

# Create analyzer with configuration
config = PipelineConfig(
    confidence_threshold=0.7,
    max_processing_time_ms=25.0,
    enable_gpu=True
)

analyzer = create_hvac_analyzer(
    model_path="./models/yolo11m-obb-hvac.pt",
    config=config
)

# Analyze a drawing
result = analyzer.analyze_drawing("drawing.png")

if result.success:
    print(f"Detections: {len(result.detection_result.detections)}")
    print(f"Text regions: {len(result.text_results)}")
    print(f"Interpretations: {len(result.interpretation_result.interpretations)}")
    print(f"Total time: {result.total_processing_time_ms:.2f}ms")
```

### HVAC SAHI Engine

```python
from services.hvac_ai.hvac_sahi_engine import create_hvac_sahi_predictor

# Initialize predictor
predictor = create_hvac_sahi_predictor(
    model_path="models/sam_hvac_finetuned.pth",
    device="cuda",
    slice_height=1024,
    overlap_height_ratio=0.3
)

# Analyze blueprint with adaptive slicing
result = predictor.predict_hvac_components(
    image_path="blueprint.png",
    adaptive_slicing=True
)

print(f"Detected {len(result['detections'])} components")
```

### HVAC System Engine

```python
from services.hvac_domain.hvac_system_engine import (
    HVACSystemEngine,
    HVACComponent,
    HVACComponentType
)

# Create engine
engine = HVACSystemEngine()

# Add components
for detection in detections:
    component = HVACComponent(
        id=detection['id'],
        component_type=HVACComponentType.DUCTWORK,
        bbox=detection['bbox'],
        confidence=detection['score']
    )
    engine.add_component(component)

# Build relationship graph
graph = engine.build_relationship_graph()

# Validate system configuration
validation = engine.validate_system_configuration()

if not validation['is_valid']:
    print(f"Found {len(validation['violations'])} violations")
    for violation in validation['violations']:
        print(f"  - {violation['message']}")
```

### Prompt Engineering

```python
from services.hvac_ai.hvac_prompt_engineering import create_hvac_prompt_framework

# Initialize framework
framework = create_hvac_prompt_framework()

# Generate HVAC-specific prompt
prompt = framework.generate_prompt(
    template_name="component_detection_cot",
    variables={
        "context": "Commercial HVAC system",
        "blueprint_type": "Supply air distribution plan"
    }
)

# Use with AI model
ai_response = your_ai_model.analyze(prompt, image)
```

### Document Processing

```python
from services.hvac_document.enhanced_document_processor import create_enhanced_processor

# Create processor
processor = create_enhanced_processor(use_cache=True)

# Process blueprint
result = processor.process_document(
    file_path="hvac_plan.pdf",
    enhance_quality=True
)

# Access processed pages
for page in result['pages']:
    image = page['image']
    quality = page['quality_metrics']
    
    print(f"Page {page['page_number']}")
    print(f"  Quality: {quality.overall_quality:.2f}")
    print(f"  Issues: {', '.join(quality.issues) if quality.issues else 'None'}")
```

## Dependencies

### Core Dependencies
- **torch** >= 2.0.0 - PyTorch for deep learning
- **sahi** >= 0.11.0 - Slicing Aided Hyper Inference
- **opencv-python** >= 4.8.0 - Image processing
- **numpy** >= 1.24.0 - Numerical operations

### Optional Dependencies
- **ezdxf** - DWG/DXF file processing
- **pymupdf** - PDF processing with layer preservation
- **redis** - Caching for production deployments

## Configuration

### SAHI Configuration

```python
from services.hvac_ai.hvac_sahi_engine import HVACSAHIConfig

config = HVACSAHIConfig(
    slice_height=1024,              # Optimized for ductwork patterns
    slice_width=1024,
    overlap_height_ratio=0.3,       # 30% overlap for continuity
    overlap_width_ratio=0.3,
    confidence_threshold=0.40,      # Higher for critical components
    iou_threshold=0.50              # For component fusion
)
```

### Document Processing Configuration

```python
config = {
    "target_dpi": 300,
    "min_acceptable_dpi": 150,
    "enhance_ductwork_lines": True,
    "enhance_symbols": True,
    "preserve_layers": True,
    "multi_page_support": True,
    "quality_threshold": 0.6
}
```

## Testing

Run tests for specific services:

```bash
# Test HVAC AI services
pytest hvac-tests/unit/test_sahi_engine.py
pytest hvac-tests/unit/test_prompt_engineering.py

# Test HVAC Domain services
pytest hvac-tests/unit/test_system_engine.py

# Test Document Processing
pytest hvac-tests/unit/test_document_processor.py

# Integration tests
pytest hvac-tests/integration/
```

## Performance Considerations

### Memory Management
- SAHI slicing keeps GPU memory under 8GB
- Adaptive slicing adjusts based on blueprint complexity
- Automatic cleanup after processing

### Processing Speed
- Parallel slice processing where possible
- Efficient result fusion algorithms
- Caching for repeated analyses

### Accuracy Optimization
- Component-specific confidence thresholds
- Multi-scale detection for small components
- Relationship validation to reduce false positives

## Future Enhancements

Planned improvements:
- [ ] Multi-model ensemble with YOLOv8
- [ ] Real-time streaming analysis
- [ ] Advanced caching strategies
- [ ] Auto-scaling based on load
- [ ] Extended symbol library
- [ ] Machine learning for prompt optimization

## Contributing

When adding new services:
1. Follow the existing module structure
2. Include comprehensive docstrings
3. Add type hints for all functions
4. Write unit tests (target 85%+ coverage)
5. Update this README

## Support

For service-specific questions:
- SAHI Engine: See `hvac_sahi_engine.py` docstrings
- System Engine: See `hvac_system_engine.py` docstrings
- Prompts: See `hvac_prompt_engineering.py` docstrings
- Documents: See `hvac_document_processor.py` docstrings

---

**Status:** Active Development  
**Version:** 1.0.0
