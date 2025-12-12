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
- **SAHI Engine:** Slice-aided inference for large blueprints
- **Prompt Engineering:** Domain-specific prompting framework
- **Model Ensemble:** Multi-model prediction fusion

**Key Modules:**
- `hvac_sahi_engine.py` - SAHI-powered component detection
- `hvac_prompt_engineering.py` - Professional prompt templates

### 3. HVAC Document (`hvac-document/`)
Document processing and enhancement:
- Multi-format support (PDF, DWG, DXF, raster images)
- Quality assessment and adaptive enhancement
- Multi-page blueprint handling
- HVAC symbol extraction

**Key Modules:**
- `hvac_document_processor.py` - Document processing pipeline

### 4. HVAC Domain (`hvac-domain/`)
Business logic and domain rules:
- System relationship analysis
- ASHRAE/SMACNA compliance validation
- HVAC engineering rules engine
- Component connectivity graphs

**Key Modules:**
- `hvac_system_engine.py` - Relationship and validation engine

## Usage

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
from services.hvac_document.hvac_document_processor import (
    create_hvac_document_processor,
    BlueprintFormat
)

# Create processor
processor = create_hvac_document_processor(config={
    "target_dpi": 300,
    "enhance_ductwork_lines": True,
    "enhance_symbols": True
})

# Process blueprint
result = processor.process_document(
    file_path="hvac_plan.pdf",
    format_hint=BlueprintFormat.PDF
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
