# ADR 004: Advanced AI Document Processing for HVAC Blueprints

## Status
Proposed

## Context

Based on comprehensive research of state-of-the-art document processing techniques, including:
- ArXiv 2411.03707: Semantic analysis and caching for document processing
- Engineering drawing processing with AI-powered OCR
- Complex document recognition beyond traditional OCR
- Commercial proposal parsing with Vision-Language Models

Traditional OCR approaches fail on complex HVAC blueprints due to:
- Mixed text/graphics with varying orientations
- Technical symbols and annotations
- Multi-column layouts and tables
- Poor scan quality and handwritten notes
- Context-dependent information (relationships between components)

## Research Findings

### Key Methodologies Identified

#### 1. Hybrid OCR + Vision-Language Model Pipeline
**Source**: Complex Document Recognition research (HackerNoon)

**Approach**:
- Traditional OCR for raw text extraction
- VLM (Vision-Language Model) for contextual understanding
- Semantic post-processing to verify and structure results

**Benefits**:
- 70-90% improvement in accuracy over OCR-only
- Handles tables, diagrams, and mixed content
- Reduces hallucinations through multi-model verification

**Implementation for HVAC**:
```python
class HybridDocumentProcessor:
    def __init__(self):
        self.ocr_engine = AdvancedOCR()  # EasyOCR or PaddleOCR
        self.vlm_model = VisionLanguageModel()  # Qwen2-VL or similar
        self.semantic_validator = SemanticValidator()
    
    def process_blueprint(self, image):
        # Stage 1: Raw OCR extraction
        ocr_results = self.ocr_engine.extract(image)
        
        # Stage 2: VLM contextual understanding
        vlm_results = self.vlm_model.analyze(image, ocr_results)
        
        # Stage 3: Semantic validation and fusion
        validated = self.semantic_validator.merge(ocr_results, vlm_results)
        
        return validated
```

#### 2. Layout-Aware Segmentation
**Source**: Engineering Drawing Processing (HackerNoon)

**Approach**:
- Detect document regions (title block, drawings, notes, tables)
- Process each region with specialized algorithms
- Use bounding boxes to isolate features before OCR

**Benefits**:
- Handles rotated text and annotations
- Improves accuracy by 40-60% on engineering drawings
- Enables parallel processing of regions

**Implementation for HVAC**:
```python
class LayoutSegmenter:
    def segment_blueprint(self, image):
        # Detect regions using object detection
        regions = {
            'title_block': self.detect_title_block(image),
            'main_drawing': self.detect_main_area(image),
            'schedules': self.detect_schedules(image),
            'notes': self.detect_notes(image),
            'legends': self.detect_legends(image)
        }
        
        # Process each region with specialized pipeline
        results = {}
        for region_name, region_bbox in regions.items():
            region_image = self.crop_region(image, region_bbox)
            results[region_name] = self.process_region(
                region_image, 
                region_type=region_name
            )
        
        return results
```

#### 3. Rotation-Invariant Text Detection
**Source**: Engineering Drawing Processing, AI-Based OCR for Engineering Drawings

**Approach**:
- Detect text orientation before extraction
- Apply rotation correction per text block
- Use oriented bounding boxes (OBB) instead of axis-aligned

**Benefits**:
- Handles text at any angle (0-360°)
- Essential for engineering drawings with rotated annotations
- Reduces misreads by 50-80%

**Implementation for HVAC**:
```python
class RotationInvariantOCR:
    def extract_text(self, image):
        # Detect text regions with orientation
        text_regions = self.detect_oriented_text(image)
        
        results = []
        for region in text_regions:
            # Rotate region to horizontal
            normalized = self.rotate_to_horizontal(
                image, 
                region.bbox, 
                region.angle
            )
            
            # Extract text from normalized region
            text = self.ocr_engine.extract(normalized)
            
            results.append({
                'text': text,
                'bbox': region.bbox,
                'angle': region.angle,
                'confidence': region.confidence
            })
        
        return results
```

#### 4. Semantic Caching
**Source**: ArXiv 2411.03707 (LLMCache)

**Approach**:
- Cache intermediate processing results based on semantic similarity
- Reuse computations for repeated document patterns
- Layer-wise caching for VLM inference

**Benefits**:
- 60-80% reduction in processing time for similar documents
- Reduced GPU memory usage
- Enables real-time processing of large blueprint sets

**Implementation for HVAC**:
```python
class SemanticCache:
    def __init__(self):
        self.cache = {}
        self.embedder = SentenceTransformer()
        self.similarity_threshold = 0.85
    
    def get_or_process(self, image, processor_fn):
        # Generate semantic embedding
        embedding = self.embedder.encode(self.extract_features(image))
        
        # Check cache for similar results
        for cached_key, cached_result in self.cache.items():
            similarity = cosine_similarity(embedding, cached_key)
            if similarity > self.similarity_threshold:
                return cached_result
        
        # Process and cache
        result = processor_fn(image)
        self.cache[embedding] = result
        
        return result
```

#### 5. Structured Extraction with Context
**Source**: Commercial Proposals Parsing, Document Parser for RAG

**Approach**:
- Extract not just text, but relationships and hierarchies
- Use domain knowledge (ASHRAE standards) to validate
- Output structured JSON instead of plain text

**Benefits**:
- Enables automatic compliance checking
- Supports relationship graph construction
- 90%+ accuracy on structured data extraction

**Implementation for HVAC**:
```python
class StructuredExtractor:
    def extract_hvac_data(self, blueprint):
        # Extract with context
        raw_data = self.hybrid_processor.process(blueprint)
        
        # Structure according to HVAC domain model
        structured = {
            'system_info': self.extract_system_info(raw_data),
            'components': self.extract_components(raw_data),
            'connections': self.extract_connections(raw_data),
            'specifications': self.extract_specs(raw_data),
            'compliance': self.check_compliance(raw_data)
        }
        
        # Validate relationships
        validated = self.validate_relationships(structured)
        
        return validated
```

#### 6. Multi-Stage Preprocessing Pipeline
**Source**: Engineering Drawing Processing, Complex Document Recognition

**Approach**:
1. **Acquisition Enhancement**: Upscaling, denoising
2. **Segmentation**: Layout analysis, region detection
3. **Preprocessing**: Contrast enhancement, binarization
4. **Detection**: Text/symbol/table detection
5. **Extraction**: OCR + VLM processing
6. **Post-processing**: Validation, structuring

**Benefits**:
- Each stage optimized for specific task
- Can skip stages based on document quality
- Supports parallel processing

**Implementation for HVAC**:
```python
class MultiStagePipeline:
    def __init__(self):
        self.stages = [
            QualityAssessment(),
            ImageEnhancement(),
            LayoutSegmentation(),
            RotationCorrection(),
            HybridExtraction(),
            SemanticValidation(),
            StructuredOutput()
        ]
    
    def process(self, blueprint):
        result = blueprint
        metadata = {}
        
        for stage in self.stages:
            # Check if stage is needed
            if stage.should_skip(result, metadata):
                continue
            
            # Process with stage
            result, stage_meta = stage.process(result)
            metadata[stage.name] = stage_meta
        
        return result, metadata
```

## Decision

We will implement a **Hybrid Multi-Stage Document Processing Pipeline** that combines:

1. **Enhanced Preprocessing** with quality assessment and adaptive enhancement
2. **Layout-Aware Segmentation** for region-specific processing
3. **Hybrid OCR + VLM** for accurate text and context extraction
4. **Rotation-Invariant Detection** for engineering drawing annotations
5. **Semantic Caching** for performance optimization
6. **Structured Output** with HVAC domain validation

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Implement layout segmentation module
- [ ] Add rotation-invariant text detection
- [ ] Enhance preprocessing pipeline
- [ ] Create semantic caching layer

### Phase 2: Hybrid Processing (Week 3-4)
- [ ] Integrate VLM with existing OCR
- [ ] Implement context-aware extraction
- [ ] Add table/schedule recognition
- [ ] Create validation framework

### Phase 3: Optimization (Week 5-6)
- [ ] Implement semantic caching
- [ ] Add parallel region processing
- [ ] Optimize VLM inference
- [ ] Create performance benchmarks

### Phase 4: Integration (Week 7-8)
- [ ] Integrate with existing SAHI pipeline
- [ ] Update API endpoints
- [ ] Add monitoring and logging
- [ ] Create documentation and examples

## Consequences

### Positive
- **Accuracy**: 70-90% improvement in text extraction accuracy
- **Coverage**: Handle rotated text, tables, handwritten notes
- **Performance**: 60-80% faster with semantic caching
- **Reliability**: Multi-model validation reduces errors
- **Scalability**: Parallel processing of document regions

### Negative
- **Complexity**: More components to maintain
- **Dependencies**: Additional models (VLM) required
- **Resource Usage**: Higher GPU memory for VLM inference
- **Setup Time**: More complex installation and configuration

### Mitigation
- Modular design allows gradual adoption
- Semantic caching reduces runtime resource needs
- Comprehensive tests ensure reliability
- Fallback to basic OCR if VLM unavailable

## References

1. ArXiv 2411.03707 - Semantic analysis and caching techniques
2. HackerNoon: "How To Process Engineering Drawings With AI" - Layout segmentation and rotation handling
3. HackerNoon: "Complex Document Recognition: OCR Doesn't Work and Here's How You Fix It" - Hybrid OCR+VLM approach
4. HackerNoon: "AI Parsing of Commercial Proposals" - Structured extraction and validation
5. ArXiv: "Document Parsing Unveiled" - Multi-stage pipeline architecture
6. Contextual AI: "Document Parser for RAG" - Semantic understanding and context preservation

## Related ADRs
- [ADR 001: SAHI Integration](001-sahi-integration.md)
- [ADR 002: HVAC Prompt Engineering](002-hvac-prompt-engineering.md)
- [ADR 003: System Relationship Validation](003-system-relationship-validation.md)
