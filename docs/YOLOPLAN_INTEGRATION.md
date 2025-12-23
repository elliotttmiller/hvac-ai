# YOLOplan Integration Plan

## Overview

YOLOplan is an open-source project by DynMEP that automates symbol detection and counting in technical drawings (floor plans, PDFs, CAD files) for MEP (Mechanical, Electrical, Plumbing) professionals. This document outlines the integration plan for incorporating YOLOplan capabilities into the HVAC AI platform.

## YOLOplan Analysis

### Key Features

**Core Capabilities:**
- Symbol detection and counting in technical drawings
- Multi-format support: PDF, JPG, PNG, BMP, TIFF
- Batch processing for multiple drawings
- Custom training for specialized symbols
- Export to CSV, Excel, JSON
- Netlist connectivity analysis for electrical drawings
- Adaptive preprocessing and synthetic data generation
- Hyperparameter optimization with Optuna
- Parallel processing for performance

**Architecture:**
- Based on YOLO11/YOLOv8 (state-of-the-art object detection)
- Web interface (Streamlit) for user interaction
- Configurable detection parameters
- Model layer with trained weights
- Automated result exports

**MEP Use Cases:**
- Symbol takeoff/counting for BOMs
- Quantity extraction for construction estimation
- BIM workflow integration
- Asset inventory from as-built drawings
- QA/QC verification
- Legacy drawing digitization

### Technical Specifications

**Detection Performance:**
- YOLOv5/v8/v11 architectures
- Higher mAP than Faster R-CNN on floor plans
- Fast inference (real-time capable)
- Robust to noisy/complex drawings

**Supported Symbols:**
- Electrical symbols
- HVAC equipment symbols
- Plumbing fixtures
- Architectural elements
- Custom symbols (via training)

## Integration Strategy

### Phase 1: Analysis and Evaluation (Week 1-2)

**Objective:** Understand YOLOplan codebase and assess compatibility

**Tasks:**
1. Clone and analyze YOLOplan repository
2. Review model architecture and training pipeline
3. Evaluate pre-trained models on HVAC drawings
4. Identify integration points with existing system
5. Assess licensing and dependencies

**Deliverables:**
- Technical analysis document
- Compatibility assessment
- Integration architecture diagram

### Phase 2: Core Integration (Weeks 3-6)

**Objective:** Integrate YOLOplan detection engine into HVAC AI platform

#### 2.1 Model Integration

```python
# python-services/core/ai/yoloplan_detector.py

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class YOLOplanDetector:
    """
    YOLOplan-based symbol detector for MEP drawings
    
    Integrates DynMEP/YOLOplan capabilities for:
    - HVAC equipment symbol detection
    - Electrical symbol detection
    - Plumbing fixture detection
    - Custom symbol detection
    """
    
    def __init__(self, model_path: str = "models/yoloplan_hvac.pt"):
        """
        Initialize YOLOplan detector
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.model = YOLO(model_path)
        self.symbol_classes = self._load_symbol_classes()
        logger.info(f"YOLOplan detector initialized with {len(self.symbol_classes)} classes")
        
    def detect_symbols(self, image: np.ndarray,
                      confidence: float = 0.5,
                      symbol_types: List[str] = None) -> Dict[str, Any]:
        """
        Detect MEP symbols in drawing
        
        Args:
            image: Input blueprint image
            confidence: Detection confidence threshold
            symbol_types: Filter for specific symbol types
            
        Returns:
            Detection results with symbols, counts, locations
        """
        # Run detection
        results = self.model.predict(
            image,
            conf=confidence,
            verbose=False
        )[0]
        
        # Parse results
        detections = self._parse_detections(results, symbol_types)
        
        # Count symbols by type
        counts = self._count_symbols(detections)
        
        # Generate netlist (connectivity)
        netlist = self._generate_netlist(detections)
        
        return {
            'detections': detections,
            'counts': counts,
            'netlist': netlist,
            'total_symbols': len(detections)
        }
    
    def batch_detect(self, images: List[np.ndarray],
                    confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Batch detection for multiple drawings
        
        Args:
            images: List of blueprint images
            confidence: Detection confidence threshold
            
        Returns:
            List of detection results
        """
        results = []
        for image in images:
            result = self.detect_symbols(image, confidence)
            results.append(result)
        
        return results
    
    def _parse_detections(self, results, symbol_types):
        """Parse YOLO results into structured format"""
        detections = []
        
        boxes = results.boxes
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = self.symbol_classes[class_id]
            
            # Filter by symbol type if specified
            if symbol_types and class_name not in symbol_types:
                continue
            
            detection = {
                'id': i,
                'symbol_type': class_name,
                'bbox': box.xyxy[0].tolist(),
                'confidence': float(box.conf[0]),
                'center': self._compute_center(box.xyxy[0].tolist())
            }
            
            detections.append(detection)
        
        return detections
    
    def _count_symbols(self, detections):
        """Count symbols by type"""
        counts = {}
        for det in detections:
            symbol_type = det['symbol_type']
            counts[symbol_type] = counts.get(symbol_type, 0) + 1
        
        return counts
    
    def _generate_netlist(self, detections):
        """Generate connectivity netlist for electrical/HVAC systems"""
        # TODO: Implement connectivity analysis
        # - Find connections between symbols
        # - Build system graph
        # - Identify circuits/zones
        
        return {
            'connections': [],
            'circuits': [],
            'zones': []
        }
    
    def _compute_center(self, bbox):
        """Compute center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def _load_symbol_classes(self):
        """Load symbol class names from model"""
        return self.model.names
    
    def export_results(self, results: Dict[str, Any],
                      format: str = 'json') -> str:
        """
        Export detection results
        
        Args:
            results: Detection results
            format: Export format ('json', 'csv', 'excel')
            
        Returns:
            Exported data as string or file path
        """
        if format == 'json':
            import json
            return json.dumps(results, indent=2)
        
        elif format == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.DictWriter(output, 
                                   fieldnames=['id', 'symbol_type', 'confidence', 'x', 'y'])
            writer.writeheader()
            
            for det in results['detections']:
                writer.writerow({
                    'id': det['id'],
                    'symbol_type': det['symbol_type'],
                    'confidence': det['confidence'],
                    'x': det['center'][0],
                    'y': det['center'][1]
                })
            
            return output.getvalue()
        
        elif format == 'excel':
            # TODO: Implement Excel export
            raise NotImplementedError("Excel export not yet implemented")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
```

#### 2.2 Training Pipeline

```python
# python-services/core/ai/yoloplan_trainer.py

class YOLOplanTrainer:
    """
    Training pipeline for YOLOplan models on HVAC symbols
    
    Features:
    - Custom dataset preparation
    - Hyperparameter optimization with Optuna
    - Synthetic data generation
    - Model evaluation and validation
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        
    def prepare_dataset(self):
        """
        Prepare HVAC symbol dataset
        
        Structure:
        - images/train/
        - images/val/
        - labels/train/
        - labels/val/
        - data.yaml
        """
        pass
    
    def generate_synthetic_data(self, n_samples: int = 1000):
        """
        Generate synthetic HVAC drawings with symbols
        
        Techniques:
        - Template-based generation
        - Symbol placement variations
        - Background augmentation
        - Noise and distortion
        """
        pass
    
    def optimize_hyperparameters(self, n_trials: int = 50):
        """
        Hyperparameter optimization with Optuna
        
        Optimizes:
        - Learning rate
        - Batch size
        - Image size
        - Augmentation parameters
        """
        pass
    
    def train(self, epochs: int = 100, batch_size: int = 16):
        """
        Train YOLOplan model on HVAC symbols
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
        """
        pass
```

#### 2.3 Integration with Existing Pipeline

```python
# python-services/core/ai/integrated_detector.py

class IntegratedHVACDetector:
    """
    Integrated detector combining:
    - YOLOplan for MEP symbols
    - SAHI for large blueprint components
    - Enhanced document processing for text
    """
    
    def __init__(self):
        self.yoloplan = YOLOplanDetector()
        self.sahi_detector = HVACDetector()  # Existing
        self.doc_processor = create_enhanced_processor()  # Existing
        self.hybrid_processor = create_hybrid_processor()  # Existing
        
    def analyze_blueprint(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Complete blueprint analysis
        
        Pipeline:
        1. Document processing (text, metadata)
        2. YOLOplan symbol detection
        3. SAHI component detection
        4. Result fusion and validation
        """
        # Document processing
        doc_results = self.doc_processor.process(image)
        text_results = self.hybrid_processor.process(image)
        
        # Symbol detection with YOLOplan
        symbol_results = self.yoloplan.detect_symbols(image)
        
        # Component detection with SAHI
        component_results = self.sahi_detector.detect_with_sahi(image)
        
        # Merge results
        merged = self._merge_results(
            doc_results,
            text_results,
            symbol_results,
            component_results
        )
        
        return merged
    
    def _merge_results(self, doc, text, symbols, components):
        """Merge and validate all detection results"""
        return {
            'document': doc,
            'text': text,
            'symbols': symbols,
            'components': components,
            'metadata': {
                'total_symbols': symbols['total_symbols'],
                'total_components': len(components),
                'text_elements': len(text['results'])
            }
        }
```

### Phase 3: Custom Training (Weeks 7-10)

**Objective:** Train YOLOplan models on HVAC-specific symbols

**Tasks:**
1. Collect HVAC symbol dataset
   - Equipment symbols (AHU, fan, damper, etc.)
   - Duct/pipe symbols
   - Sensor/control symbols
   - Minimum 1,000 annotated images

2. Data preparation and augmentation
   - Annotation in YOLO format
   - Synthetic data generation
   - Augmentation pipeline

3. Model training
   - Transfer learning from YOLOv11
   - Hyperparameter optimization
   - Multi-scale training

4. Evaluation and validation
   - mAP, precision, recall
   - Real-world HVAC drawing tests
   - A/B comparison with existing detector

**Expected Improvements:**
- Symbol detection accuracy: 85% → 95%+
- New symbol types supported: 50+
- Detection speed: <2s per drawing

### Phase 4: Advanced Features (Weeks 11-14)

**Objective:** Implement YOLOplan advanced features

#### 4.1 Connectivity Analysis

```python
class ConnectivityAnalyzer:
    """
    Netlist generation for HVAC systems
    
    Analyzes:
    - Duct connections
    - Pipe routing
    - Equipment relationships
    - Zone assignments
    """
    
    def generate_netlist(self, symbols: List[Dict], 
                        connections: List[Dict]) -> Dict:
        """
        Generate system connectivity netlist
        
        Returns:
        - Connection graph
        - System hierarchy
        - Zone assignments
        """
        pass
```

#### 4.2 BOM Generation

```python
class BOMGenerator:
    """
    Bill of Materials generation from symbol counts
    
    Features:
    - Symbol to equipment mapping
    - Quantity takeoff
    - Specification lookup
    - Cost estimation integration
    """
    
    def generate_bom(self, symbol_counts: Dict) -> List[Dict]:
        """
        Generate BOM from detected symbols
        
        Returns list of:
        - Item description
        - Quantity
        - Unit
        - Specifications
        - Cost (if available)
        """
        pass
```

#### 4.3 Batch Processing

```python
class BatchProcessor:
    """
    Batch processing for multiple drawings
    
    Features:
    - Parallel processing
    - Progress tracking
    - Error handling
    - Consolidated reporting
    """
    
    def process_batch(self, drawing_paths: List[str],
                     output_dir: str) -> Dict:
        """
        Process multiple drawings in batch
        
        Returns:
        - Individual results per drawing
        - Aggregate statistics
        - Export files (CSV, Excel)
        """
        pass
```

### Phase 5: UI Integration (Weeks 15-16)

**Objective:** Add YOLOplan features to frontend

**Features:**
1. Symbol detection toggle/settings
2. Symbol counts display
3. BOM generation and export
4. Connectivity visualization
5. Symbol type filtering

**UI Components:**
```typescript
// src/components/yoloplan/SymbolDetection.tsx

interface SymbolDetectionProps {
  blueprintId: string;
  onDetectionComplete: (results: SymbolResults) => void;
}

export function SymbolDetection({ blueprintId, onDetectionComplete }: SymbolDetectionProps) {
  // Symbol detection UI
  // - Enable/disable symbol detection
  // - Configure detection parameters
  // - Display symbol counts
  // - Export BOM
}
```

## Benefits of Integration

### For HVAC AI Platform

1. **Enhanced Symbol Detection**
   - Specialized MEP symbol recognition
   - Custom HVAC symbol training
   - Higher accuracy on technical symbols

2. **Quantity Takeoff**
   - Automated BOM generation
   - Material quantity estimation
   - Cost estimation support

3. **Connectivity Analysis**
   - System netlist generation
   - Zone identification
   - Circuit/system validation

4. **Batch Processing**
   - Process multiple drawings efficiently
   - Consolidated reporting
   - Time savings for large projects

5. **Standards Compliance**
   - Symbol standardization
   - Drawing validation
   - QA/QC automation

### For Users

1. **Time Savings**
   - Automated symbol counting (vs. manual)
   - Batch processing capability
   - Faster quantity takeoff

2. **Accuracy**
   - Consistent detection
   - Reduced human error
   - Validated results

3. **Integration**
   - BIM workflow support
   - Export to estimating tools
   - Asset management integration

## Implementation Roadmap

### Timeline: 16 Weeks

**Weeks 1-2: Analysis**
- Clone YOLOplan repository
- Evaluate models on HVAC drawings
- Design integration architecture
- Document findings

**Weeks 3-6: Core Integration**
- Implement YOLOplan detector wrapper
- Integrate with existing pipeline
- Add export functionality
- Unit tests

**Weeks 7-10: Custom Training**
- Collect HVAC symbol dataset
- Prepare annotations
- Train custom models
- Validate performance

**Weeks 11-14: Advanced Features**
- Connectivity analysis
- BOM generation
- Batch processing
- Integration tests

**Weeks 15-16: UI Integration**
- Frontend components
- API endpoints
- User documentation
- End-to-end testing

### Success Metrics

**Technical:**
- Symbol detection accuracy: 95%+
- Processing speed: <2s per drawing
- Batch throughput: 50+ drawings/hour
- Model size: <100MB

**Business:**
- User adoption: 80%+ of users enable symbol detection
- Time savings: 70%+ reduction in manual takeoff
- Accuracy improvement: 90%+ vs. manual counting
- BOM export usage: 60%+ of analyses

## Dependencies

**Python Packages:**
```
ultralytics>=8.0.0  # YOLO models
optuna>=3.0.0       # Hyperparameter optimization
roboflow>=1.0.0     # Dataset management (optional)
streamlit>=1.30.0   # UI (optional, for standalone)
openpyxl>=3.0.0     # Excel export
```

**Infrastructure:**
- GPU: NVIDIA GPU for training (A100 recommended)
- Storage: 100GB+ for datasets and models
- Compute: 16GB+ RAM for inference

## Risks and Mitigation

### Technical Risks

1. **Model Performance**
   - Risk: Pre-trained models may not work well on HVAC
   - Mitigation: Custom training on HVAC dataset, transfer learning

2. **Integration Complexity**
   - Risk: Integration with existing system may be complex
   - Mitigation: Modular design, comprehensive testing

3. **Licensing**
   - Risk: YOLOplan license may restrict use
   - Mitigation: Review license, contact DynMEP if needed

### Business Risks

1. **Development Time**
   - Risk: 16 weeks may be too long
   - Mitigation: Phased rollout, MVP in 6 weeks

2. **User Adoption**
   - Risk: Users may not understand benefits
   - Mitigation: Clear documentation, training, examples

3. **Maintenance**
   - Risk: YOLOplan updates may break integration
   - Mitigation: Version pinning, automated tests

## Next Steps

**Immediate Actions:**
1. Review and approve integration plan
2. Clone YOLOplan repository for evaluation
3. Test pre-trained models on sample HVAC drawings
4. Prepare dataset collection plan
5. Allocate resources (GPU, team)

**Week 1 Deliverable:**
- YOLOplan evaluation report
- Integration architecture document
- Dataset requirements specification
- Resource allocation plan

## References

1. **YOLOplan Repository**: https://github.com/DynMEP/YOLOplan
2. **YOLOplan Documentation**: https://github.com/DynMEP/YOLOplan/blob/main/docs/YOLOplan.md
3. **Ultralytics YOLO**: https://docs.ultralytics.com
4. **Floor Plan Object Detection Research**: Various academic papers on YOLO for floor plans

## Conclusion

Integrating YOLOplan into the HVAC AI platform will significantly enhance symbol detection capabilities, enable automated quantity takeoff and BOM generation, and provide connectivity analysis for MEP systems. The modular integration approach ensures compatibility with existing systems while providing clear benefits to users.

This integration complements the recently implemented advanced document processing (OCR + VLM) by adding specialized symbol detection, creating a comprehensive blueprint analysis platform.

**Combined Capabilities:**
- Document processing: Text extraction, quality assessment, layout segmentation
- Symbol detection: MEP symbols, equipment, fixtures
- Component detection: Large HVAC components with SAHI
- Connectivity: System relationships and netlists
- Export: BOM, CSV, Excel, JSON

**This positions the HVAC AI platform as an industry-leading solution for complete MEP drawing analysis.**
