# CPU Compatibility Solution - Autodistill Pipeline

## Critical Discovery

**The Problem:** `autodistill-grounded-sam-2` package has **hardcoded GPU dependencies** that cannot be removed through notebook optimizations alone.

```
autodistill-grounded-sam-2 (package metadata)
└── Requires: autodistill-florence-2
    └── Requires: flash-attn (GPU-only, needs CUDA)
```

This dependency is **baked into the package** at PyPI level, not just in the code.

---

## Why Our CPU Notebook Can't Work As-Is

### What We Did (Correctly) ✅
- CPU-only PyTorch installation
- Removed `torch.cuda` calls
- CPU thread optimization
- All code-level CPU optimizations

### What We Can't Control ❌
- Package-level dependencies defined in `setup.py`
- `autodistill-grounded-sam-2` requires `autodistill-florence-2`
- `autodistill-florence-2` requires `flash-attn`
- `flash-attn` requires CUDA toolkit

**Even when installing with CPU-only PyTorch, pip will try to install these GPU dependencies.**

---

## Solution Options

### Option 1: Remove Grounded-SAM-2 Entirely ⭐ **RECOMMENDED**

Create a **true CPU notebook** that uses CPU-compatible auto-labeling:

**Replace:** Grounded-SAM-2 (GPU-only)  
**With:** Template Matching (OpenCV, CPU-native)

**New Architecture:**
```python
# Instead of:
from autodistill_grounded_sam_2 import GroundedSAM2  # ❌ GPU-only

# Use:
import cv2
import numpy as np
# Classical computer vision template matching ✅ CPU-native
```

**Advantages:**
- ✅ True CPU compatibility (no GPU dependencies)
- ✅ No CUDA downloads
- ✅ Works in any environment
- ✅ Lower accuracy but functional

**Disadvantages:**
- ⚠️ Lower detection accuracy than SAM-2
- ⚠️ Requires careful threshold tuning
- ⚠️ May need pre-processing

### Option 2: Rename to "Training-Only" Notebook

**Accept the limitation** and rebrand:

**Current:** `autodistill_hvac_pipeline_CPU.ipynb`  
**Better:** `autodistill_hvac_training_CPU.ipynb`

**Description:** "CPU-optimized notebook for **training and inference** on pre-labeled datasets. For auto-labeling, use the GPU version."

**Phases Included:**
- Configuration ✅
- YOLOv8 Training ✅ (CPU-compatible)
- Inference ✅ (CPU-compatible)
- Quality metrics ✅

**Phases Excluded:**
- Auto-labeling ❌ (requires GPU version)
- Ontology generation ❌ (requires GPU version)

### Option 3: Create Both Variants

Provide **two CPU notebooks**:

1. **`autodistill_hvac_pipeline_CPU_templatematching.ipynb`**
   - Complete pipeline with OpenCV template matching
   - Lower accuracy, full CPU compatibility
   - For truly GPU-less environments

2. **`autodistill_hvac_training_CPU.ipynb`**
   - Training/inference only
   - Higher accuracy (uses GPU-labeled data)
   - For hybrid workflows

---

## Recommended Implementation

### Create: `autodistill_hvac_pipeline_CPU_templatematching.ipynb`

**Phase Structure:**

1. **Environment Setup** ✅
   ```python
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install opencv-python numpy ultralytics supervision
   # NO autodistill-grounded-sam-2
   ```

2. **Configuration** ✅
   ```python
   TEMPLATE_MATCH_THRESHOLD = 0.8
   NMS_THRESHOLD = 0.3
   ```

3. **Template Matching Auto-Labeling** ✅
   ```python
   import cv2
   
   def template_match_detect(image, template, threshold=0.8):
       result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
       locations = np.where(result >= threshold)
       return bounding_boxes
   ```

4. **Convert to YOLO Format** ✅
   ```python
   def save_yolo_annotations(boxes, classes, image_size, output_path):
       # Standard YOLO format: class x_center y_center width height
       pass
   ```

5. **YOLOv8 Training** ✅
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='dataset.yaml', epochs=50)  # CPU-friendly
   ```

6. **Inference** ✅
   ```python
   results = model.predict(image)
   ```

---

## Update Strategy

### Immediate Actions

1. **Update Current CPU Notebook** ⭐
   - Keep existing file for documentation
   - Add prominent "GPU Required" warning (already done)
   - Explain it's for training/inference only
   - Reference hybrid workflow

2. **Create Template Matching Version** ⭐
   - New file: `autodistill_hvac_pipeline_CPU_templatematching.ipynb`
   - Use OpenCV for auto-labeling
   - True CPU compatibility
   - Document accuracy trade-offs

3. **Update Documentation** ⭐
   - Explain why original CPU version can't work
   - Document package-level dependency issue
   - Provide clear guidance on which notebook to use

### Files to Create/Update

**New Files:**
```
ai_model/notebooks/
├── autodistill_hvac_pipeline_CPU_templatematching.ipynb  ← NEW
└── TEMPLATE_MATCHING_GUIDE.md  ← NEW
```

**Update Files:**
```
ai_model/notebooks/
├── autodistill_hvac_pipeline_CPU.ipynb  ← UPDATE (warning already added)
├── README_CPU_vs_GPU.md  ← UPDATE
└── CPU_VERSION_IMPLEMENTATION.md  ← UPDATE
```

---

## Template Matching Implementation Outline

```python
# Pseudo-code for CPU-compatible auto-labeling

import cv2
import numpy as np
from pathlib import Path

def load_templates(template_dir):
    """Load all template images and extract class names."""
    templates = {}
    for template_path in Path(template_dir).glob('*.PNG'):
        class_name = template_path.stem.replace('template_', '')
        template = cv2.imread(str(template_path), 0)  # Grayscale
        templates[class_name] = template
    return templates

def detect_with_template_matching(image, templates, threshold=0.8):
    """Detect objects using template matching."""
    detections = []
    
    for class_name, template in templates.items():
        # Multi-scale template matching
        for scale in np.linspace(0.5, 2.0, 20):
            resized_template = cv2.resize(template, None, fx=scale, fy=scale)
            
            # Skip if template is larger than image
            if (resized_template.shape[0] > image.shape[0] or 
                resized_template.shape[1] > image.shape[1]):
                continue
            
            # Template matching
            result = cv2.matchTemplate(image, resized_template, 
                                      cv2.TM_CCOEFF_NORMED)
            
            # Find locations above threshold
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                h, w = resized_template.shape
                bbox = [pt[0], pt[1], pt[0]+w, pt[1]+h]
                confidence = result[pt[1], pt[0]]
                detections.append({
                    'class': class_name,
                    'bbox': bbox,
                    'confidence': confidence
                })
    
    # Apply Non-Maximum Suppression
    detections = apply_nms(detections, nms_threshold=0.3)
    
    return detections

def apply_nms(detections, nms_threshold=0.3):
    """Remove overlapping detections."""
    # Use supervision library for NMS
    import supervision as sv
    # Implementation here
    return filtered_detections

def save_yolo_format(detections, image_size, output_path, class_map):
    """Save detections in YOLO format."""
    with open(output_path, 'w') as f:
        for det in detections:
            class_id = class_map[det['class']]
            # Convert to YOLO format (normalized)
            x_center = (det['bbox'][0] + det['bbox'][2]) / 2 / image_size[1]
            y_center = (det['bbox'][1] + det['bbox'][3]) / 2 / image_size[0]
            width = (det['bbox'][2] - det['bbox'][0]) / image_size[1]
            height = (det['bbox'][3] - det['bbox'][1]) / image_size[0]
            
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
```

---

## Performance Comparison

| Method | Accuracy | Speed (CPU) | Dependencies | GPU Required |
|--------|----------|-------------|--------------|--------------|
| **Grounded-SAM-2** | 95%+ | 10-60s/img | Many (GPU) | ✅ YES |
| **Template Matching** | 60-80% | 1-5s/img | Minimal (CPU) | ❌ NO |
| **Manual Labeling** | 100% | 60-300s/img | None | ❌ NO |

---

## Conclusion

**The Issue:** Not a code problem - it's a package architecture issue.

**The Fix:** Create alternative implementations that don't use `autodistill-grounded-sam-2`.

**Next Steps:**
1. Keep current CPU notebook with warnings
2. Create template matching version for true CPU compatibility
3. Update all documentation
4. Guide users to appropriate workflow

---

**Document Status:** Solution Defined  
**Created:** December 15, 2024  
**Next Action:** Create template matching notebook variant
