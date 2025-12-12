# HVAC-AI Examples

Practical examples demonstrating how to use the HVAC-AI services.

## Available Examples

### 1. Complete HVAC Blueprint Analysis (`hvac_analysis_example.py`)

Demonstrates the complete workflow for HVAC blueprint analysis:

**What it covers:**
- Document processing and quality assessment
- HVAC component detection (simulated)
- System relationship analysis
- Configuration validation
- Result export

**Usage:**
```bash
python examples/hvac_analysis_example.py <path_to_blueprint>

# Example
python examples/hvac_analysis_example.py blueprints/office_plan.pdf
```

**Output:**
```
============================================================
HVAC Blueprint Analysis Pipeline
============================================================

Step 1: Processing Blueprint Document
------------------------------------------------------------
✓ Processed 1 page(s)
  Page 1: Quality 0.75

Step 2: Detecting HVAC Components (SAHI)
------------------------------------------------------------
✓ Detected 4 HVAC components
  duct_001: ductwork (confidence: 0.95)
  diffuser_001: diffuser (confidence: 0.92)
  ...

Step 3: Analyzing System Relationships
------------------------------------------------------------
✓ Built relationship graph with 4 components
✓ Found 3 relationships

Step 4: Validating System Configuration
------------------------------------------------------------
✓ System configuration is VALID
  Components: 4
  Violations: 0
  Warnings: 0

Step 5: Exporting Analysis Results
------------------------------------------------------------
✓ Exported system graph
  Nodes: 4
  Edges: 3

============================================================
Analysis Complete!
============================================================
```

## Running Examples

### Prerequisites

1. **Install Dependencies:**
```bash
pip install -r python-services/requirements.txt
```

2. **Setup Environment:**
```bash
# Copy and configure .env
cp .env.example .env
```

3. **Prepare Test Data:**
```bash
# Place test blueprints in a directory
mkdir -p test-blueprints
# Add your HVAC blueprint files
```

### Common Issues

**Import Errors:**
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Missing SAHI:**
```bash
# Install SAHI
pip install sahi>=0.11.0
```

**OpenCV Errors:**
```bash
# Install OpenCV
pip install opencv-python>=4.8.0
```

## Example Modifications

### Use Real SAHI Detection

Replace the simulated detection section with:

```python
# Step 2: Component Detection with SAHI
predictor = create_hvac_sahi_predictor(
    model_path="models/sam_hvac_finetuned.pth",
    device="cuda"  # or "cpu"
)

result = predictor.predict_hvac_components(
    image_path=processed_image_path,
    adaptive_slicing=True
)

detections = result['detections']
```

### Save Results to File

Add at the end of analysis:

```python
import json

with open('analysis_results.json', 'w') as f:
    json.dump({
        'document': doc_result,
        'detections': detections,
        'system_graph': graph_export,
        'validation': validation
    }, f, indent=2)

print("✓ Results saved to analysis_results.json")
```

### Add Visualization

```python
import matplotlib.pyplot as plt
import cv2

# Load image
image = cv2.imread(blueprint_path)

# Draw detections
for detection in detections:
    x, y, w, h = detection['bbox']
    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    cv2.putText(image, detection['type'].value, (int(x), int(y-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save visualization
cv2.imwrite('analysis_visualization.png', image)
print("✓ Visualization saved to analysis_visualization.png")
```

## Creating New Examples

Template for new examples:

```python
#!/usr/bin/env python3
"""
Example: [Description]

Usage:
    python examples/my_example.py
"""

import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.hvac_ai.hvac_sahi_engine import create_hvac_sahi_predictor

def main():
    """Main function"""
    print("Example starting...")
    
    # Your code here
    
    print("✓ Example complete")

if __name__ == "__main__":
    main()
```

## Contributing

When adding new examples:
1. Follow the existing structure and style
2. Include comprehensive docstrings
3. Add error handling
4. Update this README
5. Test with sample data

## Support

For example-related questions:
- Review existing examples for patterns
- Check service documentation in `services/README.md`
- See main documentation in `docs/`

---

**Examples Status:** Active Development  
**Last Updated:** December 2024
