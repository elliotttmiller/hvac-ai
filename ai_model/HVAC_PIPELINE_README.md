# HVAC-Specific Annotation Pipeline

## Overview

This pipeline is specifically designed for HVAC diagrams and technical drawings. It provides specialized handling for:

- **Small text labels** (id_letters, tag_number, etc.)
- **Point annotations** (single-point labels)
- **Polygon to bounding box conversion** with HVAC-specific rules
- **Dataset sanitization** with container validation
- **Comprehensive reporting** with visualizations

## Features

### 1. **Text Label Preservation**
- Detects text-related classes (id_letters, tag_number, etc.)
- Converts single-point text labels to tiny fixed-size boxes
- Ensures small text labels meet minimum size requirements (4px default)
- Preserves critical information that would otherwise be lost

### 2. **Point Annotation Conversion**
- Detects point annotations (single-point polygons)
- Converts to tiny boxes centered on the point
- Maintains annotation integrity for HVAC symbols

### 3. **Smart Polygon Conversion**
- Converts polygons to bounding boxes
- Applies different rules for text vs. objects
- Maintains minimum sizes:
  - Text: 4px minimum
  - Objects: 6px minimum

### 4. **Dataset Sanitization**
Three modes available:
- **Strict**: Delete images with ANY incomplete containers (high quality, less data)
- **Relaxed**: Delete images with NO complete containers (balanced approach)
- **None**: Keep all images (maximum data, lower quality)

Validation criteria:
- Each instrument container must contain BOTH:
  1. `id_letters`
  2. `tag_number`

### 5. **Comprehensive Reporting**
Generates:
- Interactive HTML report with HVAC-specific insights
- CSV file with deletion details
- Visualizations (charts and graphs)
- Visual examples of deleted images (first 10)

## Usage

### Running in Google Colab

```python
# 1. Upload the script to Colab or fetch it from GitHub
!wget https://raw.githubusercontent.com/elliotttmiller/hvac-ai/main/ai_model/hvac_annotation_pipeline.py

# 2. Run the script
!python hvac_annotation_pipeline.py
```

The script will:
1. Install all required dependencies automatically
2. Ask for your Roboflow API key
3. Prompt you to choose a sanitation mode
4. Process your dataset with HVAC-specific rules
5. Generate a comprehensive report

### Running Locally

```bash
# 1. Navigate to the ai_model directory
cd ai_model

# 2. Run the script
python hvac_annotation_pipeline.py
```

### Configuration

Default configuration (can be modified in the script):

```python
config = {
    'workspace_id': 'elliotttmiller',
    'project_id': 'hvacai-s3kda',
    'version': 37,
    'sanitation_mode': 'strict',
    
    # HVAC-specific settings
    'text_classes': [
        'id_letters', 'tag_number', 'text_label', 'value', 
        'tag', 'number', 'label', 'id', 'symbol', 'text'
    ],
    'min_text_size': 4,  # Minimum pixel size for text
    'min_object_size': 6,  # Minimum pixel size for objects
    'point_annotation_size': 4,  # Size for point annotations
    
    # Output settings
    'output_dir': '/content/hvac_pipeline_output',
    'temp_dir': '/content/temp_hvac',
    'generate_report': True,
    'save_deleted_examples': True
}
```

## Output

After running, you'll get:

### 1. Final Dataset Zip
```
/content/hvac_pipeline_output/
└── roboflow_elliotttmiller_hvacai-s3kda_v37_strict_hvac_final.zip
```

### 2. Comprehensive Report
```
/content/hvac_pipeline_output/report/
├── sanitization_report.html           # Interactive HTML report
├── deletion_details.csv               # Detailed CSV log
└── sanitization_visualizations.png    # Charts and graphs
```

### 3. Deleted Examples
```
/content/hvac_pipeline_output/deleted_examples/
├── deleted_image1.png
├── deleted_image2.png
└── ...
```

## HVAC-Specific Metrics

The pipeline tracks and reports:

- **Text annotations preserved**: Number of small text labels kept
- **Small annotations fixed**: Number of annotations expanded to meet minimum size
- **Point annotations processed**: Number of single-point labels converted
- **Polygons converted**: Total polygons converted to bounding boxes
- **Container completion rate**: Percentage of complete containers
- **Deletion rate**: Percentage of images deleted

## Requirements

The script automatically installs all required dependencies:

- roboflow >= 1.0.0
- opencv-python-headless >= 4.8.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- tqdm >= 4.65.0
- pyyaml >= 6.0.0
- requests >= 2.31.0
- scikit-image >= 0.21.0
- pandas >= 2.0.0
- seaborn >= 0.12.0

## Roboflow API Key

You can provide your API key in three ways:

1. **Colab Secrets** (Recommended for Colab):
   - Go to Runtime > Manage secrets
   - Add a secret named `ROBOFLOW_API_KEY`
   - Set its value to your API key

2. **Environment Variable**:
   ```bash
   export ROBOFLOW_API_KEY="your_api_key_here"
   ```

3. **Interactive Input**:
   - The script will prompt you to enter it when running

Get your API key from: https://app.roboflow.com/settings/api

## Pipeline Workflow

1. **Download Dataset**: Fetch from Roboflow API
2. **Convert Polygons**: Convert to bounding boxes with HVAC-specific rules
3. **Sanitize Dataset**: Validate containers and remove incomplete images
4. **Generate Report**: Create HTML/CSV reports with visualizations
5. **Create Final Zip**: Package sanitized dataset

## Best Practices

### For Training Data
- Use `relaxed` or `none` mode to preserve more images
- Focus on quantity while maintaining baseline quality

### For Validation/Test Data
- Use `strict` mode for high-quality evaluation
- Ensures only complete annotations are used

### Improving Annotations
- Review the HTML report to identify missing items
- Check the `most_common_missing` metric
- Focus annotation efforts on frequently missing items
- Review deleted examples to understand patterns

## Troubleshooting

### Import Errors
The script automatically installs dependencies. If you still get import errors:
```bash
pip install roboflow opencv-python-headless numpy matplotlib tqdm pyyaml pandas scikit-image seaborn
```

### API Key Issues
- Verify your API key is correct
- Check you have access to the workspace and project
- Ensure the project version exists

### Memory Issues
- The pipeline processes images in batches
- For very large datasets, consider splitting into smaller versions
- Use a machine with adequate RAM (8GB+ recommended)

## Why This Pipeline Works for HVAC Diagrams

Standard object detection tools often fail on HVAC diagrams because:

1. **Text labels are too small**: Standard conversion discards annotations below a certain size
2. **Point annotations**: Single-point labels have no area and get lost
3. **Polygon complexity**: Technical drawings use complex polygons that need special handling
4. **Domain-specific validation**: HVAC instruments have specific requirements

This pipeline addresses all these issues with HVAC-specific rules and validation.

## Support

For issues or questions:
1. Check the generated HTML report for insights
2. Review the deletion examples to understand patterns
3. Open an issue on the GitHub repository

## License

This pipeline is part of the hvac-ai project. See the main repository for license information.
