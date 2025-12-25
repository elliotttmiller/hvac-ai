# Quick Start Guide: HVAC Annotation Pipeline

## For Google Colab Users (Recommended)

### Step 1: Upload or Fetch the Script

```python
# Option A: Fetch from GitHub
!wget https://raw.githubusercontent.com/elliotttmiller/hvac-ai/main/ai_model/hvac_annotation_pipeline.py

# Option B: Upload manually using Colab's file upload feature
from google.colab import files
uploaded = files.upload()
```

### Step 2: Set Up Your Roboflow API Key (Optional but Recommended)

```python
# Go to Runtime > Manage secrets
# Add a secret named: ROBOFLOW_API_KEY
# Set its value to your API key from: https://app.roboflow.com/settings/api

# Or set it as an environment variable:
import os
os.environ['ROBOFLOW_API_KEY'] = 'your_api_key_here'
```

### Step 3: Run the Pipeline

```python
# Run the script
!python hvac_annotation_pipeline.py

# The script will:
# 1. Auto-install all dependencies
# 2. Ask for your API key (if not set)
# 3. Prompt you to choose sanitation mode
# 4. Process your dataset
# 5. Generate comprehensive reports
```

### Step 4: Download Your Results

```python
# Download the final dataset
from google.colab import files
files.download('/content/hvac_pipeline_output/roboflow_elliotttmiller_hvacai-s3kda_v37_strict_hvac_final.zip')

# Download the HTML report
files.download('/content/hvac_pipeline_output/report/sanitization_report.html')
```

## For Local/Server Users

### Step 1: Navigate to the AI Model Directory

```bash
cd /path/to/hvac-ai/ai_model
```

### Step 2: Set Up Your API Key

```bash
# Set environment variable
export ROBOFLOW_API_KEY="your_api_key_here"

# Or create a .env file
echo "ROBOFLOW_API_KEY=your_api_key_here" > .env
```

### Step 3: Run the Pipeline

```bash
python hvac_annotation_pipeline.py
```

### Step 4: Find Your Results

```bash
# Results will be in:
ls -lh hvac_pipeline_output/

# Final dataset:
ls -lh hvac_pipeline_output/*.zip

# Reports:
ls -lh hvac_pipeline_output/report/

# Deleted examples:
ls -lh hvac_pipeline_output/deleted_examples/
```

## Customization

### Modify Configuration

Edit the `config` dictionary in `run_hvac_pipeline()`:

```python
config = {
    'workspace_id': 'your_workspace',
    'project_id': 'your_project',
    'version': 42,  # Your version number
    'sanitation_mode': 'strict',  # or 'relaxed' or 'none'
    
    # HVAC-specific settings
    'min_text_size': 4,  # Adjust for your text size
    'min_object_size': 6,  # Adjust for your object size
    'point_annotation_size': 4,  # Size for point annotations
    
    # Output paths (adjust for your environment)
    'output_dir': '/content/hvac_pipeline_output',
    'temp_dir': '/content/temp_hvac',
}
```

### Skip Interactive Mode

For automated runs, modify the mode selection:

```python
# Instead of:
mode_choice = input("Choose sanitation mode...")

# Use:
mode_choice = 'strict'  # or 'relaxed' or 'none'
```

## Understanding Sanitation Modes

### Strict Mode (Recommended for Validation/Test Data)
- **Deletes**: Any image with incomplete containers
- **Result**: High quality, fewer images
- **Use when**: You need perfect annotations for evaluation

### Relaxed Mode (Recommended for Training Data)
- **Deletes**: Only images with NO complete containers
- **Result**: Balanced quality and quantity
- **Use when**: You want more training data while maintaining baseline quality

### None Mode (Maximum Data)
- **Deletes**: Nothing (only converts annotations)
- **Result**: All images kept
- **Use when**: You need maximum data volume

## Expected Output

After running, you'll see progress like this:

```
================================================================================
🚀 INITIALIZING HVAC-SPECIFIC ANNOTATION PIPELINE
📦 INSTALLING REQUIRED DEPENDENCIES...
================================================================================
   ✅ roboflow>=1.0.0 already available
   ✅ opencv-python-headless>=4.8.0 already available
   ...

================================================================================
🔑 ROBOFLOW API KEY SETUP
================================================================================
✅ Found API key in environment variables

============================================================
⬇️ DOWNLOADING ROBOFLOW DATASET
Workspace: elliotttmiller
Project: hvacai-s3kda
Version: 37
============================================================
📡 Connecting to Roboflow API...
✅ Dataset downloaded to: /content/temp_hvac/download

============================================================
🔄 CONVERTING POLYGONS TO BOUNDING BOXES (HVAC-SPECIFIC)
   ⚠️  Special handling for text labels, symbols, and small annotations
   📏 Minimum text size: 4px
   📏 Minimum object size: 6px
============================================================
Converting annotations: 100%|██████████| 150/150 [00:05<00:00, 30.12it/s]
✅ Converted 450 polygons to bounding boxes
   📏 Preserved 125 text/symbol annotations
   📏 Fixed 38 small annotations
   📏 Processed 15 point annotations

============================================================
🧹 SANITIZING DATASET (STRICT MODE)
Criteria: Each instrument container must contain BOTH:
  1. 'id_letters'
  2. 'tag_number'
============================================================
Sanitizing images: 100%|██████████| 150/150 [00:02<00:00, 65.23it/s]

✅ Sanitization complete!
   📊 Total images: 150
   ✅ Kept images: 127
   🗑️ Deleted images: 23
   📦 Total containers analyzed: 180
   ✅ Complete containers: 152

============================================================
📋 GENERATING COMPREHENSIVE REPORT
============================================================
✅ Report generated at: /content/hvac_pipeline_output/report/sanitization_report.html
📊 CSV report saved: /content/hvac_pipeline_output/report/deletion_details.csv
📈 Visualizations saved: /content/hvac_pipeline_output/report/sanitization_visualizations.png

📦 Creating final zip archive: /content/hvac_pipeline_output/roboflow_elliotttmiller_hvacai-s3kda_v37_strict_hvac_final.zip
✅ Zip created successfully! Size: 45.23 MB

================================================================================
🎉 PIPELINE COMPLETED SUCCESSFULLY!
⏰ End time: 2024-12-25 10:30:15
⏱️  Total execution time: 127.45 seconds
📤 Final dataset zip: /content/hvac_pipeline_output/roboflow_elliotttmiller_hvacai-s3kda_v37_strict_hvac_final.zip
   📏 Text annotations preserved: 125
   📏 Small annotations fixed: 38
   📏 Point annotations processed: 15
================================================================================
```

## Troubleshooting

### "Module not found" errors
The script auto-installs dependencies. If you still get errors:
```bash
pip install roboflow opencv-python-headless numpy matplotlib tqdm pyyaml pandas scikit-image seaborn
```

### "API key invalid" errors
- Verify your API key at https://app.roboflow.com/settings/api
- Check you have access to the workspace and project
- Ensure the project version exists

### Memory issues
- Use a machine with at least 8GB RAM
- For very large datasets, split into smaller versions

### Permission errors
```bash
# Ensure you have write permissions
chmod +w ai_model/
```

## Next Steps

After running the pipeline:

1. **Review the HTML report** to understand what was processed
2. **Check deleted examples** to see why images were removed
3. **Adjust configuration** if needed and re-run
4. **Use the final ZIP** for training your HVAC model

## Support

For questions or issues:
1. Review the comprehensive README at `ai_model/HVAC_PIPELINE_README.md`
2. Check the HTML report for insights
3. Open an issue on GitHub with:
   - Your configuration
   - Error messages
   - Relevant logs

## Key Features

✅ **Preserves small text labels** (4px minimum)
✅ **Handles point annotations** (single-point labels)
✅ **Smart polygon conversion** (HVAC-specific rules)
✅ **Container validation** (id_letters + tag_number)
✅ **Comprehensive reporting** (HTML + CSV + visualizations)
✅ **Visual deletion examples** (first 10 images)
✅ **Multiple sanitation modes** (strict/relaxed/none)
✅ **Auto-dependency installation**
✅ **Progress tracking** with emojis and progress bars

Designed specifically for HVAC technical drawings! 🔥
