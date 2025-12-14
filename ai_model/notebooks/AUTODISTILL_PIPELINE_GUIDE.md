# HVAC Autodistill Pipeline - Complete Guide

## 📋 Overview

This guide documents the production-grade HVAC auto-labeling pipeline built with **Grounded-SAM-2** and **Autodistill**. The pipeline follows official best practices and implements advanced features for high-precision detection.

## 🎯 Key Features

### 1. Progress Tracking & Logging ✅
- **Comprehensive Logging System**
  - Timestamped log files in `pipeline_logs/` directory
  - Dual output: file and console logging
  - Detailed logging at every pipeline step
  - Performance metrics tracking

- **ProgressTracker Class**
  - Phase-based timing (Configuration, Ontology Generation, Auto-Labeling, Quality Review, Training)
  - Metrics recording (detection counts, confidence scores, processing times)
  - Comprehensive execution summary

### 2. Optimized Ontology Generation ✅
- **Intelligent Prompt Engineering**
  - Context-aware prompts for HVAC components
  - Enhanced prompts for valves: `"hvac {valve_type}"`
  - Enhanced prompts for instruments: `"hvac control {instrument_type}"`
  - Enhanced prompts for signals: `"{signal_type} line"`

- **Category-Based Organization**
  - Valves (ball, butterfly, globe, etc.)
  - Instruments (discrete, shared, PLC, computer, etc.)
  - Signals (electrical, pneumatic, wireless, etc.)
  - Other components

- **Validation System**
  - Ontology integrity checks
  - Class uniqueness validation
  - Completeness verification
  - Category distribution analysis

### 3. Enhanced Per-Class Detection Strategy ✅
- **Dual Detection Modes**
  ```python
  DETECTION_MODE = 'per_class'  # or 'batch'
  ```

- **Batch Mode**
  - Fast processing of all classes together
  - Leverages multi-class capabilities
  - Suitable for initial testing

- **Per-Class Mode** (Recommended for Production)
  - Iterative detection for each class individually
  - Higher precision for technical drawings
  - Better class separation
  - Reduced false positives
  - Improved confidence scores

- **Detection Process**
  1. Load image
  2. For each class in ontology:
     - Run prediction with class-specific prompt
     - Set class IDs
     - Track detections
  3. Merge all detections
  4. Add to dataset

### 4. Dataset Quality Metrics & Validation ✅
- **Detection Metrics**
  - Total detections across dataset
  - Detections per image (average, min, max)
  - Images with detections (count and percentage)
  - Detection success rate

- **Confidence Score Statistics**
  - Mean confidence
  - Standard deviation
  - Min/Max values
  - Median confidence
  - Quality warnings for low confidence

- **Class Distribution Analysis**
  - Top 15 detected classes with percentages
  - Class balance ratio (most:least common)
  - Imbalance warnings (>10:1 ratio)
  - Unique classes detected

- **Bounding Box Analysis**
  - Average bbox area
  - Median bbox area
  - Standard deviation
  - Size distribution

- **Processing Metrics**
  - Total processing time
  - Average time per image
  - Processing speed (images/second)

### 5. Improved Visualization & Review Workflow ✅
- **Enhanced Statistics Display**
  - Comprehensive detection summaries
  - Visual class distribution charts
  - Quality validation checks
  - Percentage-based breakdowns

- **Annotated Visualization**
  - Color-coded bounding boxes
  - Class name labels
  - Confidence scores overlay
  - Detection count per image

- **Manual Approval Gate**
  - Interactive review checkpoint
  - Quality checklist
  - Approval/rejection workflow
  - Improvement recommendations

## 📊 Pipeline Phases

### Phase 1: Environment Setup
- Install PyTorch with CUDA
- Install Autodistill core
- Install Grounded-SAM-2
- Install YOLOv8
- Install supporting libraries
- Initialize logging system
- Create ProgressTracker

**Output**: Configured environment with logging

### Phase 2: Configuration & Logging
- Set up logging infrastructure
- Configure paths (Colab vs Local)
- Set detection parameters
  - `BOX_THRESHOLD = 0.27`
  - `TEXT_THRESHOLD = 0.22`
- Set training parameters
  - `TRAINING_EPOCHS = 100`
  - `YOLO_MODEL_SIZE = "yolov8n.pt"`
- Initialize progress tracking

**Output**: Configured pipeline with timestamped logs

### Phase 3: Optimized Ontology Generation
- Scan template directory
- Discover all template files
- Apply intelligent prompt engineering
- Categorize classes
- Create CaptionOntology
- Validate ontology integrity
- Log statistics

**Output**: Validated ontology with categorized classes

### Phase 4: Enhanced Auto-Labeling
- Initialize Grounded-SAM-2 model
- Scan for unlabeled images
- Choose detection mode (batch/per_class)
- Process each image:
  - Per-class detection (if enabled)
  - Track statistics
  - Record confidence scores
- Generate YOLO format dataset
- Compute quality metrics

**Output**: Auto-labeled dataset with comprehensive statistics

### Phase 5: Enhanced Quality Review
- Load YOLO dataset
- Compute comprehensive statistics:
  - Detection counts
  - Class distribution
  - Confidence scores
  - Bounding box analysis
  - Class balance
- Visualize annotated samples
- Display quality metrics
- Manual approval checkpoint

**Output**: Validated dataset ready for training

### Phase 6: Model Training
- Load YOLOv8 model with security context
- Configure training parameters
- Train on auto-labeled dataset
- Track training metrics
- Save best checkpoint

**Output**: Trained YOLOv8 model

### Phase 7: Inference
- Load trained model
- Run inference on test images
- Visualize results
- Display detection details
- Save annotated outputs

**Output**: Inference results with annotations

## 🔧 Configuration Options

### Detection Parameters
```python
BOX_THRESHOLD = 0.27   # Range: 0.20-0.35
TEXT_THRESHOLD = 0.22  # Range: 0.15-0.30
```

**Tuning Guide**:
- **Lower thresholds** (0.20-0.25): More detections, higher recall, more false positives
- **Higher thresholds** (0.30-0.35): Fewer detections, higher precision, may miss objects
- **Recommended for HVAC**: 0.25-0.28 for both parameters

### Detection Mode
```python
DETECTION_MODE = 'per_class'  # Options: 'batch' or 'per_class'
```

**Mode Comparison**:
| Feature | Batch Mode | Per-Class Mode |
|---------|-----------|----------------|
| Speed | Fast | Slower |
| Precision | Good | Excellent |
| Class Separation | Moderate | Superior |
| False Positives | More | Fewer |
| Recommended For | Testing | Production |

### Training Parameters
```python
TRAINING_EPOCHS = 100           # Range: 50-200
YOLO_MODEL_SIZE = "yolov8n.pt"  # Options: n, s, m, l, x
```

## 📈 Quality Metrics Interpretation

### Good Quality Indicators
- ✅ Detection coverage > 70% of images
- ✅ Average confidence > 0.40
- ✅ Class imbalance ratio < 10:1
- ✅ Reasonable bbox size distribution
- ✅ Low standard deviation in confidence

### Warning Signs
- ⚠️ Detection coverage < 50%
- ⚠️ Average confidence < 0.30
- ⚠️ Class imbalance ratio > 15:1
- ⚠️ Very high confidence std dev (>0.20)
- ⚠️ Extreme bbox size variance

### Improvement Actions
1. **Low Detection Rate**: Lower BOX_THRESHOLD
2. **Low Confidence**: Refine prompts, increase TEXT_THRESHOLD
3. **Class Imbalance**: Add more diverse training images
4. **High False Positives**: Increase both thresholds
5. **Missed Detections**: Lower both thresholds, improve prompts

## 📝 Log File Analysis

### Log Location
```
pipeline_logs/autodistill_pipeline_YYYYMMDD_HHMMSS.log
```

### Key Log Sections
1. **Phase Timing**: Duration of each phase
2. **Detection Statistics**: Per-class and per-image counts
3. **Quality Metrics**: Confidence scores, validation results
4. **Warnings**: Quality issues, imbalance notifications
5. **Errors**: Processing failures, validation errors

### Example Log Analysis
```
grep "Phase" pipeline_logs/*.log           # View phase timings
grep "WARNING" pipeline_logs/*.log         # Find warnings
grep "ERROR" pipeline_logs/*.log           # Find errors
grep "Metric" pipeline_logs/*.log          # View all metrics
```

## 🎓 Best Practices

### 1. Start Small
- Test with 5-10 images first
- Validate detection quality
- Tune parameters iteratively
- Scale up gradually

### 2. Iterative Improvement
- Review auto-labeled samples carefully
- Adjust thresholds based on results
- Refine prompts for problematic classes
- Add more templates if needed

### 3. Quality Over Quantity
- Prefer per-class mode for production
- Accept only high-quality labels for training
- Manual correction may still be beneficial
- Validate on diverse test images

### 4. Monitor Metrics
- Check all quality metrics before training
- Log and compare across runs
- Track improvement over iterations
- Document parameter changes

### 5. Documentation
- Keep notes on parameter changes
- Document special cases
- Record training runs
- Save successful configurations

## 🚀 Usage Examples

### Google Colab
1. Upload notebook to Colab
2. Mount Google Drive
3. Upload templates to `MyDrive/HVAC_AutoLabeling/hvac_templates/`
4. Upload images to `MyDrive/HVAC_AutoLabeling/hvac_example_images/`
5. Run all cells sequentially
6. Review visualizations and approve dataset
7. Training will save to `MyDrive/HVAC_AutoLabeling/hvac_yolov8_training/`

### Local Environment
1. Ensure templates are in `ai_model/datasets/hvac_templates/hvac_templates/`
2. Ensure images are in `ai_model/datasets/hvac_example_images/hvac_example_images/`
3. Run notebook cells sequentially
4. Outputs will be in `ai_model/outputs/`

## 🔍 Troubleshooting

### Issue: No Detections
**Solutions**:
- Lower BOX_THRESHOLD to 0.20
- Lower TEXT_THRESHOLD to 0.18
- Verify templates are present
- Check image quality
- Review log files for errors

### Issue: Low Confidence Scores
**Solutions**:
- Refine ontology prompts
- Increase TEXT_THRESHOLD slightly
- Check for image quality issues
- Ensure templates match target objects

### Issue: High Class Imbalance
**Solutions**:
- Add more diverse training images
- Use data augmentation in training
- Consider class-weighted loss
- Review class definitions

### Issue: Out of Memory
**Solutions**:
- Use DETECTION_MODE = 'batch'
- Process fewer images at once
- Reduce image resolution
- Use smaller YOLO model (yolov8n)

### Issue: Slow Processing
**Solutions**:
- Ensure GPU is available
- Use DETECTION_MODE = 'batch'
- Process in smaller batches
- Check system resources

## 📚 References

- [Grounded-SAM-2 Official Docs](https://docs.autodistill.com/base_models/grounded-sam-2/)
- [Autodistill GitHub](https://github.com/autodistill/autodistill-grounded-sam-2)
- [Autodistill Quickstart](https://docs.autodistill.com/quickstart/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Tutorials](https://blog.roboflow.com/label-data-with-grounded-sam-2/)

## 🤝 Support

For issues or questions:
1. Check log files in `pipeline_logs/`
2. Review quality metrics output
3. Consult troubleshooting section
4. Check official documentation
5. Review GitHub issues

## 📄 License

This pipeline follows the licenses of its components:
- Autodistill: Apache 2.0
- Grounded-SAM-2: Apache 2.0
- YOLOv8: AGPL-3.0

---

**Last Updated**: 2025-12-14  
**Version**: 2.0.0  
**Author**: HVAC AI Development Team
