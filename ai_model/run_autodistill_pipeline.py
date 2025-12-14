#!/usr/bin/env python3
"""
Production-Grade HVAC Auto-Labeling Pipeline with Grounded-SAM-2
Converted from notebook for end-to-end execution with 1 epoch for testing.
"""

import os
import sys
import logging
import time
import glob
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

print("="*70)
print("🚀 HVAC AUTO-LABELING PIPELINE - AUTOMATED EXECUTION")
print("="*70)

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

IN_COLAB = False  # Running locally
HOME = os.getcwd()
print(f"ℹ️  Running in local environment")
print(f"📂 Working Directory: {HOME}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pipeline configuration
PROCEED_TO_TRAINING = True  # Set to True for automated execution
TRAINING_EPOCHS = 1  # Set to 1 for testing as requested

print(f"\n⚙️  Pipeline Configuration:")
print(f"   • Auto-proceed to training: {PROCEED_TO_TRAINING}")
print(f"   • Training epochs: {TRAINING_EPOCHS}")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

print("\n" + "="*70)
print("📝 SETTING UP LOGGING SYSTEM")
print("="*70)

# Create logs directory
LOG_DIR = os.path.join(os.getcwd(), "pipeline_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"autodistill_pipeline_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("="*70)
logger.info("HVAC AUTO-LABELING PIPELINE - STARTING")
logger.info("="*70)
logger.info(f"Log file: {log_file}")
logger.info(f"Timestamp: {timestamp}")

print(f"✅ Logging system initialized")
print(f"   • Log file: {log_file}")

# ============================================================================
# PROGRESS TRACKING UTILITIES
# ============================================================================

class ProgressTracker:
    """Track progress and performance metrics throughout the pipeline."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.phase_times = {}
        self.metrics = {}
        self.current_phase = None
        self.phase_start = None
    
    def start_phase(self, phase_name):
        """Start tracking a new phase."""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_start = datetime.now()
        logger.info(f"Starting phase: {phase_name}")
    
    def end_phase(self):
        """End current phase and record time."""
        if self.current_phase and self.phase_start:
            duration = (datetime.now() - self.phase_start).total_seconds()
            self.phase_times[self.current_phase] = duration
            logger.info(f"Completed phase: {self.current_phase} (Duration: {duration:.2f}s)")
            self.current_phase = None
            self.phase_start = None
    
    def record_metric(self, metric_name, value):
        """Record a metric value."""
        self.metrics[metric_name] = value
        logger.info(f"Metric - {metric_name}: {value}")
    
    def get_total_time(self):
        """Get total elapsed time."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "="*70)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"\n⏱️  Total Pipeline Time: {self.get_total_time()/60:.2f} minutes")
        
        if self.phase_times:
            print("\n🔄 Phase Breakdown:")
            for phase, duration in self.phase_times.items():
                print(f"   • {phase:<30} {duration:>8.2f}s")
        
        if self.metrics:
            print("\n📈 Key Metrics:")
            for metric, value in self.metrics.items():
                print(f"   • {metric:<30} {value}")
        
        logger.info("Pipeline execution summary printed")

# Initialize global progress tracker
progress = ProgressTracker()
logger.info("Progress tracker initialized")

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

progress.start_phase("Configuration")

# Local environment paths
BASE_PATH = Path.cwd()
TEMPLATE_FOLDER_PATH = str(BASE_PATH / "ai_model" / "datasets" / "hvac_templates" / "hvac_templates")
UNLABELED_IMAGES_PATH = str(BASE_PATH / "ai_model" / "datasets" / "hvac_example_images" / "hvac_example_images")
DATASET_OUTPUT_PATH = str(BASE_PATH / "ai_model" / "outputs" / "autodistill_dataset")
TRAINING_OUTPUT_PATH = str(BASE_PATH / "ai_model" / "outputs" / "yolov8_training")
INFERENCE_OUTPUT_PATH = str(BASE_PATH / "ai_model" / "outputs" / "inference_results")

# Create all required directories
for path_name, path_value in [
    ("Dataset Output", DATASET_OUTPUT_PATH),
    ("Training Output", TRAINING_OUTPUT_PATH),
    ("Inference Output", INFERENCE_OUTPUT_PATH)
]:
    os.makedirs(path_value, exist_ok=True)
    print(f"📂 {path_name}: {path_value}")
    logger.info(f"Created directory: {path_value}")

# Record path configuration
progress.record_metric("Template Path", TEMPLATE_FOLDER_PATH)
progress.record_metric("Images Path", UNLABELED_IMAGES_PATH)
progress.record_metric("Output Path", DATASET_OUTPUT_PATH)

# ============================================================================
# DETECTION PARAMETERS (Research-Based Optimal Values)
# ============================================================================

BOX_THRESHOLD = 0.27
TEXT_THRESHOLD = 0.22

print("\n" + "-"*70)
print("🎯 DETECTION PARAMETERS")
print("-"*70)
print(f"Box Threshold:  {BOX_THRESHOLD:.2f} (optimized for technical drawings)")
print(f"Text Threshold: {TEXT_THRESHOLD:.2f} (optimized for HVAC symbols)")

logger.info(f"Detection parameters - Box: {BOX_THRESHOLD}, Text: {TEXT_THRESHOLD}")
progress.record_metric("Box Threshold", BOX_THRESHOLD)
progress.record_metric("Text Threshold", TEXT_THRESHOLD)

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

YOLO_MODEL_SIZE = "yolov8n.pt"  # Nano model for faster training

print("\n" + "-"*70)
print("🏋️  TRAINING PARAMETERS")
print("-"*70)
print(f"Training Epochs: {TRAINING_EPOCHS}")
print(f"YOLOv8 Model:    {YOLO_MODEL_SIZE}")

logger.info(f"Training parameters - Epochs: {TRAINING_EPOCHS}, Model: {YOLO_MODEL_SIZE}")
progress.record_metric("Training Epochs", TRAINING_EPOCHS)
progress.record_metric("YOLO Model", YOLO_MODEL_SIZE)

progress.end_phase()

print("\n" + "="*70)
print("✅ CONFIGURATION COMPLETE")
print("="*70)
logger.info("Configuration phase completed successfully")

# ============================================================================
# PHASE 3: OPTIMIZED ONTOLOGY GENERATION FROM HVAC TEMPLATES
# ============================================================================

from autodistill.detection import CaptionOntology

progress.start_phase("Ontology Generation")

print("\n" + "="*70)
print("📋 OPTIMIZED HVAC ONTOLOGY GENERATION")
print("="*70)

logger.info("Starting ontology generation from templates")
print(f"\n🔍 Scanning template directory: {TEMPLATE_FOLDER_PATH}")
logger.info(f"Template directory: {TEMPLATE_FOLDER_PATH}")

# Find all template image files
template_extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
all_template_files = []
for ext in template_extensions:
    found_files = glob.glob(os.path.join(TEMPLATE_FOLDER_PATH, ext))
    all_template_files.extend(found_files)
    if found_files:
        logger.info(f"Found {len(found_files)} files with extension {ext}")

if not all_template_files:
    logger.error(f"No template files found in {TEMPLATE_FOLDER_PATH}")
    raise FileNotFoundError(
        f"❌ FATAL ERROR: No template files found in {TEMPLATE_FOLDER_PATH}\n"
        f"   Please ensure template images are present in the directory."
    )

print(f"✅ Found {len(all_template_files)} template files")
logger.info(f"Total templates discovered: {len(all_template_files)}")
progress.record_metric("Template Files Found", len(all_template_files))

# Enhanced prompt engineering
def engineer_prompt(base_name):
    """Apply intelligent prompt engineering for better detection."""
    clean = base_name.replace('template_', '').replace('_', ' ').strip()
    
    if 'valve' in clean.lower():
        prompt = f"hvac {clean}"
    elif 'instrument' in clean.lower():
        prompt = f"hvac control {clean}"
    elif 'signal' in clean.lower():
        prompt = f"{clean} line"
    else:
        prompt = clean
    
    return prompt, clean

# Build ontology
ontology_mapping = {}
categories = defaultdict(list)

print("\n" + "-"*70)
print("📝 PROCESSING TEMPLATES WITH PROMPT ENGINEERING")
print("-"*70)

logger.info("Processing templates and engineering prompts")

for i, template_path in enumerate(sorted(all_template_files), 1):
    filename = os.path.basename(template_path)
    base_name = os.path.splitext(filename)[0]
    
    prompt, class_name = engineer_prompt(base_name)
    ontology_mapping[prompt] = class_name
    
    if 'valve' in class_name.lower():
        categories['Valves'].append(class_name)
    elif 'instrument' in class_name.lower():
        categories['Instruments'].append(class_name)
    elif 'signal' in class_name.lower():
        categories['Signals'].append(class_name)
    else:
        categories['Other'].append(class_name)
    
    if i <= 10:
        print(f"   [{i:2d}] {prompt:<40} -> {class_name}")
        logger.debug(f"Mapped: {prompt} -> {class_name}")

if len(all_template_files) > 10:
    print(f"   ... and {len(all_template_files) - 10} more classes")

# Create ontology object
print("\n" + "-"*70)
print("🏗️  CREATING ONTOLOGY OBJECT")
print("-"*70)

logger.info("Creating CaptionOntology object")
ontology = CaptionOntology(ontology_mapping)
classes = ontology.classes()

print(f"✅ Ontology created successfully")
print(f"✅ Total classes in ontology: {len(classes)}")
logger.info(f"Ontology created with {len(classes)} classes")
progress.record_metric("Ontology Classes", len(classes))

# Category analysis
print("\n" + "-"*70)
print("📊 CATEGORY BREAKDOWN")
print("-"*70)

for category, class_list in sorted(categories.items()):
    print(f"\n{category} ({len(class_list)}):")
    for cls in sorted(class_list)[:5]:
        print(f"   • {cls}")
    if len(class_list) > 5:
        print(f"   ... and {len(class_list) - 5} more")

progress.end_phase()

print("\n" + "="*70)
print("✅ ONTOLOGY GENERATION COMPLETE")
print("="*70)

# ============================================================================
# PHASE 4: ENHANCED AUTO-LABELING WITH PER-CLASS DETECTION
# ============================================================================

from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import DetectionDataset

progress.start_phase("Auto-Labeling")

print("\n" + "="*70)
print("🏷️  AUTO-LABELING WITH GROUNDED-SAM-2")
print("="*70)

logger.info("Initializing Grounded-SAM-2 base model")
print("\n🔧 Initializing Grounded-SAM-2 Model...")

# Initialize base model with optimal parameters
try:
    base_model = GroundedSAM2(
        ontology=ontology,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    print("✅ Grounded-SAM-2 model initialized successfully")
    logger.info("Grounded-SAM-2 model initialized")
except Exception as e:
    logger.error(f"Failed to initialize Grounded-SAM-2: {str(e)}")
    raise

# Find unlabeled images
image_extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
image_paths = []
for ext in image_extensions:
    found_images = glob.glob(os.path.join(UNLABELED_IMAGES_PATH, ext))
    image_paths.extend(found_images)

if not image_paths:
    logger.error(f"No images found in {UNLABELED_IMAGES_PATH}")
    raise FileNotFoundError(f"❌ No images found in {UNLABELED_IMAGES_PATH}")

print(f"\n📸 Found {len(image_paths)} images to label")
logger.info(f"Found {len(image_paths)} images")
progress.record_metric("Images to Label", len(image_paths))

# Per-class auto-labeling
print("\n" + "="*70)
print("🔄 ENHANCED PER-CLASS AUTO-LABELING")
print("="*70)
print(f"\n⚙️  Detection Mode: per_class (optimal for HVAC symbols)")
print(f"   • Processes each class independently for better accuracy")
print(f"   • Reduces false positives and improves precision")

logger.info("Starting per-class auto-labeling")
start_time = time.time()

try:
    base_model.label(
        input_folder=UNLABELED_IMAGES_PATH,
        extension=".jpg",
        output_folder=DATASET_OUTPUT_PATH,
        detection_mode="per_class"  # Use per-class mode for better accuracy
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Auto-labeling complete!")
    print(f"   • Processing time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    print(f"   • Average per image: {elapsed_time/len(image_paths):.2f}s")
    print(f"   • Output saved to: {DATASET_OUTPUT_PATH}")
    
    logger.info(f"Auto-labeling completed in {elapsed_time:.2f}s")
    progress.record_metric("Labeling Time (s)", f"{elapsed_time:.2f}")
    
except Exception as e:
    logger.error(f"Auto-labeling failed: {str(e)}")
    raise

progress.end_phase()

print("\n" + "="*70)
print("✅ AUTO-LABELING COMPLETE")
print("="*70)

# ============================================================================
# PHASE 5: QUALITY REVIEW & VISUALIZATION
# ============================================================================

import supervision as sv

progress.start_phase("Quality Review")

print("\n" + "="*70)
print("🔍 DATASET QUALITY REVIEW")
print("="*70)

logger.info("Loading dataset for quality review")

try:
    review_dataset = DetectionDataset.load(DATASET_OUTPUT_PATH)
    print(f"✅ Dataset loaded successfully")
    print(f"   • Location: {DATASET_OUTPUT_PATH}")
    print(f"   • Images: {len(review_dataset)}")
    print(f"   • Classes: {len(review_dataset.classes)}")
    
    logger.info(f"Dataset loaded: {len(review_dataset)} images, {len(review_dataset.classes)} classes")
    progress.record_metric("Dataset Images", len(review_dataset))
    progress.record_metric("Dataset Classes", len(review_dataset.classes))
    
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    raise

# Compute comprehensive statistics
print("\n" + "-"*70)
print("📊 COMPREHENSIVE DATASET STATISTICS")
print("-"*70)
logger.info("Computing dataset statistics")

total_detections = 0
class_counts = Counter()
images_with_detections = 0
detection_counts_per_image = []
bbox_sizes = []

for image_path, detections in review_dataset:
    num_detections = len(detections)
    detection_counts_per_image.append(num_detections)
    
    if num_detections > 0:
        images_with_detections += 1
        total_detections += num_detections
        
        for class_id in detections.class_id:
            class_name = review_dataset.classes[class_id]
            class_counts[class_name] += 1
        
        if hasattr(detections, 'xyxy') and detections.xyxy is not None:
            for box in detections.xyxy:
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height
                bbox_sizes.append(area)

# Basic statistics
print(f"\n📈 Detection Summary:")
print(f"   • Images with detections: {images_with_detections}/{len(review_dataset)} ({images_with_detections/len(review_dataset)*100:.1f}%)")
print(f"   • Total detections: {total_detections}")
if images_with_detections > 0:
    print(f"   • Average detections per image: {total_detections/images_with_detections:.2f}")
    if detection_counts_per_image and any(c > 0 for c in detection_counts_per_image):
        print(f"   • Min detections in an image: {min([c for c in detection_counts_per_image if c > 0])}")
        print(f"   • Max detections in an image: {max(detection_counts_per_image)}")

logger.info(f"Dataset stats: {total_detections} detections across {images_with_detections} images")
progress.record_metric("Total Dataset Detections", total_detections)
progress.record_metric("Images with Detections", f"{images_with_detections}/{len(review_dataset)}")

# Bounding box statistics
if bbox_sizes:
    print(f"\n📏 Bounding Box Statistics:")
    print(f"   • Average area: {np.mean(bbox_sizes):.1f} px²")
    print(f"   • Median area: {np.median(bbox_sizes):.1f} px²")
    print(f"   • Std deviation: {np.std(bbox_sizes):.1f} px²")
    logger.info(f"Avg bbox area: {np.mean(bbox_sizes):.1f} px²")

# Class distribution
if class_counts:
    print(f"\n🏷️  Class Distribution (Top 15):")
    for i, (class_name, count) in enumerate(class_counts.most_common(15), 1):
        percentage = (count / total_detections) * 100
        print(f"   {i:2d}. {class_name:<35} {count:>3} ({percentage:>5.1f}%)")
        logger.debug(f"Class {class_name}: {count} detections ({percentage:.1f}%)")
    
    if len(class_counts) > 15:
        remaining = len(class_counts) - 15
        remaining_detections = sum(count for _, count in list(class_counts.items())[15:])
        print(f"   ... and {remaining} more classes ({remaining_detections} detections)")
    
    progress.record_metric("Classes with Detections", len(class_counts))
    
    # Class balance analysis
    print(f"\n⚖️  Class Balance Analysis:")
    most_common_count = class_counts.most_common(1)[0][1]
    least_common_count = class_counts.most_common()[-1][1]
    imbalance_ratio = most_common_count / least_common_count if least_common_count > 0 else 0
    print(f"   • Most common class: {most_common_count} detections")
    print(f"   • Least common class: {least_common_count} detections")
    print(f"   • Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print(f"   ⚠️  WARNING: High class imbalance detected")
        logger.warning(f"High class imbalance: {imbalance_ratio:.1f}:1")
    else:
        print(f"   ✅ Reasonable class balance")
    
    logger.info(f"Class balance ratio: {imbalance_ratio:.1f}:1")

progress.end_phase()

print("\n" + "="*70)
print("✅ QUALITY REVIEW COMPLETE")
print("="*70)

# ============================================================================
# PHASE 6: TRAIN YOLOV8 MODEL
# ============================================================================

if PROCEED_TO_TRAINING:
    from autodistill_yolov8 import YOLOv8
    import torch
    import locale
    
    progress.start_phase("Training")
    
    print("\n" + "="*70)
    print("🏋️  TRAINING YOLOV8 MODEL")
    print("="*70)
    
    locale.getpreferredencoding = lambda: "UTF-8"
    
    TRAIN_DATASET_PATH = os.path.join(DATASET_OUTPUT_PATH, "data.yaml")
    
    print(f"\n📋 Training Configuration:")
    print(f"   • Dataset: {TRAIN_DATASET_PATH}")
    print(f"   • Model: {YOLO_MODEL_SIZE}")
    print(f"   • Epochs: {TRAINING_EPOCHS}")
    print(f"   • Output: {TRAINING_OUTPUT_PATH}")
    
    print("\n" + "-"*70)
    print("🏗️  INITIALIZING YOLOV8 MODEL")
    print("-"*70)
    
    from ultralytics.nn.modules import (
        C2f, Detect, Bottleneck, Conv, ConvTranspose, DFL
    )
    
    SAFE_GLOBALS = [
        C2f, Detect, Bottleneck, Conv, ConvTranspose, DFL,
        torch.nn.ModuleList
    ]
    
    try:
        with torch.serialization.safe_globals(SAFE_GLOBALS):
            target_model = YOLOv8(YOLO_MODEL_SIZE)
        
        print(f"✅ YOLOv8 model initialized successfully")
        print(f"   • Architecture: {YOLO_MODEL_SIZE}")
        print(f"   • Secure loading: Enabled")
        
    except Exception as e:
        raise RuntimeError(
            f"❌ FATAL ERROR: Failed to initialize YOLOv8 model\n"
            f"   Error: {str(e)}\n"
            f"   Please ensure YOLOv8 is installed correctly."
        )
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print("\nℹ️  This may take several minutes depending on dataset size and hardware.")
    print("   Training progress will be displayed below...\n")
    
    start_time = time.time()
    
    try:
        target_model.train(
            data_path=TRAIN_DATASET_PATH,
            epochs=TRAINING_EPOCHS,
            project=TRAINING_OUTPUT_PATH
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print(f"⏱️  Total training time: {elapsed_time/60:.2f} minutes")
        print(f"💾 Model saved to: {TRAINING_OUTPUT_PATH}")
        print("\n📊 Check the training output directory for:")
        print("   • weights/best.pt - Best model checkpoint")
        print("   • weights/last.pt - Last epoch checkpoint")
        print("   • Training curves and metrics")
        
        logger.info(f"Training completed in {elapsed_time:.2f}s")
        progress.record_metric("Training Time (s)", f"{elapsed_time:.2f}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TRAINING FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        logger.error(f"Training failed: {str(e)}")
        PROCEED_TO_TRAINING = False
        raise
    
    progress.end_phase()

else:
    print("\n" + "="*70)
    print("⏭️  TRAINING SKIPPED")
    print("="*70)

# ============================================================================
# PHASE 7: INFERENCE WITH TRAINED MODEL
# ============================================================================

if PROCEED_TO_TRAINING:
    from ultralytics import YOLO
    import cv2
    
    progress.start_phase("Inference")
    
    print("\n" + "="*70)
    print("🔮 INFERENCE WITH TRAINED MODEL")
    print("="*70)
    
    print("\n" + "-"*70)
    print("🔍 LOCATING TRAINED MODEL")
    print("-"*70)
    
    run_folders = sorted(glob.glob(os.path.join(TRAINING_OUTPUT_PATH, 'train*')))
    
    if not run_folders:
        raise FileNotFoundError(f"❌ No training runs found in {TRAINING_OUTPUT_PATH}")
    
    latest_run_folder = run_folders[-1]
    TRAINED_MODEL_PATH = os.path.join(latest_run_folder, 'weights/best.pt')
    
    if not os.path.exists(TRAINED_MODEL_PATH):
        raise FileNotFoundError(f"❌ Model checkpoint not found at {TRAINED_MODEL_PATH}")
    
    print(f"✅ Found trained model: {TRAINED_MODEL_PATH}")
    print(f"   • Run folder: {os.path.basename(latest_run_folder)}")
    
    print("\n" + "-"*70)
    print("📥 LOADING TRAINED MODEL")
    print("-"*70)
    
    try:
        model = YOLO(TRAINED_MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {str(e)}")
    
    print("\n" + "-"*70)
    print("🖼️  SELECTING TEST IMAGE")
    print("-"*70)
    
    test_image_paths = []
    for ext in image_extensions:
        test_image_paths.extend(glob.glob(os.path.join(UNLABELED_IMAGES_PATH, ext)))
    
    if not test_image_paths:
        print("⚠️  No images found for inference")
    else:
        inference_image_path = test_image_paths[0]
        print(f"📸 Test image: {os.path.basename(inference_image_path)}")
        
        print("\n" + "-"*70)
        print("🚀 RUNNING INFERENCE")
        print("-"*70)
        
        start_time = time.time()
        
        try:
            results = model(inference_image_path)
            inference_time = time.time() - start_time
            
            annotated_frame = results[0].plot()
            os.makedirs(INFERENCE_OUTPUT_PATH, exist_ok=True)
            
            output_filename = f"inference_result_{os.path.basename(inference_image_path)}"
            output_path = os.path.join(INFERENCE_OUTPUT_PATH, output_filename)
            cv2.imwrite(output_path, annotated_frame)
            
            num_detections = len(results[0].boxes)
            
            print("\n✅ Inference complete")
            print(f"   • Detections found: {num_detections}")
            print(f"   • Inference time: {inference_time:.3f} seconds")
            print(f"   • Result saved to: {output_path}")
            
            if num_detections > 0:
                print("\n📋 Detection Details:")
                for i, box in enumerate(results[0].boxes[:10], 1):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"   {i:2d}. {class_name:<30} (confidence: {confidence:.3f})")
                
                if num_detections > 10:
                    print(f"   ... and {num_detections - 10} more detections")
            
            logger.info(f"Inference completed: {num_detections} detections")
            progress.record_metric("Inference Detections", num_detections)
            progress.record_metric("Inference Time (s)", f"{inference_time:.3f}")
            
        except Exception as e:
            print(f"\n❌ Inference failed: {str(e)}")
            logger.error(f"Inference failed: {str(e)}")
            raise
    
    progress.end_phase()
    
    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETE")
    print("="*70)

else:
    print("\n" + "="*70)
    print("⏭️  INFERENCE SKIPPED")
    print("="*70)

# ============================================================================
# PIPELINE COMPLETE
# ============================================================================

print("\n" + "="*70)
print("🎉 PIPELINE COMPLETE!")
print("="*70)

progress.print_summary()

print("\n📁 Output Locations:")
print(f"   • Dataset: {DATASET_OUTPUT_PATH}")
print(f"   • Training: {TRAINING_OUTPUT_PATH}")
print(f"   • Inference: {INFERENCE_OUTPUT_PATH}")
print(f"   • Logs: {log_file}")

logger.info("="*70)
logger.info("PIPELINE COMPLETED SUCCESSFULLY")
logger.info("="*70)

print("\n✅ All phases completed successfully!")
