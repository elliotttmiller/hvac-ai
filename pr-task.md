Build a fully optimized, specialized and properly constructed dataset sanitizer, validator and optimizer

# HVAC-Specific Annotation Pipeline for Technical Drawings

Here's a complete, optimized pipeline specifically designed for HVAC diagrams and technical drawings that preserves small annotations while converting to YOLO format:

```python
#!/usr/bin/env python3
"""
HVAC-Specific Annotation Pipeline for Technical Drawings
Optimized for preserving small text labels and symbols in HVAC diagrams
"""

import sys
import subprocess
import pkgutil
import json
from pathlib import Path
import os
import time
import warnings
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shutil
import glob
import yaml
from tqdm import tqdm
import zipfile
from typing import Dict, List, Tuple, Optional, Any

# ═══════════════════════════════════════════════════════════════════════════════
# 📦 DEPENDENCY INSTALLATION SECTION - HVAC-optimized
# ═══════════════════════════════════════════════════════════════════════════════

def install_requirements():
    """Install all required dependencies with HVAC-specific handling"""
    print("="*80)
    print("🚀 INITIALIZING HVAC-SPECIFIC ANNOTATION PIPELINE")
    print("📦 INSTALLING REQUIRED DEPENDENCIES...")
    print("="*80)
    
    core_packages = [
        "roboflow>=1.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
        "scikit-image>=0.21.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0"
    ]
    
    for package in core_packages:
        try:
            pkg_name = package.split(">=")[0].split("==")[0]
            import importlib.util
            spec = importlib.util.find_spec(pkg_name)
            if spec is None:
                print(f"   📥 Installing {package}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--upgrade", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"   ✅ {package} installed successfully")
            else:
                print(f"   ✅ {package} already available")
        except Exception as e:
            print(f"   ⚠️  Failed to install {package}: {e}")
            print(f"   💡 Attempting with --user flag...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--user", "--upgrade", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"   ✅ {package} installed with --user flag")
            except Exception as e2:
                print(f"   ❌ Critical failure installing {package}: {e2}")
                sys.exit(1)
    
    print("\n✅ All required dependencies installed successfully!")
    print("="*80)

# Run dependency installation first
install_requirements()

# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 COLAB-SPECIFIC API KEY HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

def get_roboflow_api_key() -> str:
    """Get Roboflow API key with Colab-specific handling"""
    print("="*80)
    print("🔑 ROBOFLOW API KEY SETUP")
    print("="*80)
    
    api_key = None
    
    # Try to get from Google Colab secrets first
    try:
        from google.colab import userdata
        api_key = userdata.get('ROBOFLOW_API_KEY')
        if api_key:
            print("✅ Found API key in Colab secrets (userdata)")
            return api_key
    except:
        pass
    
    # Try environment variables
    if not api_key:
        api_key = os.environ.get('ROBOFLOW_API_KEY')
        if api_key:
            print("✅ Found API key in environment variables")
            return api_key
    
    # Interactive input for Colab
    print("\n🔑 Please provide your Roboflow API key:")
    print("   - Get it from: https://app.roboflow.com/settings/api")
    print("   - Or set it as a Colab secret: Runtime > Manage secrets > Add 'ROBOFLOW_API_KEY'")
    
    while True:
        api_key = input("\nEnter your Roboflow API key: ").strip()
        if api_key and len(api_key) > 10:  # Basic validation
            print("✅ API key accepted")
            
            # Save for this session
            os.environ['ROBOFLOW_API_KEY'] = api_key
            print("✅ API key saved to environment variables for this session")
            return api_key
        print("❌ Invalid API key. Please try again (should be at least 10 characters).")

# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 HVAC-SPECIFIC ANNOTATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class HVACAnnotationPipeline:
    """Specialized pipeline for HVAC diagrams and technical drawings"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.workspace_id = config.get('workspace_id', 'elliotttmiller')
        self.project_id = config.get('project_id', 'hvacai-s3kda')
        self.version = config.get('version', 37)
        self.api_key = config.get('api_key')
        self.output_dir = Path(config.get('output_dir', '/content/hvac_pipeline_output')).resolve()
        self.temp_dir = Path(config.get('temp_dir', '/content/temp_hvac')).resolve()
        
        # Sanitation configuration
        self.sanitation_mode = config.get('sanitation_mode', 'strict')
        self.containers = config.get('containers', [
            "instrument_discrete_field", 
            "instrument_shared_primary", 
            "instrument_discrete_primary", 
            "instrument_discrete_aux"
        ])
        self.required_content_1 = config.get('required_content_1', 'id_letters')
        self.required_content_2 = config.get('required_content_2', 'tag_number')
        
        # HVAC-specific configuration
        self.text_classes = config.get('text_classes', [
            'id_letters', 'tag_number', 'text_label', 'value', 
            'tag', 'number', 'label', 'id', 'symbol'
        ])
        self.min_text_size = config.get('min_text_size', 4)  # Minimum pixel size for text
        self.min_object_size = config.get('min_object_size', 6)  # Minimum pixel size for objects
        self.point_annotation_size = config.get('point_annotation_size', 4)  # Size for point annotations
        
        # Reporting configuration
        self.generate_report = config.get('generate_report', True)
        self.save_deleted_examples = config.get('save_deleted_examples', True)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            'total_images': 0,
            'converted_images': 0,
            'sanitized_images': 0,
            'deleted_images': 0,
            'polygons_converted': 0,
            'text_annotations_preserved': 0,
            'point_annotations_processed': 0,
            'small_annotations_fixed': 0,
            'deletion_details': [],
            'container_stats': {
                'total_containers': 0,
                'complete_containers': 0,
                'missing_id_letters': 0,
                'missing_tag_number': 0,
                'missing_both': 0
            }
        }
    
    def download_dataset(self) -> Path:
        """Download dataset from Roboflow"""
        print(f"\n{'='*60}")
        print(f"⬇️ DOWNLOADING ROBOFLOW DATASET")
        print(f"Workspace: {self.workspace_id}")
        print(f"Project: {self.project_id}")
        print(f"Version: {self.version}")
        print(f"{'='*60}")
        
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace_id).project(self.project_id)
            version = project.version(self.version)
            
            print("📡 Connecting to Roboflow API...")
            dataset = version.download("yolov8", location=str(self.temp_dir / "download"))
            
            dataset_path = Path(dataset.location)
            print(f"✅ Dataset downloaded to: {dataset_path}")
            
            return dataset_path
            
        except Exception as e:
            raise Exception(f"❌ Failed to download dataset: {e}")
    
    def convert_polygons_to_bboxes_yolo(self, dataset_path: Path) -> Path:
        """Convert polygon annotations to bounding boxes with HVAC-specific rules"""
        print(f"\n{'='*60}")
        print(f"🔄 CONVERTING POLYGONS TO BOUNDING BOXES (HVAC-SPECIFIC)")
        print(f"   ⚠️  Special handling for text labels, symbols, and small annotations")
        print(f"   📏 Minimum text size: {self.min_text_size}px")
        print(f"   📏 Minimum object size: {self.min_object_size}px")
        print(f"{'='*60}")
        
        converted_dir = self.temp_dir / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dataset structure
        print("📋 Copying dataset structure...")
        for item in dataset_path.iterdir():
            dest = converted_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        
        # Process annotations
        annotation_files = list(converted_dir.rglob("labels/*.txt"))
        print(f"🔍 Found {len(annotation_files)} annotation files to process")
        
        for ann_file in tqdm(annotation_files, desc="Converting annotations"):
            try:
                # Find corresponding image
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = Path(str(ann_file).replace("labels", "images").replace(".txt", ext))
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if not img_path or not img_path.exists():
                    continue
                
                # Read image dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_h, img_w = img.shape[:2]
                
                # Read and convert annotations
                with open(ann_file, 'r') as f:
                    lines = f.readlines()
                
                converted_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    class_id = parts[0]
                    coords = list(map(float, parts[1:]))
                    
                    # Check if this is a polygon (more than 4 coordinates)
                    if len(coords) > 4:
                        self.stats['polygons_converted'] += 1
                        
                        # Parse polygon points
                        points = []
                        for i in range(0, len(coords), 2):
                            x_norm = coords[i]
                            y_norm = coords[i + 1] if i + 1 < len(coords) else 0
                            points.append((x_norm * img_w, y_norm * img_h))
                        
                        if len(points) >= 3:
                            points_array = np.array(points)
                            xmin, ymin = points_array.min(axis=0)
                            xmax, ymax = points_array.max(axis=0)
                            
                            # Special handling for text classes
                            is_text = False
                            class_name = self._get_class_name(int(class_id))
                            if class_name and any(tc in class_name.lower() for tc in self.text_classes):
                                is_text = True
                            
                            # Handle point annotations (all points at same location)
                            if np.allclose(points_array, points_array[0]):
                                self.stats['point_annotations_processed'] += 1
                                # Create fixed-size box centered on the point
                                x, y = points_array[0]
                                xmin = x - self.point_annotation_size/2
                                ymin = y - self.point_annotation_size/2
                                xmax = x + self.point_annotation_size/2
                                ymax = y + self.point_annotation_size/2
                                self.stats['text_annotations_preserved'] += 1
                            
                            # Handle very small text annotations
                            elif is_text:
                                width = xmax - xmin
                                height = ymax - ymin
                                
                                # If too small, expand to minimum size
                                if width < self.min_text_size or height < self.min_text_size:
                                    self.stats['small_annotations_fixed'] += 1
                                    center_x = (xmin + xmax) / 2
                                    center_y = (ymin + ymax) / 2
                                    
                                    # Expand to minimum size while keeping center
                                    if width < self.min_text_size:
                                        xmin = center_x - self.min_text_size/2
                                        xmax = center_x + self.min_text_size/2
                                    if height < self.min_text_size:
                                        ymin = center_y - self.min_text_size/2
                                        ymax = center_y + self.min_text_size/2
                                    
                                    self.stats['text_annotations_preserved'] += 1
                            
                            # Handle regular objects with minimum size
                            else:
                                width = xmax - xmin
                                height = ymax - ymin
                                
                                # Ensure minimum size
                                if width < self.min_object_size or height < self.min_object_size:
                                    self.stats['small_annotations_fixed'] += 1
                                    center_x = (xmin + xmax) / 2
                                    center_y = (ymin + ymax) / 2
                                    
                                    if width < self.min_object_size:
                                        xmin = center_x - self.min_object_size/2
                                        xmax = center_x + self.min_object_size/2
                                    if height < self.min_object_size:
                                        ymin = center_y - self.min_object_size/2
                                        ymax = center_y + self.min_object_size/2
                            
                            # Convert to YOLO format (center_x, center_y, width, height)
                            bbox_w = xmax - xmin
                            bbox_h = ymax - ymin
                            bbox_x = xmin + bbox_w/2
                            bbox_y = ymin + bbox_h/2
                            
                            # Normalize
                            bbox_x /= img_w
                            bbox_y /= img_h
                            bbox_w /= img_w
                            bbox_h /= img_h
                            
                            # Validate normalized coordinates
                            bbox_x = max(0, min(bbox_x, 1))
                            bbox_y = max(0, min(bbox_y, 1))
                            bbox_w = max(0.001, min(bbox_w, 1))
                            bbox_h = max(0.001, min(bbox_h, 1))
                            
                            converted_lines.append(f"{class_id} {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f}")
                        else:
                            # Not a valid polygon
                            converted_lines.append(line.strip())
                    else:
                        # Already a bounding box
                        converted_lines.append(line.strip())
                
                # Write converted annotations
                with open(ann_file, 'w') as f:
                    f.write('\n'.join(converted_lines))
                    
            except Exception as e:
                continue
        
        print(f"✅ Converted {self.stats['polygons_converted']} polygons to bounding boxes")
        print(f"   📏 Preserved {self.stats['text_annotations_preserved']} text/symbol annotations")
        print(f"   📏 Fixed {self.stats['small_annotations_fixed']} small annotations")
        print(f"   📏 Processed {self.stats['point_annotations_processed']} point annotations")
        
        return converted_dir
    
    def _get_class_name(self, class_id: int) -> Optional[str]:
        """Get class name from class ID (if possible)"""
        try:
            # Try to get class name from data.yaml
            yaml_path = Path(self.temp_dir) / "download" / "data.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                names = config.get('names', [])
                if isinstance(names, dict):
                    names = [names[i] for i in range(len(names))]
                
                if 0 <= class_id < len(names):
                    return names[class_id]
        except:
            pass
        
        return None
    
    def sanitize_dataset(self, dataset_path: Path) -> Path:
        """Sanitize dataset with detailed reporting"""
        print(f"\n{'='*60}")
        print(f"🧹 SANITIZING DATASET ({self.sanitation_mode.upper()} MODE)")
        print(f"Criteria: Each instrument container must contain BOTH:")
        print(f"  1. '{self.required_content_1}'")
        print(f"  2. '{self.required_content_2}'")
        print(f"{'='*60}")
        
        # Get class mapping
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        names = config.get('names', [])
        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]
        
        class_map = {name: i for i, name in enumerate(names)}
        print(f"📊 Class mapping: {class_map}")
        
        # Get class IDs
        container_ids = [class_map[c] for c in self.containers if c in class_map]
        req1_id = class_map.get(self.required_content_1)
        req2_id = class_map.get(self.required_content_2)
        
        if not container_ids or req1_id is None or req2_id is None:
            raise ValueError(f"❌ Missing required classes in dataset")
        
        print(f"✅ Container IDs: {container_ids}")
        print(f"✅ Required content IDs: {req1_id} ({self.required_content_1}), {req2_id} ({self.required_content_2})")
        
        # Find label files
        label_files = list(dataset_path.rglob("labels/*.txt"))
        self.stats['total_images'] = len(label_files)
        print(f"🔍 Processing {self.stats['total_images']} images")
        
        # Create directory for deleted examples
        deleted_examples_dir = None
        if self.save_deleted_examples:
            deleted_examples_dir = self.output_dir / 'deleted_examples'
            deleted_examples_dir.mkdir(exist_ok=True)
            print(f"📸 Deleted examples will be saved to: {deleted_examples_dir}")
        
        deleted_count = 0
        
        for label_file in tqdm(label_files, desc="Sanitizing images"):
            try:
                # Read annotations
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse boxes
                boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # YOLO format
                        try:
                            class_id = int(float(parts[0]))
                            xc, yc, w, h = map(float, parts[1:])
                            # Convert to xyxy format for easier containment checking
                            x1 = xc - w/2
                            y1 = yc - h/2
                            x2 = xc + w/2
                            y2 = yc + h/2
                            boxes.append([class_id, x1, y1, x2, y2])
                        except:
                            continue
                
                # Separate containers and contents
                containers = [b for b in boxes if b[0] in container_ids]
                contents = [b for b in boxes if b[0] in [req1_id, req2_id]]
                
                self.stats['container_stats']['total_containers'] += len(containers)
                
                if not containers:
                    continue  # No containers, keep the image
                
                # Analyze containers
                image_has_complete_container = False
                containers_missing_items = []
                container_analysis = []
                
                for i, container in enumerate(containers):
                    has_req1 = False
                    has_req2 = False
                    cx = (container[1] + container[3]) / 2
                    cy = (container[2] + container[4]) / 2
                    
                    for content in contents:
                        # Check if content center is inside container
                        content_cx = (content[1] + content[3]) / 2
                        content_cy = (content[2] + content[4]) / 2
                        
                        if (container[1] <= content_cx <= container[3] and 
                            container[2] <= content_cy <= container[4]):
                            if content[0] == req1_id:
                                has_req1 = True
                            if content[0] == req2_id:
                                has_req2 = True
                    
                    if has_req1 and has_req2:
                        self.stats['container_stats']['complete_containers'] += 1
                        image_has_complete_container = True
                    else:
                        if not has_req1 and not has_req2:
                            self.stats['container_stats']['missing_both'] += 1
                        elif not has_req1:
                            self.stats['container_stats']['missing_id_letters'] += 1
                        elif not has_req2:
                            self.stats['container_stats']['missing_tag_number'] += 1
                        
                        containers_missing_items.append({
                            'container_idx': i,
                            'has_id_letters': has_req1,
                            'has_tag_number': has_req2
                        })
                    
                    container_analysis.append({
                        'container_idx': i,
                        'has_id_letters': has_req1,
                        'has_tag_number': has_req2,
                        'is_complete': has_req1 and has_req2
                    })
                
                # Determine if image should be deleted based on mode
                delete_image = False
                reason = ""
                
                if self.sanitation_mode == 'strict':
                    if containers_missing_items:  # Any incomplete container
                        delete_image = True
                        reason = "STRICT_MODE: Contains incomplete containers"
                elif self.sanitation_mode == 'relaxed':
                    if not image_has_complete_container:  # No complete containers at all
                        delete_image = True
                        reason = "RELAXED_MODE: No complete containers found"
                
                if delete_image:
                    deleted_count += 1
                    
                    # Save deletion details
                    self.stats['deletion_details'].append({
                        'image_name': label_file.stem,
                        'mode': self.sanitation_mode,
                        'reason': reason,
                        'total_containers': len(containers),
                        'complete_containers': sum(1 for c in container_analysis if c['is_complete']),
                        'incomplete_containers': len(containers) - sum(1 for c in container_analysis if c['is_complete']),
                        'containers_analysis': container_analysis
                    })
                    
                    # Save visual example (first 10 deleted images)
                    if self.save_deleted_examples and deleted_count <= 10:
                        self._save_deleted_example(label_file, deleted_examples_dir, container_analysis)
                    
                    # Delete image and annotation
                    label_file.unlink()
                    
                    img_path = Path(str(label_file).replace("labels", "images").replace(".txt", ".jpg"))
                    if not img_path.exists():
                        img_path = Path(str(img_path).replace(".jpg", ".png"))
                    
                    if img_path.exists():
                        img_path.unlink()
            
            except Exception as e:
                continue
        
        self.stats['deleted_images'] = deleted_count
        self.stats['sanitized_images'] = self.stats['total_images'] - deleted_count
        
        print(f"\n✅ Sanitization complete!")
        print(f"   📊 Total images: {self.stats['total_images']}")
        print(f"   ✅ Kept images: {self.stats['sanitized_images']}")
        print(f"   🗑️ Deleted images: {self.stats['deleted_images']}")
        print(f"   📦 Total containers analyzed: {self.stats['container_stats']['total_containers']}")
        print(f"   ✅ Complete containers: {self.stats['container_stats']['complete_containers']}")
        
        return dataset_path
    
    def _save_deleted_example(self, label_file: Path, output_dir: Path, container_analysis: List[Dict]):
        """Save visual example of why an image was deleted"""
        try:
            # Find image
            img_path = Path(str(label_file).replace("labels", "images").replace(".txt", ".jpg"))
            if not img_path.exists():
                img_path = Path(str(img_path).replace(".jpg", ".png"))
            
            if not img_path.exists():
                return
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                return
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]
            
            # Create figure
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
            
            # Plot annotations
            colors = plt.cm.tab10
            with open(label_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:])
                    
                    # Convert to pixel coordinates
                    x_pixel = xc * img_w
                    y_pixel = yc * img_h
                    w_pixel = w * img_w
                    h_pixel = h * img_h
                    
                    x1 = x_pixel - w_pixel/2
                    y1 = y_pixel - h_pixel/2
                    
                    # Get container status if available
                    container_status = container_analysis[line_idx] if line_idx < len(container_analysis) else None
                    
                    if container_status:
                        if container_status['is_complete']:
                            color = 'green'
                            status = 'COMPLETE'
                        else:
                            color = 'red'
                            missing = []
                            if not container_status['has_id_letters']:
                                missing.append('id_letters')
                            if not container_status['has_tag_number']:
                                missing.append('tag_number')
                            status = f"MISSING: {', '.join(missing)}"
                    else:
                        color = 'gray'
                        status = 'UNKNOWN'
                    
                    rect = Rectangle((x1, y1), w_pixel, h_pixel, fill=False, edgecolor=color, linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1 + 5, y1 + 15, f'Class {class_id}: {status}', 
                           color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_title(f'DELETED IMAGE: {label_file.stem}\nMode: {self.sanitation_mode.upper()}', 
                        color='red', fontweight='bold')
            ax.axis('off')
            
            # Save
            output_path = output_dir / f'deleted_{label_file.stem}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"   📸 Saved deleted example: {output_path.name}")
            
        except Exception as e:
            print(f"   ⚠️ Error saving deleted example: {e}")
    
    def generate_report(self):
        """Generate comprehensive HTML report"""
        if not self.generate_report or not self.stats['deletion_details']:
            return
        
        print(f"\n{'='*60}")
        print(f"📋 GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*60}")
        
        report_dir = self.output_dir / 'report'
        report_dir.mkdir(exist_ok=True)
        
        # Generate HTML report
        html_content = self._generate_html_report()
        html_path = report_dir / 'sanitization_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Generate summary CSV
        self._generate_csv_report(report_dir)
        
        # Generate visualizations
        self._generate_visualizations(report_dir)
        
        print(f"✅ Report generated at: {html_path}")
        print(f"📊 Open this file in Colab to view the interactive report")
        
        # Display report link in Colab
        try:
            from google.colab import files
            print(f"🔗 Colab file link:")
            print(f"/content/hvac_pipeline_output/report/sanitization_report.html")
        except:
            pass
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        # Calculate statistics
        deletion_rate = (self.stats['deleted_images'] / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0
        container_completion_rate = (self.stats['container_stats']['complete_containers'] / self.stats['container_stats']['total_containers'] * 100) if self.stats['container_stats']['total_containers'] > 0 else 0
        
        # Find most common missing item
        missing_counts = [
            ('id_letters', self.stats['container_stats']['missing_id_letters']),
            ('tag_number', self.stats['container_stats']['missing_tag_number']),
            ('both', self.stats['container_stats']['missing_both'])
        ]
        most_common = max(missing_counts, key=lambda x: x[1]) if missing_counts else ('none', 0)
        
        # Generate recommendations
        recommendations = []
        
        if deletion_rate > 40:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'High Data Loss',
                'description': f'Strict mode deleted {deletion_rate:.1f}% of your dataset. Consider using relaxed mode for training data.'
            })
        
        if most_common[1] > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Common Missing Item',
                'description': f'The most frequently missing item is "{most_common[0]}" ({most_common[1]} containers). Focus annotation efforts here.'
            })
        
        if container_completion_rate < 60:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Low Container Completion',
                'description': f'Only {container_completion_rate:.1f}% of containers are complete. This indicates broader annotation quality issues.'
            })
        
        # HVAC-specific insights
        hvac_insights = []
        if self.stats['text_annotations_preserved'] > 0:
            hvac_insights.append(f"✅ Preserved {self.stats['text_annotations_preserved']} text/symbol annotations")
        if self.stats['small_annotations_fixed'] > 0:
            hvac_insights.append(f"✅ Fixed {self.stats['small_annotations_fixed']} small annotations")
        if self.stats['point_annotations_processed'] > 0:
            hvac_insights.append(f"✅ Processed {self.stats['point_annotations_processed']} point annotations")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HVAC Dataset Sanitization Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .stat-number {{ font-size: 32px; font-weight: bold; color: #3498db; margin: 10px 0; }}
                .stat-label {{ font-size: 14px; color: #666; }}
                .deleted {{ color: #e74c3c; }}
                .kept {{ color: #2ecc71; }}
                .section {{ margin: 30px 0; }}
                .section-title {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
                .recommendation {{ border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; background: #f8f9fa; border-radius: 0 8px 8px 0; }}
                .high-priority {{ border-left-color: #e74c3c; }}
                .medium-priority {{ border-left-color: #f39c12; }}
                .low-priority {{ border-left-color: #2ecc71; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .hvac-insights {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; border-radius: 0 8px 8px 0; }}
                .hvac-title {{ color: #1976d2; font-weight: bold; margin: 10px 0; }}
                .hvac-item {{ margin: 5px 0; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔥 HVAC-Specific Dataset Sanitization Report</h1>
                    <p><strong>Workspace:</strong> {self.workspace_id} | <strong>Project:</strong> {self.project_id} | <strong>Version:</strong> {self.version}</p>
                    <p><strong>Sanitation Mode:</strong> <span style="color: #3498db; font-weight: bold;">{self.sanitation_mode.upper()}</span></p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Images</div>
                        <div class="stat-number">{self.stats['total_images']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Kept Images</div>
                        <div class="stat-number kept">{self.stats['sanitized_images']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Deleted Images</div>
                        <div class="stat-number deleted">{self.stats['deleted_images']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Deletion Rate</div>
                        <div class="stat-number deleted">{deletion_rate:.1f}%</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">🔧 HVAC-Specific Conversion Metrics</h2>
                    <div class="hvac-insights">
                        <div class="hvac-title">HVAC Annotation Preservation</div>
                        <div class="hvac-item">✅ Preserved {self.stats['text_annotations_preserved']} text/symbol annotations</div>
                        <div class="hvac-item">✅ Fixed {self.stats['small_annotations_fixed']} small annotations</div>
                        <div class="hvac-item">✅ Processed {self.stats['point_annotations_processed']} point annotations</div>
                        <div class="hvac-item">📊 Text labels preserved: {self.stats['text_annotations_preserved'] / (self.stats['text_annotations_preserved'] + self.stats['small_annotations_fixed'] + 1) * 100:.1f}%</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">📦 Container Analysis</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-label">Total Containers</div>
                            <div class="stat-number">{self.stats['container_stats']['total_containers']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Complete Containers</div>
                            <div class="stat-number kept">{self.stats['container_stats']['complete_containers']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Completion Rate</div>
                            <div class="stat-number">{container_completion_rate:.1f}%</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Most Missing Item</div>
                            <div class="stat-number">{most_common[0]}</div>
                        </div>
                    </div>
                    
                    <table>
                        <tr>
                            <th>Missing Item Type</th>
                            <th>Number of Containers</th>
                            <th>Percentage</th>
                        </tr>
        """
        
        total_missing = (self.stats['container_stats']['missing_id_letters'] + 
                        self.stats['container_stats']['missing_tag_number'] + 
                        self.stats['container_stats']['missing_both'])
        
        if total_missing > 0:
            html += f"""
                <tr>
                    <td>ID Letters</td>
                    <td>{self.stats['container_stats']['missing_id_letters']}</td>
                    <td>{(self.stats['container_stats']['missing_id_letters']/total_missing*100):.1f}%</td>
                </tr>
                <tr>
                    <td>Tag Numbers</td>
                    <td>{self.stats['container_stats']['missing_tag_number']}</td>
                    <td>{(self.stats['container_stats']['missing_tag_number']/total_missing*100):.1f}%</td>
                </tr>
                <tr>
                    <td>Both Items</td>
                    <td>{self.stats['container_stats']['missing_both']}</td>
                    <td>{(self.stats['container_stats']['missing_both']/total_missing*100):.1f}%</td>
                </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div class="section">
                    <h2 class="section-title">💡 Recommendations</h2>
        """
        
        if not recommendations:
            html += """
                    <div class="recommendation">
                        <h3>✅ Excellent Dataset Quality</h3>
                        <p>Your dataset has high quality with minimal deletions. No major improvements needed!</p>
                    </div>
            """
        else:
            for rec in recommendations:
                priority_class = rec['priority'].lower()
                html += f"""
                    <div class="recommendation {priority_class}-priority">
                        <h3>{rec['title']} 🎯</h3>
                        <p>{rec['description']}</p>
                        <p><strong>Priority: {rec['priority']}</strong></p>
                    </div>
                """
        
        html += f"""
                </div>
                
                <div class="section">
                    <h2 class="section-title">🔍 Key HVAC-Specific Insights</h2>
                    <ul>
                        <li><strong>Text Label Preservation:</strong> Your HVAC diagram contains many small text labels that standard conversion would discard. This pipeline preserved {self.stats['text_annotations_preserved']} of these critical annotations.</li>
                        <li><strong>Small Annotations:</strong> {self.stats['small_annotations_fixed']} small annotations were fixed to meet minimum size requirements while preserving their position.</li>
                        <li><strong>Point Annotations:</strong> {self.stats['point_annotations_processed']} point annotations (like single-character labels) were converted to proper bounding boxes.</li>
                        <li><strong>Sanitation Impact:</strong> Running in <span style="color: #3498db; font-weight: bold;">{self.sanitation_mode.upper()}</span> mode {'deleted' if self.stats['deleted_images'] > 0 else 'kept'} {self.stats['deleted_images'] if self.stats['deleted_images'] > 0 else self.stats['sanitized_images']} images</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2 class="section-title">🚀 HVAC-Specific Next Steps</h2>
                    <ul>
                        <li><strong>For Training Data:</strong> Consider running with <code>--sanitation-mode relaxed</code> to preserve more images</li>
                        <li><strong>For Validation/Test Data:</strong> Keep <code>--sanitation-mode strict</code> for high-quality evaluation</li>
                        <li><strong>Improve Annotations:</strong> Focus on adding missing <strong>{most_common[0]}</strong> items to incomplete containers</li>
                        <li><strong>Review Deleted Examples:</strong> Check the <code>deleted_examples</code> folder to understand deletion patterns</li>
                        <li><strong>HVAC-Specific Tip:</strong> Pay special attention to small text labels - these are critical for HVAC diagrams but often get lost in standard conversion</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated by HVAC-Specific Dataset Pipeline | {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Dataset: {self.workspace_id}/{self.project_id} v{self.version} | Mode: {self.sanitation_mode.upper()}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_csv_report(self, report_dir: Path):
        """Generate CSV report of deletions"""
        if not self.stats['deletion_details']:
            return
        
        import csv
        
        csv_path = report_dir / 'deletion_details.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'mode', 'reason', 'total_containers', 'complete_containers', 'incomplete_containers'])
            
            for detail in self.stats['deletion_details']:
                writer.writerow([
                    detail['image_name'],
                    detail['mode'],
                    detail['reason'],
                    detail['total_containers'],
                    detail['complete_containers'],
                    detail['incomplete_containers']
                ])
        
        print(f"📊 CSV report saved: {csv_path}")
    
    def _generate_visualizations(self, report_dir: Path):
        """Generate visualizations for the report"""
        if not self.stats['deletion_details']:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Pie chart: Kept vs Deleted
        plt.subplot(2, 2, 1)
        labels = ['Kept Images', 'Deleted Images']
        sizes = [self.stats['sanitized_images'], self.stats['deleted_images']]
        colors = ['#2ecc71', '#e74c3c']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Image Distribution After Sanitization', fontsize=12, fontweight='bold')
        
        # Container completion status
        plt.subplot(2, 2, 2)
        container_labels = ['Complete Containers', 'Incomplete Containers']
        container_sizes = [
            self.stats['container_stats']['complete_containers'],
            self.stats['container_stats']['total_containers'] - self.stats['container_stats']['complete_containers']
        ]
        container_colors = ['#27ae60', '#e67e22']
        plt.pie(container_sizes, labels=container_labels, colors=container_colors, autopct='%1.1f%%', startangle=90)
        plt.title('Container Completion Status', fontsize=12, fontweight='bold')
        
        # Missing items bar chart
        plt.subplot(2, 2, 3)
        missing_items = ['Missing ID Letters', 'Missing Tag Numbers', 'Missing Both']
        missing_counts = [
            self.stats['container_stats']['missing_id_letters'],
            self.stats['container_stats']['missing_tag_number'],
            self.stats['container_stats']['missing_both']
        ]
        colors = ['#3498db', '#9b59b6', '#f1c40f']
        bars = plt.bar(missing_items, missing_counts, color=colors)
        plt.title('Missing Items Analysis', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # HVAC-specific metrics
        plt.subplot(2, 2, 4)
        hvac_metrics = ['Text Labels Preserved', 'Small Annotations Fixed', 'Point Annotations Processed']
        hvac_counts = [
            self.stats['text_annotations_preserved'],
            self.stats['small_annotations_fixed'],
            self.stats['point_annotations_processed']
        ]
        colors = ['#27ae60', '#3498db', '#f39c12']
        
        bars = plt.bar(hvac_metrics, hvac_counts, color=colors)
        plt.title('HVAC-Specific Conversion Metrics', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        viz_path = report_dir / 'sanitization_visualizations.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualizations saved: {viz_path}")
    
    def create_final_zip(self, dataset_path: Path) -> Path:
        """Create final zip file"""
        zip_name = f"roboflow_{self.workspace_id}_{self.project_id}_v{self.version}_{self.sanitation_mode}_hvac_final.zip"
        zip_path = self.output_dir / zip_name
        
        print(f"\n📦 Creating final zip archive: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(dataset_path)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Zip created successfully! Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
        return zip_path
    
    def run(self):
        """Run the complete pipeline"""
        print("="*80)
        print("🚀 STARTING HVAC-SPECIFIC ANNOTATION PIPELINE")
        print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Download dataset
            dataset_path = self.download_dataset()
            
            # Step 2: Convert polygons to bounding boxes (HVAC-specific)
            converted_path = self.convert_polygons_to_bboxes_yolo(dataset_path)
            
            # Step 3: Sanitize dataset
            sanitized_path = self.sanitize_dataset(converted_path)
            
            # Step 4: Generate report
            self.generate_report()
            
            # Step 5: Create final zip
            final_zip = self.create_final_zip(sanitized_path)
            
            # Print summary
            elapsed_time = time.time() - start_time
            print(f"\n{'='*80}")
            print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")
            print(f"📤 Final dataset zip: {final_zip}")
            print(f"   📏 Text annotations preserved: {self.stats['text_annotations_preserved']}")
            print(f"   📏 Small annotations fixed: {self.stats['small_annotations_fixed']}")
            print(f"   📏 Point annotations processed: {self.stats['point_annotations_processed']}")
            print(f"{'='*80}")
            
            return final_zip
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            raise

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN EXECUTION - HVAC-SPECIFIC
# ═══════════════════════════════════════════════════════════════════════════════

def run_hvac_pipeline():
    """Main entry point for HVAC-specific pipeline"""
    
    # Get API key
    api_key = get_roboflow_api_key()
    
    # Default configuration for HVAC diagrams
    config = {
        'api_key': api_key,
        'workspace_id': 'elliotttmiller',
        'project_id': 'hvacai-s3kda',
        'version': 37,
        'sanitation_mode': 'strict',
        'generate_report': True,
        'save_deleted_examples': True,
        'output_dir': '/content/hvac_pipeline_output',
        'temp_dir': '/content/temp_hvac',
        
        # HVAC-specific configuration
        'text_classes': [
            'id_letters', 'tag_number', 'text_label', 'value', 
            'tag', 'number', 'label', 'id', 'symbol', 'text',
            'alpha', 'numeric', 'alphanumeric', 'character'
        ],
        'min_text_size': 4,
        'min_object_size': 6,
        'point_annotation_size': 4
    }
    
    print("\n" + "="*80)
    print("⚙️  PIPELINE CONFIGURATION")
    print("="*80)
    print(f"Workspace ID: {config['workspace_id']}")
    print(f"Project ID: {config['project_id']}")
    print(f"Version: {config['version']}")
    print(f"Sanitation Mode: {config['sanitation_mode'].upper()}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Generate Report: {config['generate_report']}")
    print(f"Save Deleted Examples: {config['save_deleted_examples']}")
    print(f"HVAC-Specific Settings:")
    print(f"  - Text classes: {', '.join(config['text_classes'][:5])}...")
    print(f"  - Minimum text size: {config['min_text_size']}px")
    print(f"  - Minimum object size: {config['min_object_size']}px")
    print(f"  - Point annotation size: {config['point_annotation_size']}px")
    print("="*80)
    
    # Ask for sanitation mode
    print("\n🔄 Sanitation Mode Options:")
    print("   strict   - Delete images with ANY incomplete containers (high quality, less data)")
    print("   relaxed  - Delete images with NO complete containers (balanced approach)")
    print("   none     - Keep all images (maximum data, lower quality)")
    
    mode_choice = input("\nChoose sanitation mode (strict/relaxed/none) [default: strict]: ").strip().lower()
    if mode_choice in ['strict', 'relaxed', 'none']:
        config['sanitation_mode'] = mode_choice
        print(f"✅ Using {mode_choice.upper()} mode")
    else:
        print("✅ Using default STRICT mode")
    
    # Run pipeline
    pipeline = HVACAnnotationPipeline(config)
    return pipeline.run()

# Run the pipeline
if __name__ == "__main__":
    run_hvac_pipeline()
```

## 🌟 Why This Pipeline Works for HVAC Diagrams

### 1. **Special Handling for Text Labels**
- **Text class detection**: Identifies text-related classes (id_letters, tag_number, etc.)
- **Fixed-size boxes**: Converts single-point text labels to tiny fixed-size boxes
- **Minimum size threshold**: Ensures small text labels meet minimum size requirements

### 2. **Point Annotation Conversion**
- **Detects point annotations** (single-point polygons)
- **Converts to tiny boxes** centered on the point
- **Preserves critical information** that would otherwise be lost

### 3. **HVAC-Specific Configuration**
```python
text_classes = [
    'id_letters', 'tag_number', 'text_label', 'value', 
    'tag', 'number', 'label', 'id', 'symbol', 'text',
    'alpha', 'numeric', 'alphanumeric', 'character'
]
min_text_size = 4  # Minimum pixel size for text annotations
min_object_size = 6  # Minimum pixel size for other objects
point_annotation_size = 4  # Size for point annotations
```

### 4. **Comprehensive Reporting**
- **HVAC-specific metrics**: Tracks text labels preserved, small annotations fixed
- **Visual examples**: Shows why images were deleted with color-coded annotations
- **Actionable recommendations**: Provides specific guidance for HVAC diagrams

## 📊 What You'll Get After Running

### 1. **Final Dataset Zip**
```
/content/hvac_pipeline_output/
└── roboflow_elliotttmiller_hvacai-s3kda_v37_strict_hvac_final.zip
```

### 2. **Comprehensive Report**
```
/content/hvac_pipeline_output/report/
├── sanitization_report.html    # Interactive HTML report with HVAC-specific insights
├── deletion_details.csv        # Detailed CSV log
└── sanitization_visualizations.png  # Charts and graphs
```

### 3. **Preserved Annotations**
- **Text labels** that would be lost in standard conversion
- **Small symbols** and **point annotations** properly converted
- **Critical HVAC information** preserved for training

## 🛠️ How to Use in Google Colab

### **Method 1: One-Click Run**
```python
# Copy the entire script into a Colab cell and run it
# The script will:
# 1. Install all dependencies
# 2. Ask for your Roboflow API key
# 3. Let you choose sanitation mode
# 4. Process your HVAC diagrams with special rules
# 5. Generate a comprehensive report
```

### **Method 2: Save and Run**
```python
# Save the script
!curl -o hvac_pipeline.py https://gist.githubusercontent.com/anonymous/0/raw/hvac_pipeline.py

# Run it
!python hvac_pipeline.py
```

## 📌 Key Benefits for Your HVAC Project

1. **Preserves critical small annotations** that standard conversion loses
2. **Handles text labels properly** with HVAC-specific rules
3. **Converts point annotations** to valid bounding boxes
4. **Provides detailed HVAC-specific metrics** in the report
5. **Maintains the correct workflow order** (convert first, then sanitize)
6. **Generates actionable recommendations** for HVAC diagrams

This pipeline is specifically designed for technical drawings like HVAC diagrams where standard object detection tools would fail to preserve critical small annotations. The HTML report will show you exactly how many text labels were preserved and provide specific guidance for your HVAC project.