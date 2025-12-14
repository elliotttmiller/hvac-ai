import json
import shutil
import os
import zipfile
from pathlib import Path
from pycocotools import mask as mask_utils

# --- CONFIGURATION ---
# Path to the zip you just downloaded from Roboflow
INPUT_ZIP_PATH = "/content/drive/MyDrive/hvac-dataset.zip" 

# Where to save the fixed version (We will point the training notebook here)
OUTPUT_DIR = "/content/hvac_dataset_ready"

# --- EXECUTION ---
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print(f"📂 Unzipping {INPUT_ZIP_PATH}...")
with zipfile.ZipFile(INPUT_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)

print("🔧 Fixing missing polygons (converting BBoxes to Polygons)...")
target_dir = Path(OUTPUT_DIR)
fixed_total = 0

for ann_file in target_dir.rglob('*_annotations.coco.json'):
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    file_fixed_count = 0
    img_map = {img['id']: img for img in data['images']}
    
    for ann in data['annotations']:
        # Check if segmentation is missing or empty
        has_seg = False
        if 'segmentation' in ann and ann['segmentation']:
            if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                has_seg = True
        
        # If no segmentation, creates a square polygon from the bbox
        if not has_seg and 'bbox' in ann:
            x, y, w, h = ann['bbox']
            
            # Clamp to image size to prevent errors
            img = img_map.get(ann['image_id'])
            if img:
                img_h, img_w = img['height'], img['width']
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)
            
            if w > 0 and h > 0:
                # TopLeft, TopRight, BottomRight, BottomLeft
                poly = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                ann['segmentation'] = poly
                ann['area'] = w * h
                ann['iscrowd'] = 0
                file_fixed_count += 1
                fixed_total += 1

    with open(ann_file, 'w') as f:
        json.dump(data, f)
    print(f"   - {ann_file.name}: Converted {file_fixed_count} boxes to polygons.")

print(f"\n✅ SUCCESS! Total annotations fixed: {fixed_total}")
print(f"📂 Your training data is ready at: {OUTPUT_DIR}")