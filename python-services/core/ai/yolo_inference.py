"""
YOLO11 Inference Module (Frontend Compatible)
Handles loading the fine-tuned HVAC model and converting outputs to SAM-style RLE.
"""

import logging
import time
import numpy as np
import torch
import cv2
from typing import Dict, List, Any, Optional
from ultralytics import YOLO
from pycocotools import mask as mask_utils

logger = logging.getLogger(__name__)

class YOLOInferenceEngine:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"🚀 Loading YOLO11 model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"✅ Model loaded. Classes: {self.model.names}")
            
            # Warmup
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_input, verbose=False)
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise RuntimeError(f"Could not initialize YOLO model.")

    def _polygon_to_rle(self, polygon: np.ndarray, height: int, width: int) -> Dict:
        """
        Converts a polygon point list into a COCO RLE dictionary.
        Format: {"size": [h, w], "counts": "string"}
        """
        if len(polygon) == 0:
            return None

        # Create a blank binary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        # Draw the polygon on the mask (filled)
        # polygon needs to be shape (N, 1, 2) for fillPoly, currently (N, 2)
        pts = polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

        # Encode to RLE (Fortran array required)
        rle = mask_utils.encode(np.asfortranarray(mask))
        
        # Decode bytes to string for JSON serialization
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        # Get original image dimensions for correct RLE sizing
        orig_h, orig_w = image.shape[:2]

        # Run Inference
        results = self.model.predict(
            image, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            retina_masks=True, # High quality masks
            verbose=False
        )[0]

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        detections = []
        class_counts = {}

        if not results.boxes:
            return {
                "total_objects_found": 0,
                "counts_by_category": {},
                "segments": [],
                "processing_time_ms": processing_time_ms
            }

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            # Generate RLE Mask
            rle_mask = None
            if results.masks is not None:
                # Get polygon points for this detection
                poly = results.masks.xy[i]
                # Convert to RLE
                rle_mask = self._polygon_to_rle(poly, orig_h, orig_w)

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            detections.append({
                "id": i,
                "label": class_name,
                "score": confidence,
                "bbox": bbox,
                "mask": rle_mask, # <--- THIS IS WHAT THE FRONTEND NEEDS
            })

        return {
            "total_objects_found": len(detections),
            "counts_by_category": class_counts,
            "segments": detections,
            "processing_time_ms": processing_time_ms
        }

def create_yolo_engine(model_path: str) -> YOLOInferenceEngine:
    return YOLOInferenceEngine(model_path=model_path)