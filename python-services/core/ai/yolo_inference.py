"""
YOLO11 Inference Module (Text Rejection Tuned)
Adds aspect-ratio filtering to remove text labels detected as valves.
"""

import logging
import time
import numpy as np
import torch
import cv2
from typing import Dict, List, Any, Optional
from ultralytics import YOLO

try:
    from pycocotools import mask as mask_utils
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

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
            logger.info(f"🎯 Using device: {self.device}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Warm-up inference to prepare model
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_input, verbose=False)
            
            logger.info(f"✅ Model loaded successfully. Classes: {len(self.model.names)}")
            logger.info(f"📋 Available classes: {list(self.model.names.values())}")
        except FileNotFoundError:
            logger.error(f"❌ Model file not found at: {self.model_path}")
            raise RuntimeError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize YOLO model: {str(e)}")

    def _polygon_to_rle(self, polygon: np.ndarray, height: int, width: int) -> Optional[Dict]:
        """Convert polygon mask to RLE format for efficient storage.
        
        Args:
            polygon: Numpy array of polygon points
            height: Image height
            width: Image width
            
        Returns:
            RLE encoded mask dictionary or None if conversion fails
        """
        if not HAS_COCO or len(polygon) == 0:
            return None
        
        try:
            mask = np.zeros((height, width), dtype=np.uint8)
            pts = polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle
        except Exception as e:
            logger.warning(f"Failed to convert polygon to RLE: {e}")
            return None

    def predict(self, image: np.ndarray, conf_threshold: float = 0.50, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Perform inference on an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            conf_threshold: Confidence threshold for filtering detections (0.0-1.0)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing detection results
            
        Raises:
            ValueError: If input parameters are invalid
        """
        # Input validation
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Image must be a valid numpy array")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be in (H, W, 3) format, got shape: {image.shape}")
        
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got: {conf_threshold}")
        
        start_time = time.perf_counter()
        orig_h, orig_w = image.shape[:2]
        image_area = orig_h * orig_w
        
        logger.info(f"📥 [PROCESS] Image: {orig_w}x{orig_h} | Conf Thresh: {conf_threshold}")
        
        # Send initial progress if callback provided
        if progress_callback:
            progress_callback({"type": "status", "message": "Starting inference...", "percent": 10})
        
        results = self.model.predict(
            image, 
            conf=0.25, 
            iou=0.45, 
            retina_masks=True,
            verbose=False
        )[0]

        # Send inference complete progress
        if progress_callback:
            progress_callback({"type": "status", "message": "Processing detections...", "percent": 60})

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        detections = []
        class_counts = {}
        
        skipped_huge = 0
        skipped_text = 0 # New counter for text rejection
        skipped_conf = 0

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w = x2 - x1
            h = y2 - y1
            box_area = w * h
            
            # --- FILTER 1: CONFIDENCE ---
            if confidence < conf_threshold:
                skipped_conf += 1
                continue

            # --- FILTER 2: SIZE (Anti-Hallucination) ---
            if (box_area / image_area) > 0.10: 
                skipped_huge += 1
                continue 

            # --- FILTER 3: ASPECT RATIO (Anti-Text) ---
            # Text blocks are usually wide and short.
            # Valves are usually square (1:1) or slightly rectangular (up to 2.5:1).
            # If Width is > 3x Height, it's almost certainly text.
            aspect_ratio = w / float(h)
            if aspect_ratio > 3.0: 
                skipped_text += 1
                continue

            # --- DATA EXTRACTION ---
            bbox = [x1, y1, x2, y2]
            polygon_list = []
            rle_mask = None
            
            # Extract mask data if available
            if results.masks is not None and i < len(results.masks.xy):
                try:
                    raw_poly = results.masks.xy[i]
                    polygon_list = raw_poly.tolist()
                    # Only create RLE if polygon is valid
                    if len(polygon_list) > 0:
                        rle_mask = self._polygon_to_rle(raw_poly, orig_h, orig_w)
                except Exception as e:
                    logger.warning(f"Failed to extract mask for detection {i}: {e}")

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            detections.append({
                "id": i,
                "label": class_name,
                "score": confidence,
                "bbox": bbox,
                "polygon": polygon_list,
                "mask": rle_mask,
            })

        logger.info(f"🗑️ [FILTERS] Removed: {skipped_huge} Huge | {skipped_text} Text/Wide | {skipped_conf} Low Conf")
        logger.info(f"✅ [FINAL] Returning {len(detections)} valid components.")

        # Send final progress if callback provided
        if progress_callback:
            progress_callback({"type": "progress", "message": "Analysis complete", "percent": 90})

        return {
            "total_objects_found": len(detections),
            "counts_by_category": class_counts,
            "segments": detections,
            "processing_time_ms": processing_time_ms
        }

def create_yolo_engine(model_path: str) -> YOLOInferenceEngine:
    return YOLOInferenceEngine(model_path=model_path)