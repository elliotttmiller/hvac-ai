"""
YOLO11 Inference Module (Optimization Replacement for SAM)
Handles loading the fine-tuned HVAC model and processing images.
"""

import logging
import time
import numpy as np
import torch
import cv2
from typing import Dict, List, Any, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLOInferenceEngine:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        # Auto-detect device if not provided
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the YOLO11-Seg model."""
        try:
            logger.info(f"🚀 Loading YOLO11 model from: {self.model_path}")
            # Ultralytics handles loading logic automatically
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Log class names to verify correct model loading
            logger.info(f"✅ Model loaded. Classes: {self.model.names}")
            
            # Warmup run to initialize CUDA context
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_input, verbose=False)
            logger.info("🔥 Model warmed up and ready.")
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise RuntimeError(f"Could not initialize YOLO model.")

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: Numpy array (H, W, C) - RGB
            conf_threshold: Confidence threshold for detection
            iou_threshold: NMS threshold for overlap removal
            
        Returns:
            Dict containing formatted results for the frontend.
        """
        start_time = time.perf_counter()
        
        # Run Inference
        # retina_masks=True ensures high-quality polygon edges
        results = self.model.predict(
            image, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            retina_masks=True,
            verbose=False
        )[0] # We only process one image at a time

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse Results
        detections = []
        class_counts = {}

        # If no objects detected, return empty
        if not results.boxes:
            logger.info(f"No objects detected. Time: {processing_time_ms:.2f}ms")
            return {
                "total_objects_found": 0,
                "counts_by_category": {},
                "segments": [],
                "processing_time_ms": processing_time_ms
            }

        # Iterate through detections
        # YOLOv8/11 returns boxes and masks containers
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            
            # Bounding Box (x1, y1, x2, y2)
            # Convert to int list for JSON serialization
            bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            # Polygon Mask
            # results.masks.xy is a list of arrays (one per object) containing polygon points
            polygon = []
            if results.masks is not None:
                # Get the polygon points for this specific detection
                # Format: [[x1, y1], [x2, y2], ...]
                poly_points = results.masks.xy[i]
                # Flatten to simple list [x1, y1, x2, y2...] or keep as list of points depending on frontend needs
                # Here we keep as list of lists for clarity
                polygon = poly_points.tolist()

            # Update Counts
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Append to results
            detections.append({
                "id": i,
                "label": class_name,
                "score": confidence,
                "bbox": bbox,       # [x_min, y_min, x_max, y_max]
                "polygon": polygon, # Actual contour points of the object
            })

        logger.info(f"✅ Detected {len(detections)} objects. Time: {processing_time_ms:.2f}ms")

        return {
            "total_objects_found": len(detections),
            "counts_by_category": class_counts,
            "segments": detections, # Frontend expects 'segments' list
            "processing_time_ms": processing_time_ms
        }

def create_yolo_engine(model_path: str) -> YOLOInferenceEngine:
    return YOLOInferenceEngine(model_path=model_path)