"""
YOLO11 Object Detection Module (Bounding Box Detection)
Performs object detection with bounding boxes for HVAC components.
Includes text rejection filtering based on aspect ratio.
"""

import logging
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from ultralytics import YOLO

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
            
            # Detect whether this model supports oriented bounding boxes (OBB).
            # Newer/OBB-trained YOLO variants may expose a `predict_obb` API or
            # have 'obb' in the model path name. Prefer explicit method check.
            self.is_obb = hasattr(self.model, 'predict_obb') or ('obb' in str(self.model_path).lower())
            self.model_type = 'obb' if self.is_obb else 'standard'

            # Warm-up inference to prepare model. Use the appropriate method for
            # OBB models when available so the model is initialized on the right
            # codepaths (and moved to GPU if applicable).
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                if self.is_obb and hasattr(self.model, 'predict_obb'):
                    # Some implementations may require different signature
                    self.model.predict_obb(dummy_input, verbose=False)
                else:
                    self.model.predict(dummy_input, verbose=False)
            except Exception:
                # Do not fail startup for warm-up issues; log and continue
                logger.debug('Warm-up inference failed (non-fatal)', exc_info=True)

            logger.info(f"✅ Model loaded successfully. Type: {self.model_type} | Classes: {len(self.model.names)}")
            logger.info(f"📋 Available classes: {list(self.model.names.values())}")
        except FileNotFoundError:
            logger.error(f"❌ Model file not found at: {self.model_path}")
            raise RuntimeError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize YOLO model: {str(e)}")



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
        
        # Run model prediction. Some versions of the ultralytics model may
        # return an empty list or results object with `boxes` set to None
        # under certain failure modes—handle that defensively to avoid
        # raising TypeError: 'NoneType' object is not iterable when iterating
        # over `results.boxes` below.
        # Use OBB prediction API when available for oriented bounding box models.
        predict_kwargs = dict(conf=conf_threshold, iou=0.45, verbose=False)
        try:
            if getattr(self, 'is_obb', False) and hasattr(self.model, 'predict_obb'):
                raw_preds = self.model.predict_obb(image, **predict_kwargs)
            else:
                raw_preds = self.model.predict(image, **predict_kwargs)
        except Exception as e:
            logger.exception('Model prediction failed')
            raise

        if not raw_preds:
            logger.warning("⚠️ Model returned no predictions (empty raw_preds)")
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            if progress_callback:
                progress_callback({"type": "progress", "message": "No detections", "percent": 100})
            return {
                "total_objects_found": 0,
                "counts_by_category": {},
                "segments": [],
                "processing_time_ms": processing_time_ms,
            }

        results = raw_preds[0]

        # Defensive check: results.boxes may be None in some cases. Some models
        # (OBB variants) place detections under `results.obb`. Prefer boxes,
        # fall back to obb, and if neither exist return an empty result.
        boxes_iterable = getattr(results, "boxes", None)
        is_obb_mode = False
        if boxes_iterable is None or (hasattr(boxes_iterable, '__len__') and len(boxes_iterable) == 0):
            obb_iterable = getattr(results, "obb", None)
            if obb_iterable is not None and (not hasattr(obb_iterable, '__len__') or len(obb_iterable) > 0):
                boxes_iterable = obb_iterable
                is_obb_mode = True

        if boxes_iterable is None or (hasattr(boxes_iterable, '__len__') and len(boxes_iterable) == 0):
            logger.warning("⚠️ Prediction result has no 'boxes' or 'obb' (None or empty). Returning empty detections.")
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            if progress_callback:
                progress_callback({"type": "progress", "message": "No detections", "percent": 100})
            return {
                "total_objects_found": 0,
                "counts_by_category": {},
                "segments": [],
                "processing_time_ms": processing_time_ms,
            }

        # Send inference complete progress
        if progress_callback:
            progress_callback({"type": "status", "message": "Processing detections...", "percent": 60})

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        detections = []
        class_counts = {}
        
        skipped_huge = 0
        skipped_text = 0 # New counter for text rejection
        skipped_conf = 0

        # Use the boxes iterable we validated above
        for i, box in enumerate(boxes_iterable):
            # Class id and score extraction should work for both standard and
            # OBB results; be defensive in case underlying tensor shapes differ.
            try:
                cls_id = int(box.cls[0])
            except Exception:
                # Fallback: some outputs may store cls as a scalar
                try:
                    cls_id = int(box.cls)
                except Exception:
                    cls_id = 0

            class_name = self.model.names.get(cls_id, str(cls_id)) if hasattr(self.model, 'names') else str(cls_id)

            try:
                confidence = float(box.conf[0])
            except Exception:
                try:
                    confidence = float(box.conf)
                except Exception:
                    confidence = 0.0

            # Extract bbox coordinates; OBB boxes may not expose `xyxy` in the
            # same way as standard boxes. Try common access patterns and fall
            # back to a zero-box to avoid crashes.
            try:
                raw_xy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = map(int, raw_xy.tolist())
            except Exception:
                try:
                    # Some OBB outputs may expose `.xyxy` directly or as a list
                    raw_xy = getattr(box, 'xyxy')
                    if hasattr(raw_xy, '__len__'):
                        arr = raw_xy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = map(int, arr.tolist())
                    else:
                        x1 = y1 = x2 = y2 = 0
                except Exception:
                    x1 = y1 = x2 = y2 = 0
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

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            detections.append({
                "id": i,
                "label": class_name,
                "score": confidence,
                "bbox": bbox,
                "obb_mode": bool(is_obb_mode),
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