"""
Lightweight YOLO inference wrapper used by the hvac-analysis service.

This file provides a optimized `YOLOInferenceEngine` class that calls an underlying
YOLO model (Ultralytics-style) and returns a JSON-serializable result dict.

"""

from typing import Any, Dict, Optional, Callable, List, Union
import time
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

class YOLOInferenceEngine:
    """Wraps a loaded Ultralytics/YOLO model instance.

    The model passed into the constructor must implement a `.predict(...)`
    method that returns an object with `.boxes`, `.masks` (optional), and
    `.names` mapping (typical of Ultralytics return objects).
    """

    def __init__(self, model: Any):
        self.model = model

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        scan_conf_threshold: float = 0.25,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Run inference and return a simple, serializable result dict.

        Args:
            image: HxWxC RGB/BGR image as numpy.ndarray.
            conf_threshold: Minimum confidence to keep a detection in final output.
            iou_threshold: IOU threshold for NMS during inference.
            scan_conf_threshold: Lower confidence threshold for initial scanning (allows filtering later).
            progress_callback: Optional callback for progress updates.

        Returns:
            A dict with keys: total_objects_found, counts_by_category, segments,
            processing_time_ms.
        """
        start_time = time.perf_counter()
        orig_h, orig_w = image.shape[:2]
        image_area = orig_h * orig_w

        logger.info("📥 [PROCESS] Received Image: %dx%d pixels", orig_w, orig_h)
        if progress_callback:
            try:
                progress_callback({"type": "status", "message": "received", "width": orig_w, "height": orig_h})
            except Exception:
                logger.debug("progress_callback failed at start", exc_info=True)

        # Run inference
        logger.info("🧠 [MODEL] Running YOLO inference (scan-conf=%s, filter-conf=%s)", scan_conf_threshold, conf_threshold)
        try:
            results = self.model.predict(
                image,
                conf=scan_conf_threshold,
                iou=iou_threshold,
                retina_masks=True,
                verbose=False,
            )[0]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"YOLO inference failed: {e}") from e

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        detections: List[Dict[str, Any]] = []
        class_counts: Dict[str, int] = {}

        skipped_huge = 0
        skipped_low_conf = 0

        # Check if results are valid
        if not hasattr(results, 'boxes'):
             logger.warning("No boxes attribute in results.")
             return {
                "total_objects_found": 0,
                "counts_by_category": {},
                "segments": [],
                "processing_time_ms": processing_time_ms,
            }

        total_boxes = len(results.boxes)
        
        # Optimize: Extract all data at once if possible, but iterating is safer for mixed processing logic
        # For very large numbers of boxes, vectorized operations would be better, but loop is fine for typical HVAC diagrams.

        for i, box in enumerate(results.boxes):
            try:
                # 1. Class and Confidence
                cls_id = int(box.cls[0].item())
                class_name = self.model.names[cls_id]
                confidence = float(box.conf[0].item())

                # 2. Bounding Box (xyxy)
                # Ensure integer coordinates
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                # 3. Filters
                # Size Filter
                box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                if image_area > 0 and (box_area / image_area) > 0.60:
                    skipped_huge += 1
                    continue

                # Confidence Filter
                if confidence < conf_threshold:
                    skipped_low_conf += 1
                    continue

                # 4. Polygon Extraction
                polygon_coords: Optional[List[List[int]]] = None
                if getattr(results, "masks", None) is not None:
                    # results.masks.xy is a list of np arrays (normalized 0-1 if retina_masks=False, but pixel coords if True?)
                    # Ultralytics docs say .xy returns pixel coordinates.
                    try:
                        # Accessing .xy[i] gives pixel coordinates directly
                        poly = results.masks.xy[i]
                        
                        # Convert to list of [x, y] integers
                        if len(poly) > 0:
                             polygon_coords = poly.astype(int).tolist()
                    except Exception as e:
                        logger.warning(f"Failed to extract polygon for box {i}: {e}")
                        polygon_coords = None

                # 5. Aggregate
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                detections.append({
                    "id": i,
                    "label": class_name,
                    "score": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "polygon": polygon_coords,
                })

                # 6. Progress Update (Throttled)
                if progress_callback and i % 5 == 0: # Update every 5 items to reduce overhead
                    try:
                        percent = int(((i + 1) / max(1, total_boxes)) * 80)
                        progress_callback({
                            "type": "progress",
                            "index": i,
                            "label": class_name,
                            "score": confidence,
                            "percent": percent,
                        })
                    except Exception:
                        pass # Suppress callback errors during loop

            except Exception as e:
                logger.exception(f"Error processing detection {i}: {e}")
                continue

        logger.info("🗑️ [FILTER] Removed %d 'Huge' boxes.", skipped_huge)
        logger.info("🗑️ [FILTER] Removed %d boxes below %s confidence.", skipped_low_conf, conf_threshold)
        logger.info("✅ [FINAL] Returning %d valid components.", len(detections))

        result = {
            "total_objects_found": len(detections),
            "counts_by_category": class_counts,
            "segments": detections,
            "processing_time_ms": processing_time_ms,
        }

        # Final progress/result notification
        if progress_callback:
            try:
                progress_callback({"type": "result", "result": result})
            except Exception:
                logger.debug("progress_callback failed at end", exc_info=True)

        return result


def create_yolo_engine(model_path: Optional[str] = None, device: Optional[str] = None) -> YOLOInferenceEngine:
    """Factory helper to create and return a YOLOInferenceEngine instance.

    Args:
        model_path: Path to the Ultralytics YOLO model weights (required).
        device: Optional device string like 'cpu' or 'cuda:0'. If None the
            underlying library decides.

    Returns:
        YOLOInferenceEngine wrapping the loaded model.
    """
    try:
        # Import lazily so module import doesn't fail if ultralytics isn't installed
        from ultralytics import YOLO as _YOLO
    except ImportError as e:
        logger.error("ultralytics package not available: %s", e)
        raise

    if not model_path:
        raise ValueError("model_path is required to create YOLO engine")

    logger.info("Loading YOLO model from %s", model_path)
    try:
        model = _YOLO(model_path)
        # If device selection was provided, attempt to set it (Ultralytics handles this via .to)
        if device:
            model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model or set device: {e}")
        raise

    return YOLOInferenceEngine(model)