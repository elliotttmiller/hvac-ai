"""
Lightweight YOLO inference wrapper used by the hvac-analysis service.

This file provides a small `YOLOInferenceEngine` class that calls an underlying
YOLO model (Ultralytics-style) and returns a JSON-serializable result dict.

Notes:
- The module intentionally uses minimal external dependencies here and keeps
  the RLE/mask conversion as a helper stub (the service can expand it if
  full RLE encoding is required).
"""

from typing import Any, Dict, Optional
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class YOLOInferenceEngine:
    """Wraps a loaded Ultralytics/YOLO model instance.

    The model passed into the constructor must implement a `.predict(...)`
    method that returns an object with `.boxes`, `.masks` (optional), and
    `.names` mapping (typical of Ultralytics return objects).
    """

    def __init__(self, model: Any):
        self.model = model

    def _polygon_to_rle(self, poly: Any, height: int, width: int) -> Optional[Dict[str, Any]]:
        """Convert polygon-like mask into a simple RLE-like placeholder.

        This is a lightweight placeholder. If the service needs proper COCO RLE
        encoding, replace this with pycocotools.mask.encode logic.
        """
        try:
            # If poly is already a list of polygons, we return a shim dict.
            return {"counts": [], "size": [height, width]}
        except Exception:
            return None

    def predict(self, image: np.ndarray, conf_threshold: float = 0.50) -> Dict[str, Any]:
        """Run inference and return a simple, serializable result dict.

        Args:
            image: HxWxC RGB/ BGR image as numpy.ndarray.
            conf_threshold: minimum confidence to keep a detection.

        Returns:
            A dict with keys: total_objects_found, counts_by_category, segments,
            processing_time_ms.
        """
        start_time = time.perf_counter()
        orig_h, orig_w = image.shape[:2]
        image_area = orig_h * orig_w

        logger.info("📥 [PROCESS] Received Image: %dx%d pixels", orig_w, orig_h)

        # Run inference: keep a lower internal conf to capture edge cases and
        # perform the final filtering below using conf_threshold.
        logger.info("🧠 [MODEL] Running YOLO inference (scan-conf=0.25, filter-conf=%s)", conf_threshold)
        results = self.model.predict(
            image,
            conf=0.25,
            iou=0.45,
            retina_masks=True,
            verbose=False,
        )[0]

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        detections = []
        class_counts = {}

        skipped_huge = 0
        skipped_low_conf = 0

        # Iterate over boxes in the Ultralytics results object.
        for i, box in enumerate(results.boxes):
            try:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                confidence = float(box.conf[0])

                # Extract xyxy coordinates; fall back if API differs.
                try:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = map(int, coords)
                except Exception:
                    # Fallback: treat box.xyxy as an array-like
                    xy = np.array(box.xyxy).astype(int).flatten()
                    x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])

                box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                if image_area > 0 and (box_area / image_area) > 0.60:
                    skipped_huge += 1
                    continue

                if confidence < conf_threshold:
                    skipped_low_conf += 1
                    continue

                rle_mask = None
                if getattr(results, "masks", None) is not None:
                    try:
                        poly = results.masks.xy[i]
                        rle_mask = self._polygon_to_rle(poly, orig_h, orig_w)
                    except Exception:
                        rle_mask = None

                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                detections.append({
                    "id": i,
                    "label": class_name,
                    "score": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "mask": rle_mask,
                })
            except Exception:
                logger.exception("Error processing detection %s", i)
                continue

        logger.info("🗑️ [FILTER] Removed %d 'Huge' boxes.", skipped_huge)
        logger.info("🗑️ [FILTER] Removed %d boxes below %s confidence.", skipped_low_conf, conf_threshold)
        logger.info("✅ [FINAL] Returning %d valid components.", len(detections))

        return {
            "total_objects_found": len(detections),
            "counts_by_category": class_counts,
            "segments": detections,
            "processing_time_ms": processing_time_ms,
        }