"""
ObjectDetector Service - Universal Object Detection Interface
Tool-agnostic wrapper for YOLO-based object detection.

Following Domain-Driven Design: Uses universal terminology (ObjectDetector)
rather than implementation-specific names (YOLOService).
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Universal object detection service.
    Currently implements YOLOv11 for HVAC component detection.
    
    Design Philosophy:
    - Generic interface that abstracts the underlying model
    - Easy to swap YOLO for EfficientDet, DETR, etc. without changing API
    - Returns standardized detection format
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.5
    ):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to the detection model weights
            device: Device to use ('cuda', 'cpu', or None for auto)
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.class_names = {}
        
        # Load model
        self._load_model()
        
        logger.info(f"✅ ObjectDetector initialized on {self.device}")
        logger.info(f"   Confidence threshold: {self.conf_threshold}")
        logger.info(f"   Model type: YOLOv11")
    
    def _load_model(self):
        """Load the detection model (internal implementation detail)."""
        try:
            logger.info(f"🚀 Loading detection model from: {self.model_path}")
            
            # Load YOLO model
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            
            # Extract class names
            self.class_names = self.model.names
            
            # Detect if model supports OBB (Oriented Bounding Boxes)
            self.supports_obb = hasattr(self.model, 'predict_obb') or 'obb' in str(self.model_path).lower()
            
            # Warm-up inference
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                if self.supports_obb and hasattr(self.model, 'predict_obb'):
                    self.model.predict_obb(dummy, verbose=False)
                else:
                    self.model.predict(dummy, verbose=False)
            except Exception:
                logger.debug("Warm-up inference failed (non-fatal)")
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Classes: {list(self.class_names.values())}")
            logger.info(f"   Supports OBB: {self.supports_obb}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load detection model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize ObjectDetector: {e}")
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            conf_threshold: Override default confidence threshold
            
        Returns:
            List of detections, each containing:
            - label: Class label
            - score: Confidence score
            - bbox: [x1, y1, x2, y2] (optional, for standard detections)
            - obb: {x_center, y_center, width, height, rotation} (optional, for OBB)
            - class_id: Numeric class ID
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=conf,
                verbose=False,
                device=self.device
            )
            
            detections = []
            
            for result in results:
                # Check if OBB results exist
                if hasattr(result, 'obb') and result.obb is not None:
                    # Process OBB detections
                    detections.extend(self._process_obb_results(result))
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    # Process standard bounding box detections
                    detections.extend(self._process_bbox_results(result))
            
            logger.debug(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return []
    
    def _process_obb_results(self, result) -> List[Dict[str, Any]]:
        """Process oriented bounding box results."""
        detections = []
        
        obb_data = result.obb
        if obb_data.xyxyxyxyn is not None and len(obb_data.xyxyxyxyn) > 0:
            # Get OBB parameters
            xywhr = obb_data.xywhr.cpu().numpy()  # [x_center, y_center, width, height, rotation]
            confs = obb_data.conf.cpu().numpy()
            classes = obb_data.cls.cpu().numpy().astype(int)
            
            for i in range(len(xywhr)):
                x_center, y_center, width, height, rotation = xywhr[i]
                
                detection = {
                    'label': self.class_names[classes[i]],
                    'score': float(confs[i]),
                    'class_id': int(classes[i]),
                    'obb': {
                        'x_center': float(x_center),
                        'y_center': float(y_center),
                        'width': float(width),
                        'height': float(height),
                        'rotation': float(rotation)
                    }
                }
                detections.append(detection)
        
        return detections
    
    def _process_bbox_results(self, result) -> List[Dict[str, Any]]:
        """Process standard bounding box results."""
        detections = []
        
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                
                detection = {
                    'label': self.class_names[classes[i]],
                    'score': float(confs[i]),
                    'class_id': int(classes[i]),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                }
                detections.append(detection)
        
        return detections
    
    def get_class_names(self) -> Dict[int, str]:
        """Get the mapping of class IDs to class names."""
        return self.class_names.copy()
