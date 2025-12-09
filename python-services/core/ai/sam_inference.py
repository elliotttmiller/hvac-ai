"""
SAM Model Inference Module
Segment Anything Model for P&ID and HVAC diagram analysis
"""

import os
import logging
import base64
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import cv2
import torch
from pycocotools import mask as mask_utils

logger = logging.getLogger(__name__)

# Complete taxonomy of HVAC/P&ID components (70 classes)
HVAC_TAXONOMY = [
    # Valves & Actuators
    "Actuator-Diaphragm",
    "Actuator-Generic",
    "Actuator-Manual",
    "Actuator-Motorized",
    "Actuator-Piston",
    "Actuator-Pneumatic",
    "Actuator-Solenoid",
    "Valve-3Way",
    "Valve-4Way",
    "Valve-Angle",
    "Valve-Ball",
    "Valve-Butterfly",
    "Valve-Check",
    "Valve-Control",
    "Valve-Diaphragm",
    "Valve-Gate",
    "Valve-Generic",
    "Valve-Globe",
    "Valve-Needle",
    "Valve-Plug",
    "Valve-Relief",
    # Equipment
    "Equipment-AgitatorMixer",
    "Equipment-Compressor",
    "Equipment-FanBlower",
    "Equipment-Generic",
    "Equipment-HeatExchanger",
    "Equipment-Motor",
    "Equipment-Pump-Centrifugal",
    "Equipment-Pump-Dosing",
    "Equipment-Pump-Generic",
    "Equipment-Pump-Screw",
    "Equipment-Vessel",
    # Instrumentation & Controls
    "Component-DiaphragmSeal",
    "Component-Switch",
    "Controller-DCS",
    "Controller-Generic",
    "Controller-PLC",
    "Instrument-Analyzer",
    "Instrument-Flow-Indicator",
    "Instrument-Flow-Transmitter",
    "Instrument-Generic",
    "Instrument-Level-Indicator",
    "Instrument-Level-Switch",
    "Instrument-Level-Transmitter",
    "Instrument-Pressure-Indicator",
    "Instrument-Pressure-Switch",
    "Instrument-Pressure-Transmitter",
    "Instrument-Temperature",
    # Piping, Ductwork & In-line Components
    "Accessory-Drain",
    "Accessory-Generic",
    "Accessory-SightGlass",
    "Accessory-Vent",
    "Damper",
    "Duct",
    "Filter",
    "Fitting-Bend",
    "Fitting-Blind",
    "Fitting-Flange",
    "Fitting-Generic",
    "Fitting-Reducer",
    "Pipe-Insulated",
    "Pipe-Jacketed",
    "Strainer-Basket",
    "Strainer-Generic",
    "Strainer-YType",
    "Trap",
]


@dataclass
class SegmentResult:
    """Result from SAM segmentation"""
    label: str
    score: float
    mask: str  # Base64 encoded RLE
    bbox: List[int]  # [x, y, width, height]


@dataclass
class CountResult:
    """Result from automated counting"""
    total_objects_found: int
    counts_by_category: Dict[str, int]


class SAMInferenceEngine:
    """
    SAM Model Inference Engine for HVAC/P&ID Analysis
    
    Provides interactive segmentation and automated component counting
    using the fine-tuned Segment Anything Model.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize SAM inference engine
        
        Args:
            model_path: Path to fine-tuned SAM model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path or os.getenv('SAM_MODEL_PATH', 'models/sam_hvac_finetuned.pth')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        self.transform = None
        
        logger.info(f"Initializing SAM Inference Engine on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load SAM model into GPU memory at startup"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}. Using mock mode.")
                self.model = "mock_model"
                return
            
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything.utils.transforms import ResizeLongestSide
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize SAM model (ViT-B by default)
            model_type = checkpoint.get('model_type', 'vit_b')
            sam = sam_model_registry[model_type]()
            
            # Load weights
            sam.load_state_dict(checkpoint['model_state_dict'])
            sam.to(device=self.device)
            sam.eval()
            
            self.model = sam
            self.image_encoder = sam.image_encoder
            self.prompt_encoder = sam.prompt_encoder
            self.mask_decoder = sam.mask_decoder
            self.transform = ResizeLongestSide(1024)
            
            logger.info("SAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            logger.warning("Using mock mode for development")
            self.model = "mock_model"
    
    def segment(self, image: np.ndarray, prompt: Dict[str, Any]) -> List[SegmentResult]:
        """
        Perform interactive segmentation based on user prompt
        
        Args:
            image: Input image as numpy array (H, W, 3)
            prompt: Prompt dictionary with type and data
                   Example: {"type": "point", "data": {"coords": [x, y], "label": 1}}
        
        Returns:
            List of segment results with masks and labels
        """
        if self.model == "mock_model":
            return self._mock_segment(image, prompt)
        
        try:
            # Parse prompt
            prompt_type = prompt.get('type')
            prompt_data = prompt.get('data', {})
            
            # Pre-process image
            original_size = image.shape[:2]
            
            # Encode image once
            with torch.no_grad():
                # Transform and prepare image
                input_image = self.transform.apply_image(image)
                input_tensor = torch.as_tensor(input_image, device=self.device)
                input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
                
                # Normalize
                pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
                pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
                input_tensor = (input_tensor - pixel_mean) / pixel_std
                
                # Encode image
                image_embedding = self.image_encoder(input_tensor)
                
                # Prepare prompt
                if prompt_type == "point":
                    coords = np.array([prompt_data['coords']])
                    labels = np.array([prompt_data.get('label', 1)])
                    
                    # Transform coordinates
                    coords = self.transform.apply_coords(coords, original_size)
                    
                    # Convert to tensors
                    point_coords = torch.as_tensor(coords, dtype=torch.float, device=self.device)[None, :, :]
                    point_labels = torch.as_tensor(labels, dtype=torch.int, device=self.device)[None, :]
                    
                    # Encode prompt
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=(point_coords, point_labels),
                        boxes=None,
                        masks=None
                    )
                else:
                    raise ValueError(f"Unsupported prompt type: {prompt_type}")
                
                # Decode mask
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True
                )
                
                # Upscale mask to original size
                masks = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False
                )
                
                masks = masks.squeeze().cpu().numpy()
                scores = iou_predictions.squeeze().cpu().numpy()
            
            # Post-process and create results
            results = []
            for mask, score in zip(masks, scores):
                # Convert mask to binary
                binary_mask = (mask > 0.0).astype(np.uint8)
                
                # Classify the segmented region (mock classification for now)
                label = self._classify_segment(image, binary_mask)
                
                # Convert to RLE and encode
                rle = self._mask_to_rle(binary_mask)
                rle_base64 = base64.b64encode(rle.encode()).decode('utf-8')
                
                # Calculate bounding box
                bbox = self._mask_to_bbox(binary_mask)
                
                results.append(SegmentResult(
                    label=label,
                    score=float(score),
                    mask=rle_base64,
                    bbox=bbox
                ))
            
            # Return best result
            return [max(results, key=lambda x: x.score)] if results else []
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise
    
    def count(self, image: np.ndarray, grid_size: int = 32, confidence_threshold: float = 0.85,
              nms_iou_threshold: float = 0.9) -> CountResult:
        """
        Perform automated component counting on entire diagram
        
        Args:
            image: Input image as numpy array (H, W, 3)
            grid_size: Grid spacing for point prompts (pixels)
            confidence_threshold: Minimum confidence score to keep
            nms_iou_threshold: IoU threshold for non-maximum suppression
        
        Returns:
            Count result with total objects and breakdown by category
        """
        if self.model == "mock_model":
            return self._mock_count(image)
        
        try:
            h, w = image.shape[:2]
            original_size = (h, w)
            
            # Encode image once
            with torch.no_grad():
                # Transform and prepare image
                input_image = self.transform.apply_image(image)
                input_tensor = torch.as_tensor(input_image, device=self.device)
                input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
                
                # Normalize
                pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
                pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
                input_tensor = (input_tensor - pixel_mean) / pixel_std
                
                # Encode image once and reuse
                image_embedding = self.image_encoder(input_tensor)
            
            # Generate grid of point prompts
            all_detections = []
            
            for y in range(grid_size // 2, h, grid_size):
                for x in range(grid_size // 2, w, grid_size):
                    with torch.no_grad():
                        # Create point prompt
                        coords = np.array([[x, y]])
                        labels = np.array([1])
                        
                        # Transform coordinates
                        coords_transformed = self.transform.apply_coords(coords, original_size)
                        
                        # Convert to tensors
                        point_coords = torch.as_tensor(coords_transformed, dtype=torch.float, device=self.device)[None, :, :]
                        point_labels = torch.as_tensor(labels, dtype=torch.int, device=self.device)[None, :]
                        
                        # Encode prompt
                        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                            points=(point_coords, point_labels),
                            boxes=None,
                            masks=None
                        )
                        
                        # Decode mask
                        low_res_masks, iou_predictions = self.mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=self.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False
                        )
                        
                        # Get score
                        score = float(iou_predictions.squeeze().cpu().numpy())
                        
                        # Only keep high confidence detections
                        if score >= confidence_threshold:
                            # Upscale mask
                            mask = torch.nn.functional.interpolate(
                                low_res_masks,
                                size=original_size,
                                mode="bilinear",
                                align_corners=False
                            )
                            mask = (mask.squeeze().cpu().numpy() > 0.0).astype(np.uint8)
                            
                            # Classify
                            label = self._classify_segment(image, mask)
                            
                            # Calculate bbox
                            bbox = self._mask_to_bbox(mask)
                            
                            all_detections.append({
                                'label': label,
                                'score': score,
                                'mask': mask,
                                'bbox': bbox
                            })
            
            # Apply Non-Maximum Suppression to remove duplicates
            unique_detections = self._apply_nms(all_detections, nms_iou_threshold)
            
            # Count by category
            counts = {}
            for det in unique_detections:
                label = det['label']
                counts[label] = counts.get(label, 0) + 1
            
            return CountResult(
                total_objects_found=len(unique_detections),
                counts_by_category=counts
            )
            
        except Exception as e:
            logger.error(f"Counting failed: {e}")
            raise
    
    def _classify_segment(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Classify a segmented region
        
        Args:
            image: Original image
            mask: Binary mask
        
        Returns:
            Component label from HVAC_TAXONOMY
        """
        # Mock classification - in production, use a trained classifier
        # This would analyze the masked region features to determine component type
        import random
        return random.choice(HVAC_TAXONOMY)
    
    def _mask_to_rle(self, mask: np.ndarray) -> str:
        """
        Convert binary mask to RLE string
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            RLE encoded string
        """
        # Fortran order required by COCO
        rle = mask_utils.encode(np.asfortranarray(mask))
        # Convert bytes to string if needed
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        return f"{rle['size'][0]}x{rle['size'][1]}:{rle['counts']}"
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """
        Calculate bounding box from binary mask
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            Bounding box [x, y, width, height]
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Filtered list of unique detections
        """
        if not detections:
            return []
        
        # Sort by score (descending)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        
        for det in detections:
            # Check if this detection overlaps significantly with any kept detection
            should_keep = True
            for kept_det in keep:
                iou = self._calculate_iou(det['mask'], kept_det['mask'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det)
        
        return keep
    
    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two masks
        
        Args:
            mask1: Binary mask 1
            mask2: Binary mask 2
        
        Returns:
            IoU score
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def _mock_segment(self, image: np.ndarray, prompt: Dict[str, Any]) -> List[SegmentResult]:
        """Mock segmentation for development/testing"""
        h, w = image.shape[:2]
        
        # Create a mock circular mask around the clicked point
        if prompt.get('type') == 'point':
            coords = prompt['data']['coords']
            x, y = coords
            
            # Create circular mask
            mask = np.zeros((h, w), dtype=np.uint8)
            radius = 30
            cv2.circle(mask, (x, y), radius, 1, -1)
            
            # Convert to RLE
            rle = self._mask_to_rle(mask)
            rle_base64 = base64.b64encode(rle.encode()).decode('utf-8')
            
            # Calculate bbox
            bbox = [max(0, x - radius), max(0, y - radius), 2 * radius, 2 * radius]
            
            return [SegmentResult(
                label="Valve-Ball",
                score=0.967,
                mask=rle_base64,
                bbox=bbox
            )]
        
        return []
    
    def _mock_count(self, image: np.ndarray) -> CountResult:
        """Mock counting for development/testing"""
        return CountResult(
            total_objects_found=87,
            counts_by_category={
                "Valve-Ball": 23,
                "Valve-Gate": 12,
                "Fitting-Bend": 31,
                "Equipment-Pump-Centrifugal": 2,
                "Instrument-Pressure-Indicator": 19
            }
        )


def create_sam_engine(model_path: Optional[str] = None, device: Optional[str] = None) -> SAMInferenceEngine:
    """Factory function to create SAM inference engine"""
    return SAMInferenceEngine(model_path, device)
