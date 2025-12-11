"""
SAM Model Inference Module (Production-Ready)
Segment Anything Model for P&ID and HVAC diagram analysis
"""

import os
import logging
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
from collections import Counter
import numpy as np
import cv2
import torch
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS (can be tuned) ---
CACHE_DEFAULT_SIZE = 10
NMS_IOU_THRESHOLD = 0.9

# --- HVAC TAXONOMY (from your training) ---
HVAC_TAXONOMY = [
    "Actuator-Diaphragm", "Actuator-Generic", "Actuator-Manual", "Actuator-Motorized",
    "Actuator-Piston", "Actuator-Pneumatic", "Actuator-Solenoid", "Valve-3Way", "Valve-4Way",
    "Valve-Angle", "Valve-Ball", "Valve-Butterfly", "Valve-Check", "Valve-Control", "Valve-Diaphragm",
    "Valve-Gate", "Valve-Generic", "Valve-Globe", "Valve-Needle", "Valve-Plug", "Valve-Relief",
    "Equipment-AgitatorMixer", "Equipment-Compressor", "Equipment-FanBlower", "Equipment-Generic",
    "Equipment-HeatExchanger", "Equipment-Motor", "Equipment-Pump-Centrifugal", "Equipment-Pump-Dosing",
    "Equipment-Pump-Generic", "Equipment-Pump-Screw", "Equipment-Vessel", "Component-DiaphragmSeal",
    "Component-Switch", "Controller-DCS", "Controller-Generic", "Controller-PLC", "Instrument-Analyzer",
    "Instrument-Flow-Indicator", "Instrument-Flow-Transmitter", "Instrument-Generic",
    "Instrument-Level-Indicator", "Instrument-Level-Switch", "Instrument-Level-Transmitter",
    "Instrument-Pressure-Indicator", "Instrument-Pressure-Switch", "Instrument-Pressure-Transmitter",
    "Instrument-Temperature", "Accessory-Drain", "Accessory-Generic", "Accessory-SightGlass",
    "Accessory-Vent", "Damper", "Duct", "Filter", "Fitting-Bend", "Fitting-Blind", "Fitting-Flange",
    "Fitting-Generic", "Fitting-Reducer", "Pipe-Insulated", "Pipe-Jacketed", "Strainer-Basket",
    "Strainer-Generic", "Strainer-YType", "Trap"
]

class SAMInferenceEngine:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = ResizeLongestSide(1024) # Standard SAM transform
        
        # LRU Cache for image embeddings. Key = image hash, Value = embedding tensor
        self.embedding_cache = lru_cache(maxsize=CACHE_DEFAULT_SIZE)(self._get_image_embedding)

        logger.info(f"Initializing SAM Inference Engine on {self.device}")
        self._load_model()
        self._warm_up_model()

    def _load_model(self):
        """Load the fine-tuned SAM model, intelligently handling both raw state_dict
        and full training checkpoint files.
        """
        try:
            logger.info(f"Attempting to load SAM model from {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            # First, load the entire file into CPU memory to inspect it
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            # Initialize the model architecture
            model_type = "vit_h"  # Our model is always a ViT-H
            self.model = sam_model_registry[model_type]()

            # --- INTELLIGENT LOADING LOGIC ---
            # Check if the loaded file is a full checkpoint (a dictionary with 'model_state_dict')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                logger.info("Detected a full training checkpoint. Extracting 'model_state_dict'...")
                # If it's a full checkpoint, we extract just the model's weights
                state_dict = checkpoint['model_state_dict']
            else:
                # Otherwise, it's a raw state_dict (like best_model.pth or the official sam_vit_h.pth)
                logger.info("Detected a raw state_dict. Loading directly...")
                state_dict = checkpoint

            # Load and move to device
            self.model.load_state_dict(state_dict)
            self.model.to(device=self.device)
            self.model.eval()

            logger.info("✅ Fine-tuned SAM model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load SAM model: {e}")
            raise RuntimeError(f"Could not initialize SAM model from {self.model_path}") from e

    def _warm_up_model(self):
        """Warm up the model with a dummy forward pass."""
        try:
            logger.info("Warming up model...")
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            self._get_image_embedding(dummy_image)
            logger.info("✅ Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _compute_image_hash(self, image: np.ndarray) -> str:
        return hashlib.sha256(image.tobytes()).hexdigest()

    def _get_image_embedding(self, image: np.ndarray) -> torch.Tensor:
        """Encodes an image and returns its embedding. This function is cached."""
        with torch.no_grad():
            input_image = self.transform.apply_image(image)
            input_tensor = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).contiguous()
            
            pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
            input_tensor = (input_tensor - pixel_mean) / pixel_std
            
            # Add batch dimension and encode
            embedding = self.model.image_encoder(input_tensor[None, :, :, :])
            return embedding

    def segment(self, image: np.ndarray, prompt: Dict[str, Any]) -> List[Dict]:
        """Perform interactive segmentation based on a user prompt."""
        start_time = time.perf_counter()
        
        original_size = image.shape[:2]
        image_embedding = self.embedding_cache(image)

        with torch.no_grad():
            if prompt.get('type') == "point":
                coords = np.array([prompt['data']['coords']])
                labels = np.array([prompt['data'].get('label', 1)])
                
                coords = self.transform.apply_coords(coords, original_size)
                
                point_coords = torch.as_tensor(coords, dtype=torch.float, device=self.device)[None, :, :]
                point_labels = torch.as_tensor(labels, dtype=torch.int, device=self.device)[None, :]
                
                sparse_embeddings, _ = self.model.prompt_encoder(
                    points=(point_coords, point_labels), boxes=None, masks=None
                )
            else:
                raise ValueError(f"Unsupported prompt type: {prompt.get('type')}")

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=None,
                multimask_output=False
            )

        final_mask = self.model.postprocess_masks(
            low_res_masks, (self.model.image_encoder.img_size, self.model.image_encoder.img_size), original_size
        ).squeeze().cpu()
        
        binary_mask = (final_mask > 0.0).numpy().astype(np.uint8)
        score = iou_predictions.squeeze().item()
        
        label = self._classify_segment(binary_mask)
        rle_mask = self._mask_to_rle(binary_mask)
        bbox = self._mask_to_bbox(binary_mask)

        processing_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Segmentation complete: '{label}' ({score:.2f}), time: {processing_time:.1f}ms")

        return [{
            "label": label, "score": score, "mask": rle_mask, "bbox": bbox
        }]

    def count(self, image: np.ndarray, grid_size: int) -> Dict:
        """Perform automated component counting on the entire diagram."""
        start_time = time.perf_counter()
        original_size = image.shape[:2]
        h, w = original_size
        
        image_embedding = self.embedding_cache(image)
        
        all_detections = []
        grid_points = [(x, y) for y in range(grid_size // 2, h, grid_size) for x in range(grid_size // 2, w, grid_size)]
        
        for x, y in grid_points:
            with torch.no_grad():
                coords = self.transform.apply_coords(np.array([[x, y]]), original_size)
                point_coords = torch.as_tensor(coords, dtype=torch.float, device=self.device)[None, :, :]
                point_labels = torch.as_tensor([1], dtype=torch.int, device=self.device)[None, :]

                sparse_embeddings, _ = self.model.prompt_encoder(
                    points=(point_coords, point_labels), boxes=None, masks=None
                )
                
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embedding, image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=None,
                    multimask_output=False
                )
                
                score = iou_predictions.squeeze().item()
                if score > 0.85: # Confidence threshold
                    mask = self.model.postprocess_masks(low_res_masks, (1024, 1024), original_size)
                    binary_mask = (mask.squeeze().cpu().numpy() > 0.0).astype(np.uint8)
                    all_detections.append({'mask': binary_mask, 'score': score})

        unique_detections = self._apply_nms(all_detections, NMS_IOU_THRESHOLD)
        
        counts = Counter()
        for det in unique_detections:
            label = self._classify_segment(det['mask'])
            counts[label] += 1
            
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Counting complete: {len(unique_detections)} objects found, time: {processing_time:.1f}ms")

        return {
            "total_objects_found": len(unique_detections),
            "counts_by_category": dict(counts),
            "processing_time_ms": processing_time
        }

    def _classify_segment(self, mask: np.ndarray) -> str:
        """Functional placeholder for classification."""
        mask_sum = int(np.sum(mask))
        return HVAC_TAXONOMY[mask_sum % len(HVAC_TAXONOMY)]

    def _mask_to_rle(self, mask: np.ndarray) -> Dict:
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return {"size": rle['size'], "counts": rle['counts']}

    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        if not rows.any() or not cols.any(): return [0, 0, 0, 0]
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]

    def _apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        if not detections: return []
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        for det in detections:
            is_unique = True
            for kept_det in keep:
                iou = self._calculate_iou(det['mask'], kept_det['mask'])
                if iou > iou_threshold:
                    is_unique = False
                    break
            if is_unique:
                keep.append(det)
        return keep

    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection / union) if union > 0 else 0.0

def create_sam_engine(model_path: str) -> SAMInferenceEngine:
    return SAMInferenceEngine(model_path=model_path)