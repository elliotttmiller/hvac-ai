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
import torch.nn.functional as F
import math
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import base64
import io
from PIL import Image as PILImage

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
        self.transform = ResizeLongestSide(1024)  # Standard SAM transform

        # Simple in-memory cache for image embeddings.
        # Key = image hash (string), Value = embedding tensor
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        self.MAX_CACHE_SIZE = CACHE_DEFAULT_SIZE

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

            # Attempt to load weights; if pos_embed shape mismatches, try to
            # interpolate the checkpoint positional embeddings to the model's
            # expected grid. This is common when fine-tuned checkpoints used a
            # different input size than the runtime model.
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.warning(f"Initial state_dict load failed: {e}; attempting pos_embed interpolation if applicable...")
                # Look for an image_encoder positional embedding in the checkpoint
                ckpt_key = None
                for k in state_dict.keys():
                    if k.endswith("image_encoder.pos_embed") or k.endswith("pos_embed") and "image_encoder" in k:
                        ckpt_key = k
                        break

                # Fallback: common key used in SAM checkpoints
                if ckpt_key is None and 'image_encoder.pos_embed' in state_dict:
                    ckpt_key = 'image_encoder.pos_embed'

                if ckpt_key is not None and hasattr(self.model.image_encoder, 'pos_embed'):
                    try:
                        old_pos = state_dict[ckpt_key]
                        new_pos = self.model.image_encoder.pos_embed.data

                        if old_pos.shape != new_pos.shape:
                            logger.info(f"Interpolating pos_embed from {old_pos.shape} to {new_pos.shape}")

                            # Separate cls token if present
                            cls_token = old_pos[:, :1, :]
                            grid_tokens = old_pos[:, 1:, :]

                            old_num = grid_tokens.shape[1]
                            old_size = int(math.sqrt(old_num))
                            new_num = new_pos.shape[1] - 1
                            new_size = int(math.sqrt(new_num))

                            grid_tokens = grid_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                            # bicubic interpolation for positional embeddings
                            grid_tokens = F.interpolate(grid_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                            grid_tokens = grid_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)

                            new_pos_tensor = torch.cat([cls_token, grid_tokens], dim=1)
                            state_dict[ckpt_key] = new_pos_tensor
                            # try load again
                            self.model.load_state_dict(state_dict)
                            logger.info("Positional embedding interpolation succeeded and checkpoint loaded.")
                        else:
                            # shapes already match but load failed for other reasons
                            raise
                    except Exception as ex:
                        logger.error(f"Positional embedding interpolation failed: {ex}")
                        raise
                else:
                    # No pos_embed key to try; re-raise original error
                    raise
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
            # Log encoder expected size and pos_embed shape for diagnostics
            try:
                pe = getattr(self.model.image_encoder, 'pos_embed', None)
                if pe is not None:
                    logger.info(f"Model image_encoder.pos_embed shape: {tuple(pe.shape)}")
                logger.info(f"Model image_encoder.img_size: {getattr(self.model.image_encoder, 'img_size', 'unknown')}")
            except Exception:
                logger.debug("Could not read model.pos_embed during warm-up")

            self._get_image_embedding(dummy_image)
            logger.info("✅ Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _compute_image_hash(self, image: np.ndarray) -> str:
        return hashlib.sha256(image.tobytes()).hexdigest()

    def _get_image_embedding(self, image: np.ndarray) -> torch.Tensor:
        """Encodes an image and returns its embedding, using a hash-keyed cache.

        The cache key is a SHA256 hash of the raw image bytes. This avoids using
        unhashable numpy arrays as cache keys.
        """
        # 1) Build a hash key for the image
        image_hash = self._compute_image_hash(image)

        # 2) Return cached embedding if present
        if image_hash in self.embedding_cache:
            logger.info("Cache hit for image embedding")
            return self.embedding_cache[image_hash]

        # 3) Not cached -> compute embedding
        with torch.no_grad():
            input_image = self.transform.apply_image(image)
            input_tensor = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).contiguous()
            
            # Use channel-first 3D tensor [C, H, W]; mean/std shaped [C,1,1]
            pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(-1, 1, 1)
            pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(-1, 1, 1)
            input_tensor = (input_tensor - pixel_mean) / pixel_std

            # Ensure input has shape [B, C, H, W] for the image encoder
            # The SAM ViT expects a consistent patch grid determined by the
            # image encoder's configured `img_size`. Resize or pad the tensor
            # so that both H and W equal that expected size. This prevents
            # positional embedding mismatches when images aren't square.
            target_size = getattr(self.model.image_encoder, "img_size", 1024)
            _, H, W = input_tensor.shape

            # If the tensor is larger than expected, downsample. If smaller
            # on one axis (ResizeLongestSide keeps the longest side==target_size),
            # pad the short axis to make a square input.
            if H != target_size or W != target_size:
                logger.debug(f"Adjusting image tensor from (H={H},W={W}) to target (H=W={target_size})")
                # If either dimension is larger than target -> resize down
                if H > target_size or W > target_size:
                    input_tensor = F.interpolate(input_tensor.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
                else:
                    pad_h = target_size - H
                    pad_w = target_size - W
                    # pad = (pad_left, pad_right, pad_top, pad_bottom)
                    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), value=0.0)

            # debug logging of tensor shapes to help diagnose recognition failures
            try:
                logger.debug(f"_get_image_embedding: input_tensor.shape={tuple(input_tensor.shape)} device={input_tensor.device}")
            except Exception:
                pass

            embedding = self.model.image_encoder(input_tensor.unsqueeze(0))
            try:
                logger.debug(f"_get_image_embedding: embedding.shape={tuple(embedding.shape)}")
            except Exception:
                pass

        # 4) Store in cache with simple eviction policy
        if len(self.embedding_cache) >= self.MAX_CACHE_SIZE:
            # naive eviction: clear entire cache to free memory
            logger.info("Embedding cache full; clearing cache")
            self.embedding_cache.clear()

        self.embedding_cache[image_hash] = embedding
        return embedding

    def segment(self, image: np.ndarray, prompt: Dict[str, Any]) -> List[Dict]:
        """Perform interactive segmentation based on a user prompt."""
        start_time = time.perf_counter()
        original_size = image.shape[:2]
        image_embedding = self._get_image_embedding(image)

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
                dense_prompt_embeddings=self.model.prompt_encoder.get_dense_pe(),
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

        # Also create a PNG (base64) of the binary mask so frontends can render reliably
        try:
            pil_mask = PILImage.fromarray((binary_mask * 255).astype(np.uint8))
            bio = io.BytesIO()
            pil_mask.save(bio, format='PNG')
            mask_png_b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
        except Exception:
            mask_png_b64 = None

        processing_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Segmentation complete: '{label}' ({score:.2f}), time: {processing_time:.1f}ms")

        return [{
            "label": label, "score": score, "mask": rle_mask, "bbox": bbox,
            "mask_png": mask_png_b64
        }]

    def count(self, image: np.ndarray, grid_size: int, min_score: float = 0.2, debug: bool = False, max_grid_points: int = 2000) -> Dict:
        """Perform automated component counting on the entire diagram."""
        start_time = time.perf_counter()
        original_size = image.shape[:2]
        h, w = original_size

        image_embedding = self._get_image_embedding(image)
        all_detections = []
        raw_grid_scores = []
        grid_points = [(x, y) for y in range(grid_size // 2, h, grid_size) for x in range(grid_size // 2, w, grid_size)]

        # If the number of grid points is very large, deterministically
        # subsample to a maximum to avoid pathological runtimes. We use
        # evenly-spaced indices so the sampling is reproducible.
        num_points = len(grid_points)
        if num_points > int(max_grid_points):
            logger.info(f"Grid has {num_points} points; subsampling to max_grid_points={max_grid_points}")
            # compute evenly spaced indices across the range [0, num_points-1]
            indices = np.linspace(0, num_points - 1, int(max_grid_points)).round().astype(int)
            grid_points = [grid_points[i] for i in indices]
            num_points = len(grid_points)

        # Precompute dense position encoding once for efficiency and to avoid None
        dense_pe = self.model.prompt_encoder.get_dense_pe()

        for idx, (x, y) in enumerate(grid_points):
            if idx % 200 == 0:
                logger.info(f"Processing grid point {idx}/{len(grid_points)}")
            with torch.no_grad():
                coords = self.transform.apply_coords(np.array([[x, y]]), original_size)
                point_coords = torch.as_tensor(coords, dtype=torch.float, device=self.device)[None, :, :]
                point_labels = torch.as_tensor([1], dtype=torch.int, device=self.device)[None, :]

                sparse_embeddings, _ = self.model.prompt_encoder(
                    points=(point_coords, point_labels), boxes=None, masks=None
                )
                
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_pe,
                    multimask_output=False
                )
                
                score = float(iou_predictions.squeeze().item())

                # Diagnostics: compute mask stats even if below threshold so we can inspect empties
                try:
                    mask = self.model.postprocess_masks(low_res_masks, (self.model.image_encoder.img_size, self.model.image_encoder.img_size), original_size)
                    binary_mask = (mask.squeeze().cpu().numpy() > 0.0).astype(np.uint8)
                    mask_sum = int(binary_mask.sum())
                    mask_shape = binary_mask.shape
                except Exception as _:
                    binary_mask = None
                    mask_sum = 0
                    mask_shape = None

                # collect raw scores for diagnostics (include mask_sum)
                raw_grid_scores.append({'idx': idx, 'x': int(x), 'y': int(y), 'score': score, 'mask_sum': mask_sum})

                # Log or debug-print depending on debug flag
                if debug:
                    logger.info(f"Grid[{idx}] ({x},{y}) score={score:.3f} mask_sum={mask_sum} mask_shape={mask_shape}")

                if score >= float(min_score) and binary_mask is not None and mask_sum > 0:
                    all_detections.append({'mask': binary_mask, 'score': score})

        unique_detections = self._apply_nms(all_detections, NMS_IOU_THRESHOLD)

        counts = Counter()
        for det in unique_detections:
            label = self._classify_segment(det['mask'])
            counts[label] += 1
            
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Counting complete: {len(unique_detections)} objects found, time: {processing_time:.1f}ms")

        # Convert detections into a frontend-friendly 'segments' list (RLE masks + bbox + score + label)
        segments = []
        for det in unique_detections:
            # det['mask'] is a binary numpy array
            rle = self._mask_to_rle(det['mask'])
            bbox = self._mask_to_bbox(det['mask'])
            label = self._classify_segment(det['mask'])

            # create PNG base64 for robust client-side rendering
            try:
                pil_mask = PILImage.fromarray((det['mask'] * 255).astype(np.uint8))
                bio = io.BytesIO()
                pil_mask.save(bio, format='PNG')
                mask_png_b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
            except Exception:
                mask_png_b64 = None

            segments.append({
                "label": label,
                "score": float(det.get('score', 0.0)),
                "mask": rle,
                "bbox": bbox,
                "mask_png": mask_png_b64
            })

        # Compute compact score statistics for debugging
        score_stats = None
        try:
            import numpy as _np
            scores = _np.array([r['score'] for r in raw_grid_scores]) if raw_grid_scores else _np.array([])
            if scores.size > 0:
                score_stats = {
                    'max': float(scores.max()),
                    'mean': float(scores.mean()),
                    'median': float(_np.median(scores)),
                    'num_above_0_1': int((scores > 0.1).sum()),
                    'num_above_0_2': int((scores > 0.2).sum()),
                    'num_grid_points': int(scores.size)
                }
        except Exception:
            score_stats = None

        return {
            "total_objects_found": len(unique_detections),
            "counts_by_category": dict(counts),
            "processing_time_ms": processing_time,
            # include segments so the frontend can overlay masks/bboxes
            "segments": segments,
            "raw_grid_scores": raw_grid_scores if debug else None,
            "score_stats": score_stats
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