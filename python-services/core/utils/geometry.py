"""
GeometryUtils Module - Perspective Transform Pipeline
Universal geometry utilities for oriented bounding box (OBB) manipulation.

Handles:
- OBB corner calculation
- Perspective transformation to rectify rotated crops
- Image preprocessing for OCR (grayscale, thresholding)
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OBB:
    """Oriented Bounding Box representation."""
    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float  # radians


class GeometryUtils:
    """
    Universal geometry utilities for handling oriented bounding boxes (OBB).
    Provides perspective transformation to rectify rotated regions.
    """
    
    @staticmethod
    def calculate_corners(obb: OBB) -> np.ndarray:
        """
        Calculate the 4 corner points of an oriented bounding box.
        
        Args:
            obb: Oriented bounding box with center, dimensions, and rotation
            
        Returns:
            Array of shape (4, 2) containing corner coordinates in order:
            [top-left, top-right, bottom-right, bottom-left]
        """
        cx, cy = obb.x_center, obb.y_center
        w, h = obb.width, obb.height
        angle = obb.rotation
        
        # Calculate half dimensions
        half_w = w / 2.0
        half_h = h / 2.0
        
        # Calculate corners in local coordinate system (before rotation)
        corners_local = np.array([
            [-half_w, -half_h],  # top-left
            [half_w, -half_h],   # top-right
            [half_w, half_h],    # bottom-right
            [-half_w, half_h]    # bottom-left
        ])
        
        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Rotate corners
        corners_rotated = corners_local @ rotation_matrix.T
        
        # Translate to world coordinates
        corners_world = corners_rotated + np.array([cx, cy])
        
        return corners_world
    
    @staticmethod
    def rectify_obb_region(
        image: np.ndarray,
        obb: OBB,
        padding: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract and rectify a rotated region from an image using perspective transform.
        
        Args:
            image: Source image (H, W, C) or (H, W)
            obb: Oriented bounding box defining the region
            padding: Extra padding around the extracted region (pixels)
            
        Returns:
            Tuple of (rectified_crop, metadata_dict)
            - rectified_crop: Horizontally aligned crop (rotation corrected to 0 degrees)
            - metadata_dict: Information about the transformation
        """
        try:
            # Calculate corner points
            corners = GeometryUtils.calculate_corners(obb)
            
            # Ensure corners are within image bounds
            h, w = image.shape[:2]
            corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
            corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
            
            # Source points (corners in original image)
            src_points = corners.astype(np.float32)
            
            # Destination points (rectified rectangle)
            # Width and height of the rectified crop
            dst_w = int(obb.width) + 2 * padding
            dst_h = int(obb.height) + 2 * padding
            
            dst_points = np.array([
                [0, 0],
                [dst_w, 0],
                [dst_w, dst_h],
                [0, dst_h]
            ], dtype=np.float32)
            
            # Compute perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective warp
            rectified = cv2.warpPerspective(
                image, M, (dst_w, dst_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            
            metadata = {
                'original_rotation': obb.rotation,
                'rectified_size': (dst_w, dst_h),
                'corners': corners.tolist(),
                'transform_matrix': M.tolist()
            }
            
            return rectified, metadata
            
        except Exception as e:
            logger.error(f"Failed to rectify OBB region: {e}", exc_info=True)
            # Return a fallback empty image
            fallback = np.ones((int(obb.height), int(obb.width), 3), dtype=np.uint8) * 255
            return fallback, {'error': str(e)}
    
    @staticmethod
    def preprocess_for_ocr(
        image: np.ndarray,
        apply_threshold: bool = True,
        denoise: bool = False
    ) -> np.ndarray:
        """
        Preprocess an image for OCR by enhancing contrast and text clarity.
        
        Args:
            image: Input image (can be color or grayscale)
            apply_threshold: Whether to apply adaptive thresholding
            denoise: Whether to apply denoising
            
        Returns:
            Preprocessed image optimized for OCR
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Denoise if requested
            if denoise:
                gray = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Apply adaptive thresholding for better text contrast
            if apply_threshold:
                # Use adaptive threshold to handle varying lighting
                processed = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,  # block size
                    2    # constant subtracted from mean
                )
            else:
                processed = gray
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to preprocess image for OCR: {e}", exc_info=True)
            return image
    
    @staticmethod
    def extract_and_preprocess_obb(
        image: np.ndarray,
        obb: OBB,
        padding: int = 5,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete pipeline: extract OBB region, rectify, and preprocess for OCR.
        
        Args:
            image: Source image
            obb: Oriented bounding box
            padding: Padding around extracted region
            preprocess: Whether to apply OCR preprocessing
            
        Returns:
            Tuple of (processed_crop, metadata)
        """
        # Rectify the region
        rectified, metadata = GeometryUtils.rectify_obb_region(image, obb, padding)
        
        # Preprocess for OCR if requested
        if preprocess and 'error' not in metadata:
            processed = GeometryUtils.preprocess_for_ocr(rectified)
            metadata['preprocessed'] = True
        else:
            processed = rectified
            metadata['preprocessed'] = False
        
        return processed, metadata
    
    @staticmethod
    def obb_from_dict(obb_dict: Dict) -> OBB:
        """
        Convert a dictionary representation to an OBB object.
        
        Args:
            obb_dict: Dictionary with keys: x_center, y_center, width, height, rotation
            
        Returns:
            OBB object
        """
        return OBB(
            x_center=float(obb_dict['x_center']),
            y_center=float(obb_dict['y_center']),
            width=float(obb_dict['width']),
            height=float(obb_dict['height']),
            rotation=float(obb_dict['rotation'])
        )
    
    @staticmethod
    def validate_obb(obb: OBB, image_shape: Tuple[int, int]) -> bool:
        """
        Validate that an OBB is reasonable for the given image.
        
        Args:
            obb: Oriented bounding box to validate
            image_shape: Image shape (height, width)
            
        Returns:
            True if OBB is valid, False otherwise
        """
        h, w = image_shape
        
        # Check if center is within image bounds
        if not (0 <= obb.x_center <= w and 0 <= obb.y_center <= h):
            return False
        
        # Check if dimensions are positive and reasonable
        if obb.width <= 0 or obb.height <= 0:
            return False
        
        if obb.width > w * 2 or obb.height > h * 2:
            return False
        
        return True
