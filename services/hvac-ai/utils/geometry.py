"""
GeometryUtils Module - Perspective Transform Pipeline
Universal geometry utilities for oriented bounding box (OBB) manipulation.

Handles:
- OBB corner calculation with robust sorting
- Perspective transformation to rectify rotated crops
- Advanced image preprocessing for OCR (adaptive thresholding, inversion, denoising)
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Union
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
    def get_polygon_points(obb: OBB) -> List[List[float]]:
        """
        Calculate the 4 corner points of an oriented bounding box.
        Used for visualization on the frontend.
        """
        corners = GeometryUtils.calculate_corners(obb)
        return corners.tolist()

    @staticmethod
    def calculate_corners(obb: OBB) -> np.ndarray:
        """
        Calculate the 4 corner points of an oriented bounding box.
        
        Args:
            obb: Oriented bounding box with center, dimensions, and rotation
            
        Returns:
            Array of shape (4, 2) containing corner coordinates.
        """
        # OpenCV's boxPoints is robust and handles rotation logic internally
        # We convert our OBB format (radians) to OpenCV format (degrees)
        # rect format: ((center_x, center_y), (width, height), angle_degrees)
        rect = ((obb.x_center, obb.y_center), (obb.width, obb.height), np.degrees(obb.rotation))
        box = cv2.boxPoints(rect)
        return box.astype(np.int32)
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Sorts coordinates to a consistent order:
        [top-left, top-right, bottom-right, bottom-left]
        
        This is critical for perspective transforms to ensure the image
        doesn't get flipped or mirrored during rectification.
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left will have the smallest sum, Bottom-right will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right will have the smallest difference, Bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

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
            # 1. Get raw corners
            raw_corners = GeometryUtils.calculate_corners(obb)
            
            # 2. Sort corners consistently (TL, TR, BR, BL)
            src_points = GeometryUtils.order_points(raw_corners.astype(np.float32))
            
            # 3. Determine width/height of the new straight image
            (tl, tr, br, bl) = src_points
            
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Add padding to dimensions
            dst_w = maxWidth + 2 * padding
            dst_h = maxHeight + 2 * padding
            
            # Destination points (rectified rectangle with padding)
            dst_points = np.array([
                [0, 0],
                [dst_w - 1, 0],
                [dst_w - 1, dst_h - 1],
                [0, dst_h - 1]
            ], dtype=np.float32)
            
            # Compute perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective warp
            rectified = cv2.warpPerspective(
                image, M, (dst_w, dst_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255) # Pad with white
            )
            
            metadata = {
                'original_rotation': obb.rotation,
                'rectified_size': (dst_w, dst_h),
                'corners': raw_corners.tolist(),
            }
            
            return rectified, metadata
            
        except Exception as e:
            logger.error(f"Failed to rectify OBB region: {e}", exc_info=True)
            # Return a fallback empty image to prevent pipeline crash
            fallback = np.ones((int(obb.height or 10), int(obb.width or 10), 3), dtype=np.uint8) * 255
            return fallback, {'error': str(e)}
    
    @staticmethod
    def preprocess_for_ocr(
        image: np.ndarray,
        apply_threshold: bool = True, # Default to True for blueprints
        denoise: bool = False
    ) -> np.ndarray:
        """
        Advanced preprocessing for OCR on engineering drawings.
        Handles grid lines, low contrast, and inverted text.
        """
        try:
            # 1. Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            processed = gray

            # 2. Adaptive Thresholding (The "Blueprint Cleaner")
            # This is crucial for removing grid lines and handling uneven lighting.
            # It converts the image to pure black and white based on local neighborhoods.
            if apply_threshold:
                processed = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,  # Block size (neighborhood size)
                    2    # Constant subtracted from mean
                )

            # 3. Denoising (Optional)
            if denoise:
                processed = cv2.fastNlMeansDenoising(processed, h=10)
            
            # 4. Inversion Check
            # OCR engines expect dark text on light background.
            # If the image is mostly black (white text), invert it.
            mean_brightness = cv2.mean(processed)[0]
            if mean_brightness < 127:
                processed = cv2.bitwise_not(processed)

            # 5. Convert back to BGR
            # PaddleOCR expects 3-channel input even for grayscale images
            final_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
            return final_image
            
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
        """
        # 1. Rectify (Straighten)
        rectified, metadata = GeometryUtils.rectify_obb_region(image, obb, padding)
        
        # 2. Preprocess (Clean)
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
        """
        h, w = image_shape[:2]
        
        # Basic bounds check
        if not (0 <= obb.x_center < w and 0 <= obb.y_center < h):
            return False
        
        # Dimension check
        if obb.width <= 0 or obb.height <= 0:
            return False
        
        # Sanity check (prevent massive false positives larger than image)
        if obb.width > w * 2 or obb.height > h * 2:
            return False
        
        return True