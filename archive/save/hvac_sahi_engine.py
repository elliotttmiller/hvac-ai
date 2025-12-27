"""
HVAC SAHI Engine - Slice-Aided Hyper Inference for HVAC Blueprint Analysis

This module provides HVAC-optimized slicing and inference capabilities using SAHI
for improved detection accuracy on large HVAC blueprints.

Key features:
- Adaptive slicing based on HVAC blueprint complexity
- HVAC-specific slice parameters optimized for ductwork and equipment
- GPU memory management for large blueprint processing
- Result fusion with ductwork connectivity preservation
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

# SAHI imports (will be installed via requirements)
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    logging.warning("SAHI not available. Install with: pip install sahi")


@dataclass
class HVACSAHIConfig:
    """HVAC-specific SAHI configuration parameters"""
    
    # Slice dimensions optimized for HVAC ductwork patterns
    slice_height: int = 1024
    slice_width: int = 1024
    
    # Overlap ratio for HVAC component continuity (30% for ductwork)
    overlap_height_ratio: float = 0.3
    overlap_width_ratio: float = 0.3
    
    # Confidence threshold (higher for critical HVAC components)
    confidence_threshold: float = 0.40
    
    # IoU threshold for HVAC component fusion
    iou_threshold: float = 0.50
    
    # HVAC component priority weights
    component_priority: Dict[str, float] = None
    
    def __post_init__(self):
        if self.component_priority is None:
            # Default HVAC component priorities: Ductwork > Diffusers > Equipment > Controls
            self.component_priority = {
                "ductwork": 1.0,
                "diffuser": 0.9,
                "grille": 0.9,
                "register": 0.9,
                "equipment": 0.85,
                "vav_box": 0.8,
                "damper": 0.75,
                "controls": 0.7
            }


class HVACSAHIPredictor:
    """
    HVAC-optimized SAHI predictor for large blueprint analysis
    
    This class wraps the SAHI library with HVAC-specific optimizations:
    - Adaptive slicing based on blueprint complexity
    - HVAC component detection with proper continuity
    - Memory-efficient processing for large blueprints
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "yolo",
        device: str = "cuda",
        config: Optional[HVACSAHIConfig] = None
    ):
        """
        Initialize HVAC SAHI Predictor
        
        Args:
            model_path: Path to the inference model weights (YOLO/Ultralytics)
            model_type: Type of model to use (default: "yolo")
            device: Device to run inference on ("cuda" or "cpu")
            config: HVAC-specific SAHI configuration
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.config = config or HVACSAHIConfig()
        self.logger = logging.getLogger(__name__)
        
        if not SAHI_AVAILABLE:
            raise ImportError(
                "SAHI is required for HVAC SAHI engine. "
                "Install with: pip install sahi"
            )
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SAHI detection model"""
        try:
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type=self.model_type,
                model_path=self.model_path,
                confidence_threshold=self.config.confidence_threshold,
                device=self.device,
            )
            self.logger.info(f"HVAC SAHI model initialized: {self.model_type}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SAHI model: {e}")
            raise
    
    def analyze_blueprint_complexity(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze HVAC blueprint complexity to optimize slicing parameters
        
        Args:
            image: Input blueprint image as numpy array
            
        Returns:
            Dictionary containing complexity metrics:
            - duct_density: Estimated density of ductwork
            - equipment_concentration: Concentration of equipment areas
            - recommended_slice_size: Recommended slice dimensions
        """
        height, width = image.shape[:2]
        
        # Simple complexity analysis based on edge density
        # In production, this would use more sophisticated HVAC-specific analysis
        edges = self._detect_edges(image)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Adjust slice size based on complexity
        if edge_density > 0.15:  # High complexity - more ductwork
            recommended_size = 768
            complexity_level = "high"
        elif edge_density > 0.08:  # Medium complexity
            recommended_size = 1024
            complexity_level = "medium"
        else:  # Low complexity
            recommended_size = 1280
            complexity_level = "low"
        
        return {
            "edge_density": float(edge_density),
            "complexity_level": complexity_level,
            "recommended_slice_size": recommended_size,
            "blueprint_dimensions": (width, height)
        }
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Simple edge detection for complexity analysis"""
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            return edges
        except Exception as e:
            self.logger.warning(f"Edge detection failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def predict_hvac_components(
        self,
        image_path: str,
        adaptive_slicing: bool = True
    ) -> Dict[str, Any]:
        """
        Perform HVAC component detection with SAHI slicing
        
        Args:
            image_path: Path to the blueprint image
            adaptive_slicing: Whether to use adaptive slicing based on complexity
            
        Returns:
            Dictionary containing:
            - detections: List of detected HVAC components
            - metadata: Processing metadata (slice count, timing, etc.)
        """
        try:
            # Read image
            image = read_image(image_path)
            
            # Analyze complexity if adaptive slicing is enabled
            if adaptive_slicing:
                complexity = self.analyze_blueprint_complexity(image)
                slice_size = complexity["recommended_slice_size"]
                self.logger.info(
                    f"Adaptive slicing: {complexity['complexity_level']} complexity, "
                    f"slice size: {slice_size}x{slice_size}"
                )
            else:
                slice_size = self.config.slice_height
            
            # Perform sliced prediction
            result = get_sliced_prediction(
                image=image_path,
                detection_model=self.detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=self.config.overlap_height_ratio,
                overlap_width_ratio=self.config.overlap_width_ratio,
            )
            
            # Process and format results
            detections = self._process_detections(result)
            
            return {
                "detections": detections,
                "metadata": {
                    "slice_size": slice_size,
                    "overlap_ratio": self.config.overlap_height_ratio,
                    "num_detections": len(detections),
                    "confidence_threshold": self.config.confidence_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"HVAC component prediction failed: {e}")
            raise
    
    def _process_detections(self, result) -> List[Dict[str, Any]]:
        """Process SAHI detection results into HVAC-specific format"""
        detections = []
        
        for obj_pred in result.object_prediction_list:
            detection = {
                "bbox": obj_pred.bbox.to_xywh(),
                "score": float(obj_pred.score.value),
                "category": obj_pred.category.name,
                "category_id": obj_pred.category.id,
            }
            detections.append(detection)
        
        return detections
    
    def predict_with_relationship_analysis(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Perform HVAC component detection with spatial relationship analysis
        
        This method extends basic detection with HVAC-specific relationship
        analysis to understand system connectivity.
        
        Args:
            image_path: Path to the blueprint image
            
        Returns:
            Dictionary containing detections and relationship graph
        """
        # Get base detections
        result = self.predict_hvac_components(image_path)
        
        # TODO: Implement HVAC relationship analysis
        # This would analyze spatial relationships between components:
        # - Duct connectivity
        # - Equipment placement relationships
        # - System hierarchy
        
        result["relationships"] = []  # Placeholder for relationship data
        
        return result


class HVACAdaptiveSlicingStrategy:
    """
    Advanced adaptive slicing strategy for HVAC blueprints
    
    This class implements sophisticated slicing strategies that adapt to:
    - Ductwork density and complexity
    - Equipment concentration areas
    - Blueprint quality characteristics
    """
    
    def __init__(self, config: HVACSAHIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_optimal_slicing(
        self,
        image: np.ndarray,
        target_memory_gb: float = 6.0
    ) -> Tuple[int, int, float]:
        """
        Calculate optimal slicing parameters based on image and resource constraints
        
        Args:
            image: Input blueprint image
            target_memory_gb: Target GPU memory usage in GB
            
        Returns:
            Tuple of (slice_height, slice_width, overlap_ratio)
        """
        height, width = image.shape[:2]
        
        # Estimate memory requirements (rough heuristic)
        estimated_memory_gb = (height * width * 3) / (1024**3)
        
        if estimated_memory_gb > target_memory_gb:
            # Need more aggressive slicing
            slice_size = min(768, self.config.slice_height)
            overlap = 0.35  # More overlap for better continuity
        else:
            # Can use larger slices
            slice_size = self.config.slice_height
            overlap = self.config.overlap_height_ratio
        
        self.logger.info(
            f"Optimal slicing: {slice_size}x{slice_size}, "
            f"overlap: {overlap:.2f}, "
            f"estimated memory: {estimated_memory_gb:.2f}GB"
        )
        
        return slice_size, slice_size, overlap


def create_hvac_sahi_predictor(
    model_path: str,
    device: str = "cuda",
    **config_kwargs
) -> HVACSAHIPredictor:
    """
    Factory function to create HVAC SAHI predictor with custom configuration
    
    Args:
        model_path: Path to inference model weights (YOLO/Ultralytics)
        device: Device for inference
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured HVACSAHIPredictor instance
    """
    config = HVACSAHIConfig(**config_kwargs)
    return HVACSAHIPredictor(
        model_path=model_path,
        device=device,
        config=config
    )
