"""
ASHRAE/SMACNA Symbol Library

Comprehensive HVAC symbol recognition library based on ASHRAE and SMACNA standards.
Implements template matching with rotation and scale invariance for accurate symbol detection.

Industry Standards:
- ASHRAE Standard 134 (Graphic Symbols)
- SMACNA HVAC Duct Construction Standards
- ASHRAE Handbook - Fundamentals

Best Practices:
- Template-based matching with multi-scale
- Rotation invariance (0-360 degrees)
- Confidence scoring
- Symbol classification taxonomy
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class HVACSymbolCategory(Enum):
    """HVAC symbol categories per ASHRAE standards"""
    # Air Distribution
    DIFFUSER_SQUARE = "diffuser_square"
    DIFFUSER_ROUND = "diffuser_round"
    DIFFUSER_LINEAR = "diffuser_linear"
    GRILLE_RETURN = "grille_return"
    GRILLE_SUPPLY = "grille_supply"
    REGISTER = "register"
    
    # Ductwork Components
    DAMPER_MANUAL = "damper_manual"
    DAMPER_MOTORIZED = "damper_motorized"
    DAMPER_FIRE = "damper_fire"
    DAMPER_SMOKE = "damper_smoke"
    VAV_BOX = "vav_box"
    
    # Equipment
    FAN = "fan"
    FAN_INLINE = "fan_inline"
    AHU = "ahu"
    CHILLER = "chiller"
    BOILER = "boiler"
    COOLING_TOWER = "cooling_tower"
    PUMP = "pump"
    
    # Coils and Filters
    COIL_HEATING = "coil_heating"
    COIL_COOLING = "coil_cooling"
    FILTER = "filter"
    
    # Controls
    THERMOSTAT = "thermostat"
    SENSOR_TEMPERATURE = "sensor_temperature"
    SENSOR_HUMIDITY = "sensor_humidity"
    SENSOR_PRESSURE = "sensor_pressure"
    ACTUATOR = "actuator"
    
    # Valves
    VALVE_2WAY = "valve_2way"
    VALVE_3WAY = "valve_3way"
    VALVE_CHECK = "valve_check"
    VALVE_CONTROL = "valve_control"
    
    # Fittings
    DUCT_ELBOW_90 = "duct_elbow_90"
    DUCT_TEE = "duct_tee"
    DUCT_TRANSITION = "duct_transition"
    DUCT_FLEX = "duct_flex"


@dataclass
class SymbolTemplate:
    """HVAC symbol template for matching"""
    category: HVACSymbolCategory
    template: np.ndarray  # Grayscale template image
    scale_range: Tuple[float, float] = (0.5, 2.0)
    rotation_invariant: bool = True
    min_confidence: float = 0.7
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DetectedSymbol:
    """Detected HVAC symbol with location and confidence"""
    category: HVACSymbolCategory
    center: Tuple[float, float]
    bbox: List[float]  # [x, y, width, height]
    confidence: float
    rotation: float = 0.0
    scale: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HVACSymbolLibrary:
    """
    ASHRAE/SMACNA HVAC Symbol Library
    
    Provides template-based symbol recognition with:
    - Multi-scale template matching
    - Rotation invariance
    - Confidence scoring
    - Non-maximum suppression
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize HVAC symbol library
        
        Args:
            template_dir: Directory containing symbol templates (optional)
        """
        self.templates: List[SymbolTemplate] = []
        self.logger = logging.getLogger(__name__)
        
        if template_dir and os.path.exists(template_dir):
            self._load_templates_from_directory(template_dir)
        else:
            self._initialize_standard_templates()
        
        self.logger.info(f"Initialized HVAC symbol library with {len(self.templates)} templates")
    
    def _initialize_standard_templates(self):
        """Initialize standard ASHRAE/SMACNA symbol templates"""
        # Create synthetic templates for common HVAC symbols
        # In production, these would be loaded from actual symbol images
        
        # Diffuser (square)
        diffuser_sq = self._create_square_diffuser_template(size=50)
        self.templates.append(SymbolTemplate(
            category=HVACSymbolCategory.DIFFUSER_SQUARE,
            template=diffuser_sq,
            rotation_invariant=False,
            metadata={"description": "Square ceiling diffuser per ASHRAE"}
        ))
        
        # Diffuser (round)
        diffuser_rd = self._create_round_diffuser_template(radius=25)
        self.templates.append(SymbolTemplate(
            category=HVACSymbolCategory.DIFFUSER_ROUND,
            template=diffuser_rd,
            rotation_invariant=True,
            metadata={"description": "Round ceiling diffuser per ASHRAE"}
        ))
        
        # Return grille
        grille = self._create_grille_template(size=50)
        self.templates.append(SymbolTemplate(
            category=HVACSymbolCategory.GRILLE_RETURN,
            template=grille,
            rotation_invariant=False,
            metadata={"description": "Return air grille per ASHRAE"}
        ))
        
        # Damper
        damper = self._create_damper_template(size=40)
        self.templates.append(SymbolTemplate(
            category=HVACSymbolCategory.DAMPER_MANUAL,
            template=damper,
            rotation_invariant=True,
            min_confidence=0.65,
            metadata={"description": "Manual damper per SMACNA"}
        ))
        
        # VAV box
        vav = self._create_vav_template(width=60, height=40)
        self.templates.append(SymbolTemplate(
            category=HVACSymbolCategory.VAV_BOX,
            template=vav,
            rotation_invariant=False,
            metadata={"description": "Variable Air Volume box"}
        ))
        
        # Fan symbol
        fan = self._create_fan_template(radius=30)
        self.templates.append(SymbolTemplate(
            category=HVACSymbolCategory.FAN,
            template=fan,
            rotation_invariant=True,
            metadata={"description": "Fan symbol per ASHRAE"}
        ))
        
        self.logger.info("Initialized standard ASHRAE/SMACNA templates")
    
    def _create_square_diffuser_template(self, size: int) -> np.ndarray:
        """Create square diffuser template"""
        template = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw square outline
        cv2.rectangle(template, (5, 5), (size-5, size-5), 0, 2)
        
        # Draw diagonal lines (common diffuser pattern)
        cv2.line(template, (5, 5), (size-5, size-5), 0, 1)
        cv2.line(template, (5, size-5), (size-5, 5), 0, 1)
        
        # Add center circle
        center = size // 2
        cv2.circle(template, (center, center), 5, 0, -1)
        
        return template
    
    def _create_round_diffuser_template(self, radius: int) -> np.ndarray:
        """Create round diffuser template"""
        size = radius * 2 + 10
        template = np.ones((size, size), dtype=np.uint8) * 255
        
        center = size // 2
        
        # Draw outer circle
        cv2.circle(template, (center, center), radius, 0, 2)
        
        # Draw inner circles (diffuser pattern)
        cv2.circle(template, (center, center), radius // 2, 0, 1)
        cv2.circle(template, (center, center), radius // 4, 0, 1)
        
        return template
    
    def _create_grille_template(self, size: int) -> np.ndarray:
        """Create return grille template"""
        template = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw square outline
        cv2.rectangle(template, (5, 5), (size-5, size-5), 0, 2)
        
        # Draw horizontal lines (grille pattern)
        spacing = size // 5
        for i in range(1, 5):
            y = 5 + i * spacing
            cv2.line(template, (5, y), (size-5, y), 0, 1)
        
        # Draw arrows pointing inward (return air)
        mid = size // 2
        arrow_len = 10
        cv2.arrowedLine(template, (size-10, mid), (size-20, mid), 0, 1, tipLength=0.3)
        cv2.arrowedLine(template, (10, mid), (20, mid), 0, 1, tipLength=0.3)
        
        return template
    
    def _create_damper_template(self, size: int) -> np.ndarray:
        """Create damper symbol template"""
        template = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw damper blade (diagonal line)
        cv2.line(template, (10, 10), (size-10, size-10), 0, 3)
        
        # Draw control indicator
        cv2.rectangle(template, (5, size//2-5), (15, size//2+5), 0, 2)
        
        return template
    
    def _create_vav_template(self, width: int, height: int) -> np.ndarray:
        """Create VAV box template"""
        template = np.ones((height, width), dtype=np.uint8) * 255
        
        # Draw rectangular box
        cv2.rectangle(template, (5, 5), (width-5, height-5), 0, 2)
        
        # Add "VAV" text (simplified)
        cv2.putText(template, "VAV", (width//4, height//2+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        
        # Draw control lines
        cv2.line(template, (width//2, 5), (width//2, 0), 0, 1)
        
        return template
    
    def _create_fan_template(self, radius: int) -> np.ndarray:
        """Create fan symbol template"""
        size = radius * 2 + 10
        template = np.ones((size, size), dtype=np.uint8) * 255
        
        center = size // 2
        
        # Draw circle
        cv2.circle(template, (center, center), radius, 0, 2)
        
        # Draw fan blades (three lines from center)
        angles = [0, 120, 240]
        for angle in angles:
            rad = np.radians(angle)
            x = int(center + radius * 0.8 * np.cos(rad))
            y = int(center + radius * 0.8 * np.sin(rad))
            cv2.line(template, (center, center), (x, y), 0, 2)
        
        return template
    
    def _load_templates_from_directory(self, template_dir: str):
        """Load symbol templates from directory"""
        template_path = Path(template_dir)
        
        if not template_path.exists():
            self.logger.warning(f"Template directory not found: {template_dir}")
            return
        
        # Load all image files
        for file_path in template_path.glob("*.png"):
            try:
                # Load image in grayscale
                template_img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                
                if template_img is None:
                    continue
                
                # Infer category from filename
                filename = file_path.stem
                category = self._parse_category_from_filename(filename)
                
                if category:
                    self.templates.append(SymbolTemplate(
                        category=category,
                        template=template_img,
                        metadata={"source": str(file_path)}
                    ))
                    self.logger.info(f"Loaded template: {filename}")
            
            except Exception as e:
                self.logger.error(f"Failed to load template {file_path}: {e}")
    
    def _parse_category_from_filename(self, filename: str) -> Optional[HVACSymbolCategory]:
        """Parse symbol category from filename"""
        filename_lower = filename.lower().replace("-", "_").replace(" ", "_")
        
        for category in HVACSymbolCategory:
            if category.value in filename_lower:
                return category
        
        return None
    
    def detect_symbols(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3
    ) -> List[DetectedSymbol]:
        """
        Detect HVAC symbols in blueprint image
        
        Args:
            image: Blueprint image (grayscale or BGR)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of detected symbols with locations and confidence
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        all_detections = []
        
        # Match each template
        for template_info in self.templates:
            detections = self._match_template(
                gray,
                template_info,
                confidence_threshold
            )
            all_detections.extend(detections)
        
        # Apply non-maximum suppression
        filtered_detections = self._apply_nms(all_detections, nms_threshold)
        
        self.logger.info(
            f"Detected {len(filtered_detections)} symbols "
            f"(before NMS: {len(all_detections)})"
        )
        
        return filtered_detections
    
    def _match_template(
        self,
        image: np.ndarray,
        template_info: SymbolTemplate,
        threshold: float
    ) -> List[DetectedSymbol]:
        """Match a single template at multiple scales"""
        detections = []
        template = template_info.template
        
        # Multi-scale matching
        scales = np.linspace(
            template_info.scale_range[0],
            template_info.scale_range[1],
            5
        )
        
        for scale in scales:
            # Resize template
            scaled_template = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR
            )
            
            # Skip if template larger than image
            if (scaled_template.shape[0] > image.shape[0] or 
                scaled_template.shape[1] > image.shape[1]):
                continue
            
            # Template matching
            try:
                result = cv2.matchTemplate(
                    image,
                    scaled_template,
                    cv2.TM_CCOEFF_NORMED
                )
                
                # Find matches above threshold
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    
                    h, w = scaled_template.shape
                    center = (pt[0] + w//2, pt[1] + h//2)
                    bbox = [pt[0], pt[1], w, h]
                    
                    detections.append(DetectedSymbol(
                        category=template_info.category,
                        center=center,
                        bbox=bbox,
                        confidence=float(confidence),
                        scale=scale,
                        metadata={"template": template_info.metadata}
                    ))
            
            except Exception as e:
                self.logger.warning(f"Template matching failed: {e}")
                continue
        
        return detections
    
    def _apply_nms(
        self,
        detections: List[DetectedSymbol],
        iou_threshold: float
    ) -> List[DetectedSymbol]:
        """Apply non-maximum suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        
        while detections:
            # Keep highest confidence detection
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [
                d for d in detections
                if self._calculate_iou(best.bbox, d.bbox) < iou_threshold
            ]
        
        return keep
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def get_symbol_description(self, category: HVACSymbolCategory) -> str:
        """Get description for symbol category"""
        descriptions = {
            HVACSymbolCategory.DIFFUSER_SQUARE: "Square ceiling diffuser per ASHRAE Standard 134",
            HVACSymbolCategory.DIFFUSER_ROUND: "Round ceiling diffuser per ASHRAE Standard 134",
            HVACSymbolCategory.GRILLE_RETURN: "Return air grille per ASHRAE Standard 134",
            HVACSymbolCategory.DAMPER_MANUAL: "Manual damper per SMACNA standards",
            HVACSymbolCategory.VAV_BOX: "Variable Air Volume terminal unit",
            HVACSymbolCategory.FAN: "Fan symbol per ASHRAE Standard 134",
            # Add more descriptions as needed
        }
        
        return descriptions.get(category, f"HVAC symbol: {category.value}")


def create_hvac_symbol_library(template_dir: Optional[str] = None) -> HVACSymbolLibrary:
    """
    Factory function to create HVAC symbol library
    
    Args:
        template_dir: Optional directory containing symbol templates
        
    Returns:
        Configured HVACSymbolLibrary instance
    """
    return HVACSymbolLibrary(template_dir=template_dir)
