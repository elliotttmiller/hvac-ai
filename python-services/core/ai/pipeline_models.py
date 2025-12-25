"""
Data models for the HVAC Drawing Analysis Pipeline.
Defines strict, serializable data structures for all pipeline stages.
"""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import time


class DetectionBox(BaseModel):
    """Oriented bounding box with class information."""
    x1: float = Field(ge=0, description="Top-left x coordinate")
    y1: float = Field(ge=0, description="Top-left y coordinate")
    x2: float = Field(ge=0, description="Bottom-right x coordinate")
    y2: float = Field(ge=0, description="Bottom-right y coordinate")
    confidence: float = Field(ge=0, le=1, description="Detection confidence score")
    class_id: int = Field(ge=0, description="Class ID")
    class_name: str = Field(description="Class name")
    
    @property
    def width(self) -> float:
        """Calculate box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Calculate box height."""
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate box center point."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """Calculate box area."""
        return self.width * self.height


class DetectionResult(BaseModel):
    """Results from Stage 1: Component & Text Region Detection."""
    detections: List[DetectionBox] = Field(default_factory=list, description="All detected objects")
    text_regions: List[DetectionBox] = Field(default_factory=list, description="Detected text regions")
    processing_time_ms: float = Field(description="Stage 1 processing time in milliseconds")
    image_width: int = Field(gt=0, description="Input image width")
    image_height: int = Field(gt=0, description="Input image height")
    model_version: str = Field(description="YOLO model version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detections": [
                    {
                        "x1": 100.0, "y1": 150.0, "x2": 200.0, "y2": 250.0,
                        "confidence": 0.95, "class_id": 3, "class_name": "valve"
                    }
                ],
                "text_regions": [
                    {
                        "x1": 205.0, "y1": 155.0, "x2": 280.0, "y2": 175.0,
                        "confidence": 0.87, "class_id": 0, "class_name": "id_letters"
                    }
                ],
                "processing_time_ms": 9.5,
                "image_width": 1920,
                "image_height": 1080,
                "model_version": "yolo11m-obb"
            }
        }


class TextRecognitionResult(BaseModel):
    """Results from Stage 2: Text Recognition."""
    region: DetectionBox = Field(description="Text region bounding box")
    text: str = Field(description="Recognized text content")
    confidence: float = Field(ge=0, le=1, description="OCR confidence score")
    preprocessing_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about preprocessing (padding, region size)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "region": {
                    "x1": 205.0, "y1": 155.0, "x2": 280.0, "y2": 175.0,
                    "confidence": 0.87, "class_id": 0, "class_name": "id_letters"
                },
                "text": "VAV-101",
                "confidence": 0.92,
                "preprocessing_metadata": {
                    "padding_applied": 8,
                    "region_width": 75,
                    "region_height": 20
                }
            }
        }


class HVACEquipmentType(str, Enum):
    """HVAC equipment types."""
    VAV = "VAV"  # Variable Air Volume
    AHU = "AHU"  # Air Handling Unit
    FCU = "FCU"  # Fan Coil Unit
    PIC = "PIC"  # Pressure Indicating Controller
    TE = "TE"    # Temperature Element
    FIT = "FIT"  # Flow Indicating Transmitter
    DAMPER = "DAMPER"
    VALVE = "VALVE"
    SENSOR = "SENSOR"
    UNKNOWN = "UNKNOWN"


class HVACInterpretation(BaseModel):
    """Semantic interpretation of recognized text."""
    text: str = Field(description="Original recognized text")
    equipment_type: HVACEquipmentType = Field(description="Identified equipment type")
    zone_number: Optional[str] = Field(None, description="Zone or sequence number")
    system_id: Optional[str] = Field(None, description="System identifier")
    confidence: float = Field(ge=0, le=1, description="Interpretation confidence")
    pattern_matched: str = Field(description="Pattern that matched the text")
    associated_component: Optional[DetectionBox] = Field(
        None,
        description="Spatially associated component"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "VAV-101",
                "equipment_type": "VAV",
                "zone_number": "101",
                "system_id": "VAV",
                "confidence": 0.98,
                "pattern_matched": "VAV-\\d+",
                "associated_component": {
                    "x1": 100.0, "y1": 150.0, "x2": 200.0, "y2": 250.0,
                    "confidence": 0.95, "class_id": 3, "class_name": "valve"
                }
            }
        }


class HVACInterpretationResult(BaseModel):
    """Results from Stage 3: HVAC Semantic Interpretation."""
    interpretations: List[HVACInterpretation] = Field(
        default_factory=list,
        description="All semantic interpretations"
    )
    processing_time_ms: float = Field(description="Stage 3 processing time in milliseconds")
    total_validated: int = Field(ge=0, description="Number of successfully validated patterns")
    total_failed: int = Field(ge=0, description="Number of failed validations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "interpretations": [
                    {
                        "text": "VAV-101",
                        "equipment_type": "VAV",
                        "zone_number": "101",
                        "system_id": "VAV",
                        "confidence": 0.98,
                        "pattern_matched": "VAV-\\d+"
                    }
                ],
                "processing_time_ms": 0.8,
                "total_validated": 15,
                "total_failed": 2
            }
        }


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    DETECTION = "detection"
    TEXT_RECOGNITION = "text_recognition"
    INTERPRETATION = "interpretation"
    COMPLETE = "complete"
    FAILED = "failed"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PipelineError(BaseModel):
    """Pipeline error information."""
    stage: PipelineStage = Field(description="Stage where error occurred")
    severity: ErrorSeverity = Field(description="Error severity")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")


class HVACResult(BaseModel):
    """Final end-to-end pipeline result."""
    request_id: str = Field(description="Unique request identifier")
    stage: PipelineStage = Field(description="Current/final pipeline stage")
    
    # Stage results
    detection_result: Optional[DetectionResult] = Field(None, description="Stage 1 results")
    text_results: List[TextRecognitionResult] = Field(
        default_factory=list,
        description="Stage 2 results"
    )
    interpretation_result: Optional[HVACInterpretationResult] = Field(
        None,
        description="Stage 3 results"
    )
    
    # Timing and performance
    total_processing_time_ms: float = Field(description="Total pipeline processing time")
    stage_timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage timing breakdown"
    )
    
    # Error handling
    errors: List[PipelineError] = Field(
        default_factory=list,
        description="Any errors encountered during processing"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-critical warnings"
    )
    
    # Metadata
    image_path: Optional[str] = Field(None, description="Input image path")
    timestamp: float = Field(default_factory=time.time, description="Result timestamp")
    
    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.stage == PipelineStage.COMPLETE and not any(
            e.severity == ErrorSeverity.CRITICAL for e in self.errors
        )
    
    @property
    def partial_success(self) -> bool:
        """Check if pipeline had partial success."""
        return len(self.text_results) > 0 or self.detection_result is not None
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_abc123",
                "stage": "complete",
                "detection_result": {
                    "detections": [],
                    "text_regions": [],
                    "processing_time_ms": 9.5,
                    "image_width": 1920,
                    "image_height": 1080,
                    "model_version": "yolo11m-obb"
                },
                "text_results": [],
                "interpretation_result": {
                    "interpretations": [],
                    "processing_time_ms": 0.8,
                    "total_validated": 15,
                    "total_failed": 2
                },
                "total_processing_time_ms": 18.7,
                "stage_timings": {
                    "detection": 9.5,
                    "text_recognition": 8.2,
                    "interpretation": 0.8,
                    "overhead": 0.2
                },
                "errors": [],
                "warnings": [],
                "timestamp": 1703525423.5
            }
        }


class PipelineConfig(BaseModel):
    """Configuration for the HVAC Drawing Analyzer pipeline."""
    # Detection configuration
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections"
    )
    
    # OCR configuration
    ocr_min_size: int = Field(default=8, description="Minimum text size for OCR")
    ocr_text_threshold: float = Field(default=0.65, description="OCR text threshold")
    ocr_low_text: float = Field(default=0.3, description="OCR low text threshold")
    ocr_canvas_size: int = Field(default=1024, description="OCR canvas size")
    padding_min: int = Field(default=5, description="Minimum padding for text regions")
    padding_max: int = Field(default=10, description="Maximum padding for text regions")
    
    # Interpretation configuration
    max_association_distance_multiplier: float = Field(
        default=2.0,
        description="Maximum distance for text-component association (in component sizes)"
    )
    
    # Performance configuration
    max_processing_time_ms: float = Field(
        default=25.0,
        description="Maximum total processing time in milliseconds"
    )
    max_concurrent_requests: int = Field(
        default=4,
        description="Maximum concurrent processing requests"
    )
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    
    # Caching configuration
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "confidence_threshold": 0.7,
                "ocr_min_size": 8,
                "ocr_text_threshold": 0.65,
                "max_processing_time_ms": 25.0,
                "enable_gpu": True
            }
        }
