"""
HVAC AI Services

HVAC-specialized AI services including:
- SAHI-powered component detection
- Prompt engineering framework
- Multi-model ensemble inference
- YOLO-based detection and inference
- Integrated detector pipeline
- Vision Language Model (VLM) capabilities
"""

__version__ = "1.0.0"

# Import key components for easy access
from .hvac_sahi_engine import create_hvac_sahi_predictor, HVACSAHIConfig
from .hvac_prompt_engineering import create_hvac_prompt_framework
from .hvac_detector import HVACDetector
from .yolo_inference import YOLOInferenceEngine
from .hvac_pipeline import HVACDrawingAnalyzer, create_hvac_analyzer
from .pipeline_models import (
    PipelineConfig, DetectionBox, DetectionResult,
    HVACResult, HVACInterpretation, PipelineStage
)
from .integrated_detector import IntegratedHVACDetector
from .yoloplan_detector import YOLOplanDetector, create_yoloplan_detector
from .yoloplan_bom import create_bom_generator, create_connectivity_analyzer

__all__ = [
    "create_hvac_sahi_predictor",
    "HVACSAHIConfig",
    "create_hvac_prompt_framework",
    "HVACDetector",
    "YOLOInferenceEngine",
    "HVACDrawingAnalyzer",
    "create_hvac_analyzer",
    "PipelineConfig",
    "DetectionBox",
    "DetectionResult",
    "HVACResult",
    "HVACInterpretation",
    "PipelineStage",
    "IntegratedHVACDetector",
    "YOLOplanDetector",
    "create_yoloplan_detector",
    "create_bom_generator",
    "create_connectivity_analyzer",
]
