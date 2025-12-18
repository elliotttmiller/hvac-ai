"""
HVAC Vision-Language Model (VLM) Module

This module provides a domain-specific VLM for HVAC blueprint analysis,
built on top of open-source foundation models like Qwen2-VL and InternVL.
"""

from .data_schema import HVACComponentType, HVACDataSchema
from .model_interface import HVACVLMInterface
from .synthetic_generator import SyntheticDataGenerator

__all__ = [
    "HVACComponentType",
    "HVACDataSchema", 
    "HVACVLMInterface",
    "SyntheticDataGenerator"
]

__version__ = "1.0.0"
