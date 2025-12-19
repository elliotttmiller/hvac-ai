"""
HVAC Vision-Language Model (VLM) Module

This module provides a domain-specific VLM for HVAC blueprint analysis,
built on top of open-source foundation models like Qwen2-VL and InternVL.

Note: HVACVLMInterface and SyntheticDataGenerator require additional dependencies
(torch, PIL, etc.). Import them directly when needed to avoid dependency issues.
"""

from .data_schema import HVACComponentType, HVACDataSchema

# Lazy imports to avoid requiring torch/PIL at module import time
__all__ = [
    "HVACComponentType",
    "HVACDataSchema",
    "HVACVLMInterface",
    "SyntheticDataGenerator"
]

__version__ = "1.0.0"


def __getattr__(name):
    """Lazy import for dependencies that require torch/PIL"""
    if name == "HVACVLMInterface":
        from .model_interface import HVACVLMInterface
        return HVACVLMInterface
    elif name == "SyntheticDataGenerator":
        from .synthetic_generator import SyntheticDataGenerator
        return SyntheticDataGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
