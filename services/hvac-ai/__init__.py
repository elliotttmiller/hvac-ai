"""
HVAC AI Inference Service Package
=================================

This package contains the core AI/ML services for the HVAC platform,
architected as a distributed inference graph using Ray Serve.

It provides the necessary components to build and run the AI pipeline:
- Service classes for individual AI tasks (Vision, Language).
- Ray Serve deployments that wrap these services.
- An application builder (`build_app`) that composes the deployments into a runnable graph.

This package is designed to be the "AI Engine" of the platform.
"""

__version__ = "2.0.0"  # Signifies a major architectural refactor

# --- Core Service Classes ---
# These are the pure Python classes that contain the actual AI logic.
# They are infrastructure-agnostic and can be tested independently.
from .object_detector_service import ObjectDetector
from .text_extractor_service import TextExtractor
from .utils.geometry import GeometryUtils, OBB

# --- Ray Serve Application Entrypoint ---
# This is the primary function used by startup scripts to build and run the entire
# distributed application. It follows the official "Application Builder" pattern.
from .inference_graph import build_app

# --- Ray Serve Deployments ---
# Exposing the deployment classes allows for advanced testing or composition
# in other potential application graphs.
from .inference_graph import (
    ObjectDetectorDeployment,
    TextExtractorDeployment,
    APIServer,
)


# --- Public API Definition (`__all__`) ---
# This defines what is considered the public, stable API of this package.
# Other services should only import names listed here.
__all__ = [
    # --- Core Logic ---
    "ObjectDetector",
    "TextExtractor",
    "GeometryUtils",
    "OBB",
    
    # --- Ray Serve Application ---
    "build_app",
    
    # --- Ray Serve Components ---
    "ObjectDetectorDeployment",
    "TextExtractorDeployment",
    "APIServer",
]
