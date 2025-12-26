"""
Core services module for distributed inference.
Universal, tool-agnostic service classes following Domain-Driven Design.
"""

from .object_detector import ObjectDetector
from .text_extractor import TextExtractor

__all__ = ['ObjectDetector', 'TextExtractor']
