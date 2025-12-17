"""
Compatibility shim for legacy `sam_inference` API.

This module re-exports the YOLO inference implementation where available so
older imports of `core.ai.sam_inference` continue to work for static analysis
and runtime compatibility.
"""
try:
    # Prefer the new YOLO inference module
    from .yolo_inference import *  # noqa: F401,F403
except Exception:
    # If YOLO module not available, provide a minimal placeholder to avoid
    # import errors during static analysis. Runtime users should install the
    # appropriate legacy SAM implementation if needed.
    class SAMPlaceholder:
        """Placeholder object indicating SAM is not available."""

    HVAC_TAXONOMY = []
    create_sam_engine = None
