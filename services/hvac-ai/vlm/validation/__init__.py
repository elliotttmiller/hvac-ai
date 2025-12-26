"""
VLM Validation Module

Provides validation frameworks for HVAC VLM including:
- HVAC engineering rule validation
- Performance benchmarking
- Error analysis

Note: HVACBenchmark requires numpy. Import directly when needed.
"""

from .hvac_validator import HVACValidator, ValidationResult

__all__ = ["HVACValidator", "ValidationResult", "HVACBenchmark", "BenchmarkMetrics"]


def __getattr__(name):
    """Lazy import for dependencies that require numpy"""
    if name == "HVACBenchmark":
        from .benchmarks import HVACBenchmark
        return HVACBenchmark
    elif name == "BenchmarkMetrics":
        from .benchmarks import BenchmarkMetrics
        return BenchmarkMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
