# ADR 001: SAHI Integration for HVAC Blueprint Analysis

**Status:** Accepted  
**Date:** 2024-12-12  
**Decision Makers:** Development Team

## Context

The HVAC-AI platform needs to analyze large blueprint images (often >4000px dimensions) for HVAC component detection. The current approach using standard SAM (Segment Anything Model) faces several limitations:

1. **Memory Constraints:** Large blueprints cause GPU memory overflow (>12GB usage)
2. **Small Component Detection:** Small HVAC components (dampers, sensors) are missed at low resolutions
3. **Processing Time:** Full-resolution inference is extremely slow for large blueprints
4. **Scalability:** Performance degrades exponentially with blueprint size

## Decision

We will integrate SAHI (Slicing Aided Hyper Inference) as the primary inference engine for HVAC blueprint analysis, with HVAC-specific optimizations.

### Implementation Details

1. **Slice Configuration:**
   - Default slice size: 1024x1024 pixels (optimized for ductwork patterns)
   - Overlap ratio: 30% (balances continuity and performance)
   - Adaptive sizing based on blueprint complexity

2. **HVAC-Specific Optimizations:**
   - Priority weighting: Ductwork (1.0) > Diffusers (0.9) > Equipment (0.85) > Controls (0.7)
   - Confidence threshold: 0.40 (higher for critical components)
   - IoU threshold: 0.50 for result fusion

## Consequences

### Positive
- 90%+ detection rate on all HVAC component sizes
- GPU memory usage under 8GB for blueprints up to 10,000px
- Linear scaling with blueprint size
- Improved small component detection (dampers, sensors, controls)

### Negative
- Adds external library dependency (sahi>=0.11.0)
- More complex than simple full-image inference
- Result fusion adds processing overhead

## References

- SAHI Paper: https://arxiv.org/abs/2202.06934
- pr-document.md: Section 2.1 "HVAC SAHI Core Integration"

---

**Status:** Implemented  
**Related ADRs:** ADR-002 (Prompt Engineering), ADR-003 (System Validation)
