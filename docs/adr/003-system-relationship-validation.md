# ADR 003: HVAC System Relationship Validation

**Status:** Accepted  
**Date:** 2024-12-12  
**Decision Makers:** Development Team

## Context

Current system lacks understanding of HVAC system relationships:
- No validation of duct connectivity
- Cannot detect impossible component configurations
- No code compliance checking
- Missing spatial relationship analysis

## Decision

Implement graph-based relationship engine with ASHRAE/SMACNA validation rules:

1. **Relationship Graph:**
   - Component nodes with properties
   - Relationship edges (connected_to, feeds_into, etc.)
   - Spatial proximity analysis
   - Confidence scoring

2. **Validation Rules:**
   - Ductwork must connect to terminals (diffusers, grilles)
   - VAV boxes require main duct connection
   - Equipment clearance requirements
   - Physically possible flow paths

3. **Engineering Constraints:**
   - ASHRAE Standard 62.1 ventilation
   - SMACNA installation standards
   - Manufacturer clearance specifications
   - Local code requirements

## Consequences

### Positive
- Automatic connectivity validation
- Code compliance checking
- Impossible configuration detection
- Actionable violation reports
- System understanding for better analysis

### Negative
- Requires engineering rule maintenance
- Rule complexity for edge cases
- May have false positives initially

## Implementation

Key components:
- `HVACSystemEngine` class
- Component and relationship data structures
- Validation rule engine
- Graph export for visualization

## References

- pr-document.md: Section 2.2 "HVAC Component Relationship Engine"
- ASHRAE Standard 62.1
- SMACNA Duct Construction Standards

---

**Status:** Implemented  
**Related ADRs:** ADR-001 (SAHI), ADR-002 (Prompts)
