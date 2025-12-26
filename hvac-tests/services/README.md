# Inference Implementation Compliance Tests

This directory contains test suites to validate that the inference implementation (YOLO/Ultralytics) meets the specifications in `docs/INFERENCE_UPGRADE_IMPLEMENTATION.md`.

## Test Coverage

### test_inference_compliance.py
Validates core inference functionality (YOLO/Ultralytics):
- **Polygon/Segmentation**: Tests polygon/segmentation mask output
- **NMS Algorithm**: Tests Non-Maximum Suppression correctness
- **NMS Algorithm**: Tests Non-Maximum Suppression correctness
- **BBox Calculation**: Tests bounding box generation
- **HVAC Taxonomy**: Validates 65-class taxonomy
- **Cache Implementation**: Tests caching behavior
- **API Response Format**: Validates response structure

## Running Tests

### Prerequisites
```bash
  cd services/hvac-ai
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run All Tests
```bash
python tests/test_inference_compliance.py
```

### Run Specific Test Class
```bash
python -m unittest tests.test_inference_compliance.TestPolygonEncoding
```

### Run with Verbose Output
```bash
python tests/test_sam_compliance.py -v
```

## Test Results Interpretation

- **PASS**: Implementation meets specification
- **FAIL**: Implementation deviates from specification (see audit report)

## Known Issues

Based on the Inference Implementation Audit (docs/INFERENCE_IMPLEMENTATION_AUDIT.md), the following are known gaps:

1. **Classification System (CRITICAL)**: Current implementation uses placeholder logic instead of the documented multi-stage geometric + visual pipeline
2. **Cache Implementation**: Uses simple dictionary instead of true LRU cache
3. **API Parameters**: Some documented parameters not implemented (e.g., `return_top_k`, `enable_refinement`)

## CI/CD Integration

To integrate these tests into CI/CD:

```yaml
# .github/workflows/test.yml
- name: Run Inference Compliance Tests
  run: |
    cd python-services
    source venv/bin/activate
    python tests/test_inference_compliance.py
```

## Adding New Tests

When adding new tests:
1. Inherit from `unittest.TestCase`
2. Name test methods starting with `test_`
3. Add docstrings explaining what is being validated
4. Reference the specification section being tested
5. Update this README with test coverage

## References

- [INFERENCE_UPGRADE_IMPLEMENTATION.md](../../docs/INFERENCE_UPGRADE_IMPLEMENTATION.md) - Specification
- [INFERENCE_IMPLEMENTATION_AUDIT.md](../../docs/INFERENCE_IMPLEMENTATION_AUDIT.md) - Audit results
