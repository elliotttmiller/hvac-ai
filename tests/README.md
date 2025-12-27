# HVAC-AI Test Suite

Comprehensive test suite for HVAC-AI platform components.

## Structure

```
hvac-tests/
├── unit/
│   └── hvac-components/
│       ├── test_sahi_engine.py
│       ├── test_system_engine.py
│       ├── test_prompt_engineering.py
│       └── test_document_processor.py
└── integration/
    └── hvac-systems/
        ├── test_end_to_end_analysis.py
        └── test_pipeline_integration.py
```

## Running Tests

### All Tests
```bash
# Run all tests
pytest hvac-tests/

# With coverage
pytest hvac-tests/ --cov=services --cov-report=html
```

### Unit Tests Only
```bash
# Run all unit tests
pytest hvac-tests/unit/

# Run specific module
pytest hvac-tests/unit/hvac-components/test_sahi_engine.py

# Run specific test
pytest hvac-tests/unit/hvac-components/test_sahi_engine.py::TestHVACSAHIConfig::test_default_config_creation
```

### Integration Tests
```bash
# Run integration tests
pytest hvac-tests/integration/

# With verbose output
pytest hvac-tests/integration/ -v
```

## Test Coverage Goals

- **Unit Tests:** 85%+ coverage for core modules
- **Integration Tests:** 80%+ coverage for workflows
- **Critical Paths:** 95%+ coverage

### Current Coverage

| Module | Coverage | Status |
|--------|----------|--------|
| hvac_sahi_engine.py | 85% | ✅ |
| hvac_system_engine.py | 88% | ✅ |
| hvac_prompt_engineering.py | TBD | 🔄 |
| hvac_document_processor.py | TBD | 🔄 |

## Writing Tests

### Unit Test Template

```python
"""
Unit tests for [Module Name]
"""

import pytest
import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from services.module_name import ClassName


class TestClassName:
    """Tests for ClassName"""
    
    @pytest.fixture
    def instance(self):
        """Create instance for testing"""
        return ClassName()
    
    def test_basic_functionality(self, instance):
        """Test basic functionality"""
        result = instance.method()
        assert result is not None
    
    def test_edge_case(self, instance):
        """Test edge case handling"""
        with pytest.raises(ValueError):
            instance.method(invalid_input)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Integration Test Template

```python
"""
Integration tests for [Feature Name]
"""

import pytest
from services.hvac_ai.module1 import Class1
from services.hvac_domain.module2 import Class2


class TestFeatureIntegration:
    """Integration tests for feature workflow"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow"""
        # Setup
        component1 = Class1()
        component2 = Class2()
        
        # Execute workflow
        result1 = component1.process()
        result2 = component2.process(result1)
        
        # Verify
        assert result2['status'] == 'success'
```

## Test Data

### Sample Blueprints
Test blueprints are located in `hvac-datasets/test/`:
- `simple_plan.png` - Simple HVAC plan for basic tests
- `complex_plan.pdf` - Complex multi-page blueprint
- `poor_quality.jpg` - Low-quality scan for enhancement testing

### Mock Data
Mock components and relationships defined in `hvac-tests/fixtures/`:
```python
from hvac_tests.fixtures import mock_components, mock_relationships

def test_with_mocks():
    components = mock_components.get_sample_hvac_system()
    assert len(components) > 0
```

## Continuous Integration

Tests run automatically on:
- Every pull request
- Every commit to main branch
- Nightly builds

### CI Configuration
See `.github/workflows/test.yml` for CI setup.

### Quality Gates
- All tests must pass
- Coverage must be >= 85% for changed files
- No critical security issues
- No linting errors

## Debugging Failed Tests

### View Detailed Output
```bash
# Verbose mode
pytest hvac-tests/ -v

# Show print statements
pytest hvac-tests/ -s

# Stop on first failure
pytest hvac-tests/ -x

# Run last failed tests
pytest --lf
```

### Common Issues

**Import Errors:**
```bash
# Ensure Python path is set
export PYTHONPATH="${PYTHONPATH}:/home/runner/work/hvac-ai/hvac-ai"
```

**Missing Dependencies:**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock
```

**SAHI Not Available:**
```bash
# Install SAHI
pip install sahi>=0.11.0
```

## Performance Testing

### Benchmark Tests
```bash
# Run performance benchmarks
pytest hvac-tests/benchmarks/ --benchmark-only

# Compare with baseline
pytest hvac-tests/benchmarks/ --benchmark-compare
```

### Memory Profiling
```bash
# Profile memory usage
pytest hvac-tests/ --memprof

# Analyze large blueprint processing
pytest hvac-tests/unit/test_sahi_engine.py::test_large_blueprint --memprof
```

## Best Practices

1. **Isolation:** Each test should be independent
2. **Clarity:** Test names should describe what they test
3. **Coverage:** Test both success and failure paths
4. **Performance:** Keep unit tests fast (< 1s each)
5. **Fixtures:** Use fixtures for common setup
6. **Mocking:** Mock external dependencies

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure >85% coverage for new code
3. Run full test suite before committing
4. Update this README if adding new test categories

## Support

For test-related questions:
- Check existing tests for examples
- Review pytest documentation
- Contact the development team

---

**Status:** Active Development  
**Test Framework:** pytest  
**Last Updated:** December 2024
