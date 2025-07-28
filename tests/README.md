# PyBird Test Suite

This directory contains the improved test suite for PyBird, organized by core classes for better maintainability, speed, and coverage.

## Test Structure

The test suite is organized into modular files, each focusing on a specific PyBird class:

- **`test_utils.py`** - Test utilities and mock data generators
- **`test_bird.py`** - Tests for the Bird class (core power spectrum calculations)
- **`test_correlator.py`** - Tests for the Correlator class (main interface)
- **`test_likelihood.py`** - Tests for the Likelihood class (parameter inference)
- **`test_emulator.py`** - Tests for the Emulator class (neural network acceleration)
- **`run_tests.py`** - Test runner with multiple options

## Running Tests

### Basic Usage

```bash
# Run all basic tests (no external dependencies)
python run_tests.py

# Run specific class tests
python run_tests.py --class bird
python run_tests.py --class correlator

# Run fast tests (optimized for CI/CD)
python run_tests.py --fast

# Run comprehensive tests (includes CLASS)
python run_tests.py --comprehensive
```

### Test Options

- **`--basic`** - Run basic tests without external dependencies (default)
- **`--fast`** - Run essential tests optimized for speed
- **`--comprehensive`** - Run all tests including CLASS-dependent ones
- **`--class <name>`** - Run tests for a specific class
- **`--list-classes`** - List available test classes
- **`--all`** - Run all tests (same as `--comprehensive`)

### Individual Test Files

You can also run individual test files directly:

```bash
python test_bird.py
python test_correlator.py
python test_likelihood.py
python test_emulator.py
```

## Test Features

### Speed Optimizations

- **Mock Data**: Uses fast mock cosmological data instead of CLASS for basic tests
- **Reduced Precision**: Uses lower precision settings for faster computation
- **Smaller Arrays**: Uses smaller k-arrays and reduced resolution for speed
- **Selective Testing**: Fast mode runs only essential tests

### Good Coverage

- **Core Functionality**: Tests main methods and workflows
- **Error Handling**: Tests error conditions and edge cases
- **Configuration Options**: Tests different parameter combinations
- **Integration**: Tests interaction between classes
- **JAX Compatibility**: Tests JAX mode if available [[memory:2448258]]

### Robustness

- **Mock Data Generators**: Consistent test data without external dependencies
- **Graceful Failures**: Handles missing optional dependencies (JAX, CLASS)
- **Comprehensive Checks**: Validates output shapes, values, and types
- **Cleanup**: Properly cleans up temporary files

## Dependencies

### Required
- NumPy
- SciPy
- PyBird (obviously!)

### Optional
- **JAX** - For neural network emulation tests
- **CLASS** - For comprehensive cosmology tests
- **classy** - Python wrapper for CLASS

## Test Design Principles

1. **Speed First**: Tests should be fast enough to run frequently during development
2. **No External Dependencies**: Basic tests should work without CLASS or JAX
3. **Comprehensive Coverage**: Test core functionality, edge cases, and error handling
4. **Modular Design**: Each class has its own test file for better organization
5. **Clear Output**: Tests provide clear success/failure messages with details

## Adding New Tests

To add tests for a new class:

1. Create `test_<classname>.py` following the existing pattern
2. Add import and test functions
3. Include a `main()` function that runs all tests
4. Update `run_tests.py` to include the new test module

## Example Test Structure

```python
#!/usr/bin/env python3
"""
Test script for PyBird <Class> class.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_utils import MockData, TestHelpers, check_imports

def test_class_import():
    """Test if class can be imported."""
    # Test import logic
    pass

def test_class_functionality():
    """Test core functionality."""
    # Test core methods
    pass

def main():
    """Run all tests."""
    tests = [
        ("Import test", test_class_import),
        ("Functionality test", test_class_functionality),
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        success = TestHelpers.run_test(test_func, test_name)
        results[test_name] = success
        if not success:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

This test suite provides a solid foundation for maintaining PyBird quality while keeping tests fast and comprehensive. 