#!/usr/bin/env python3
"""
Test runner for PyBird tests.
First runs numpy mode tests for all classes, then conditionally runs JAX tests.
"""

import sys
import os
import argparse

def check_jax_requirements():
    """Check if JAX requirements are available."""
    required_packages = [
        'jax', 'jaxlib', 'interpax', 'optax', 'flax', 'numpyro', 'blackjax', 'nautilus-sampler'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'nautilus-sampler':
                __import__('nautilus')
            elif package == 'jaxlib':
                __import__('jaxlib')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def run_module_test(module_name, test_name):
    """Run a specific test module."""
    try:
        module = __import__(module_name)
        main_func = getattr(module, 'main')
        return main_func()
    except Exception as e:
        print(f"{test_name} failed: {e}")
        return False

def run_specific_test_function(module_name, function_name, test_name):
    """Run a specific test function from a module."""
    try:
        module = __import__(module_name)
        test_func = getattr(module, function_name)
        return test_func()
    except Exception as e:
        print(f"{test_name} failed: {e}")
        return False

def run_numpy_mode_tests():
    """Run numpy mode tests for all classes."""
    print("Phase 1: Running Numpy Mode Tests")
    print("=" * 60)
    
    # Explicitly disable JAX for numpy mode tests
    try:
        from pybird.config import set_jax_enabled
        set_jax_enabled(False)
        print("✓ JAX disabled - running in numpy mode")
    except ImportError:
        print("⚠ Could not import set_jax_enabled, proceeding anyway")
    
    print()
    
    # Numpy-compatible tests (excluding emulator which has no numpy mode)
    numpy_tests = [
        ('test_utils', 'main', 'Test Utilities'),
        ('test_bird', 'test_bird_import', 'Bird Import'),
        ('test_bird', 'test_bird_initialization', 'Bird Initialization'),
        ('test_bird', 'test_bird_setcosmo', 'Bird Set Cosmology'),
        ('test_bird', 'test_bird_setbias', 'Bird Set Bias'),
        ('test_bird', 'test_bird_setps', 'Bird Set Power Spectrum'),
        ('test_bird', 'test_bird_setfullps', 'Bird Set Full Power Spectrum'),
        ('test_bird', 'test_bird_different_eft_basis', 'Bird Different EFT Basis'),
        ('test_correlator', 'test_correlator_import', 'Correlator Import'),
        ('test_correlator', 'test_correlator_initialization', 'Correlator Initialization'),
        ('test_correlator', 'test_correlator_set_config', 'Correlator Set Config'),
        ('test_correlator', 'test_correlator_compute_basic', 'Correlator Compute Basic'),
        ('test_correlator', 'test_correlator_get_method', 'Correlator Get Method'),
        ('test_correlator', 'test_correlator_output_types', 'Correlator Output Types'),
        ('test_correlator', 'test_correlator_multipole_options', 'Correlator Multipole Options'),
        ('test_correlator', 'test_correlator_parameter_variations', 'Correlator Parameter Variations'),
        ('test_likelihood', 'test_likelihood_import', 'Likelihood Import'),
        ('test_likelihood', 'test_likelihood_initialization', 'Likelihood Initialization'),
        ('test_likelihood', 'test_likelihood_set_data', 'Likelihood Set Data'),
        ('test_likelihood', 'test_likelihood_eft_parameters', 'Likelihood EFT Parameters'),
        ('test_likelihood', 'test_likelihood_chi2_computation', 'Likelihood Chi2 Computation'),
        ('test_likelihood', 'test_likelihood_different_configs', 'Likelihood Different Configs'),
        ('test_likelihood', 'test_likelihood_multiple_sky', 'Likelihood Multiple Sky'),
    ]
    
    results = {}
    all_passed = True
    
    for test_info in numpy_tests:
        if len(test_info) == 3:  # (module, function, name)
            module_name, function_name, test_name = test_info
            success = run_specific_test_function(module_name, function_name, test_name)
        else:  # (module, name) - run full module
            module_name, test_name = test_info[:2]
            success = run_module_test(module_name, test_name)
        
        results[test_name] = success
        if not success:
            all_passed = False
        print()
    
    return all_passed, results

def run_jax_mode_tests():
    """Run JAX mode tests if JAX requirements are available."""
    print("Phase 2: Running JAX Mode Tests")
    print("=" * 60)
    
    # Check JAX requirements
    jax_available, missing_packages = check_jax_requirements()
    
    if not jax_available:
        print("⚠ JAX requirements not available, skipping JAX tests")
        print(f"  Missing packages: {', '.join(missing_packages)}")
        print("  Install with: pip install -e .[jax]")
        return True, {}
    
    print("✓ JAX requirements available, proceeding with JAX tests")
    
    # Explicitly enable JAX for JAX mode tests
    try:
        from pybird.config import set_jax_enabled
        set_jax_enabled(True)
        print("✓ JAX enabled - running in JAX mode")
    except ImportError:
        print("⚠ Could not import set_jax_enabled, JAX tests may fail")
    
    print()
    
    # JAX-specific tests
    jax_tests = [
        ('test_correlator', 'test_correlator_jax_mode', 'Correlator JAX Mode'),
        ('test_bird', 'test_bird_jax_compatibility', 'Bird JAX Compatibility'),
        ('test_emulator', 'test_emulator_jax_dependency', 'Emulator JAX Dependency'),
        ('test_emulator', 'test_emulator_integration_with_correlator', 'Emulator Integration with Correlator'),
    ]
    
    results = {}
    all_passed = True
    
    for test_info in jax_tests:
        if len(test_info) == 3:  # (module, function, name)
            module_name, function_name, test_name = test_info
            success = run_specific_test_function(module_name, function_name, test_name)
        else:  # (module, name) - run full module
            module_name, test_name = test_info[:2]
            success = run_module_test(module_name, test_name)
        
        results[test_name] = success
        if not success:
            all_passed = False
        print()
    
    return all_passed, results

def run_structured_tests():
    """Run structured tests: numpy mode first, then JAX mode."""
    print("PyBird Structured Test Suite")
    print("=" * 60)
    print("Testing numpy mode first, then JAX mode if requirements are available")
    print()
    
    # Phase 1: Numpy mode tests
    numpy_success, numpy_results = run_numpy_mode_tests()
    
    print("\n" + "="*60 + "\n")
    
    # Phase 2: JAX mode tests
    jax_success, jax_results = run_jax_mode_tests()
    
    # Combine results
    all_results = {**numpy_results, **jax_results}
    all_passed = numpy_success and jax_success
    
    return all_passed, all_results


def run_single_class(class_name):
    """Run tests for a single class."""
    module_map = {
        'bird': ('test_bird', 'Bird Class'),
        'correlator': ('test_correlator', 'Correlator Class'),
        'likelihood': ('test_likelihood', 'Likelihood Class'),
        'emulator': ('test_emulator', 'Emulator Class'),
        'utils': ('test_utils', 'Test Utilities'),
    }
    
    if class_name not in module_map:
        print(f"Unknown class: {class_name}")
        print(f"Available classes: {', '.join(module_map.keys())}")
        return False
    
    module_name, test_name = module_map[class_name]
    print(f"Running tests for {test_name}")
    print("=" * 60)
    
    success = run_module_test(module_name, test_name)
    return success

def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total: {passed + failed} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='PyBird Test Suite')
    parser.add_argument('--class', dest='class_name', help='Run tests for specific class (bird, correlator, likelihood, emulator, utils)')
    parser.add_argument('--check-jax', action='store_true', help='Check JAX requirements only')
    
    args = parser.parse_args()
    
    # Check JAX requirements only
    if args.check_jax:
        jax_available, missing_packages = check_jax_requirements()
        if jax_available:
            print("✓ All JAX requirements are available")
            return 0
        else:
            print("✗ JAX requirements missing:")
            for package in missing_packages:
                print(f"  - {package}")
            return 1
    
    # Run tests for specific class
    if args.class_name:
        success = run_single_class(args.class_name)
        return 0 if success else 1
    
    # Default: run structured tests (numpy then JAX)
    success, results = run_structured_tests()
    
    print_summary(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 