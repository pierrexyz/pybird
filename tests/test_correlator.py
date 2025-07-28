#!/usr/bin/env python3
"""
Test script for PyBird Correlator class.
Tests core functionality of the Correlator class with mock data for speed.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_utils import MockData, TestHelpers, get_default_correlator_config, get_default_eft_params, get_default_bias_dict, check_imports


def test_correlator_import():
    """Test if Correlator class can be imported."""
    try:
        from pybird.correlator import Correlator
        print("✓ Correlator class imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Correlator class: {e}")
        return False


def test_correlator_initialization():
    """Test Correlator class initialization."""
    try:
        from pybird.correlator import Correlator
        
        # Initialize Correlator object with configuration
        correlator = Correlator()
        config = get_default_correlator_config(fast=True)
        correlator.set(config)
        
        # Check that basic attributes exist after set()
        assert hasattr(correlator, 'co'), "Correlator should have 'co' attribute"
        assert hasattr(correlator, 'c'), "Correlator should have 'c' attribute"
        assert hasattr(correlator, 'nonlinear'), "Correlator should have 'nonlinear' attribute"
        assert hasattr(correlator, 'projection'), "Correlator should have 'projection' attribute"
        
        # Bird attribute only exists after compute() is called
        assert not hasattr(correlator, 'bird'), "Correlator should not have 'bird' attribute before compute()"
        
        print("✓ Correlator initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ Correlator initialization failed: {e}")
        return False


def test_correlator_set_config():
    """Test Correlator set method with different configurations."""
    try:
        from pybird.correlator import Correlator
        
        # Test basic configuration
        correlator = Correlator()
        config = get_default_correlator_config(fast=True)
        
        correlator.set(config)
        
        # Check that configuration was applied
        assert correlator.c["output"] == 'bPk', "Output should be set to 'bPk'"
        assert correlator.c["multipole"] == 3, "Multipole should be set to 3"
        assert correlator.c["kmax"] == 0.4, "kmax should be set to 0.4"
        
        print("✓ Correlator set config successful")
        print(f"  Output: {correlator.c['output']}")
        print(f"  Multipole: {correlator.c['multipole']}")
        print(f"  kmax: {correlator.c['kmax']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Correlator set config failed: {e}")
        return False


def test_correlator_compute_basic():
    """Test basic Correlator compute functionality."""
    try:
        from pybird.correlator import Correlator
        
        # JAX mode is controlled by test runner - don't override here
        
        # Create mock cosmology
        cosmo_data = MockData.create_simple_cosmology(nk=300)
        
        # Initialize and configure Correlator
        correlator = Correlator()
        correlator.set(get_default_correlator_config(fast=True))
        
        # Compute
        correlator.compute(cosmo_data, do_core=True, do_survey_specific=True)
        
        # Check that computation was successful
        assert hasattr(correlator, 'bird'), "Correlator should have Bird instance"
        
        # Check if we have the expected attributes
        # The bird should have either fullPs or get() should work
        try:
            # Test the get method which should work
            bias_params = get_default_bias_dict()
            result = correlator.get(bias_params)
            
            # Check result
            assert isinstance(result, np.ndarray), "Result should be numpy array"
            assert result.shape[0] == 3, "Result should have 3 multipoles"
            assert result.shape[1] > 0, "Result should have k bins"
            
            # Check for reasonable values
            assert TestHelpers.check_no_nan_inf(result, "correlator result")
            assert TestHelpers.check_values_reasonable(result, min_val=-1e10, max_val=1e10, name="correlator result")
            
            print("✓ Correlator compute basic successful")
            print(f"  Result shape: {result.shape}")
            print(f"  Result range: [{result.min():.2e}, {result.max():.2e}]")
            
        except Exception as e:
            print(f"✗ Get method failed: {e}")
            # Fallback: check for fullPs attribute
            if hasattr(correlator.bird, 'fullPs'):
                print("✓ Bird has fullPs attribute")
            else:
                print("✗ Bird does not have fullPs attribute")
            raise
        
        return True
        
    except Exception as e:
        print(f"✗ Correlator compute basic failed: {e}")
        return False


def test_correlator_get_method():
    """Test Correlator get method."""
    try:
        from pybird.correlator import Correlator
        from pybird import config
        
        # JAX mode is controlled by test runner - don't override here
        
        # Create mock cosmology
        cosmo_data = MockData.create_simple_cosmology(nk=500)
        
        # Initialize and configure Correlator
        correlator = Correlator()
        correlator.set(get_default_correlator_config(fast=True))
        
        # Compute
        correlator.compute(cosmo_data, do_core=True, do_survey_specific=True)
        
        # Test get method with bias dictionary
        bias_params = get_default_bias_dict()
        result = correlator.get(bias_params)
        
        # Check result
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        assert result.shape[0] == 3, "Result should have 3 multipoles"
        assert result.shape[1] > 0, "Result should have k bins"
        
        # Check for reasonable values
        assert TestHelpers.check_no_nan_inf(result, "correlator result")
        assert TestHelpers.check_values_reasonable(result, min_val=-1e10, max_val=1e10, name="correlator result")
        
        print("✓ Correlator get method successful")
        print(f"  Result shape: {result.shape}")
        print(f"  Result range: [{result.min():.2e}, {result.max():.2e}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Correlator get method failed: {e}")
        return False


def test_correlator_output_types():
    """Test different output types."""
    try:
        from pybird.correlator import Correlator
        from pybird import config
        
        # JAX mode is controlled by test runner - don't override here
        
        # Create mock cosmology
        cosmo_data = MockData.create_simple_cosmology(nk=300)
        
        # Test different output types
        output_types = ['bPk', 'bCf']
        
        for output_type in output_types:
            try:
                correlator = Correlator()
                config_dict = get_default_correlator_config(fast=True)
                config_dict['output'] = output_type
                correlator.set(config_dict)
                
                # Compute
                correlator.compute(cosmo_data, do_core=True, do_survey_specific=True)
                
                # Test get method
                bias_params = get_default_bias_dict()
                result = correlator.get(bias_params)
                
                # Check result
                assert isinstance(result, np.ndarray), f"Result should be numpy array for {output_type}"
                assert result.shape[0] == 3, f"Result should have 3 multipoles for {output_type}"
                assert result.shape[1] > 0, f"Result should have bins for {output_type}"
                
                print(f"✓ Output type '{output_type}' works")
                
            except Exception as e:
                print(f"✗ Output type '{output_type}' failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Correlator output types test failed: {e}")
        return False


def test_correlator_multipole_options():
    """Test different multipole options."""
    try:
        from pybird.correlator import Correlator
        from pybird import config
        
        # JAX mode is controlled by test runner - don't override here
        
        # Create mock cosmology
        cosmo_data = MockData.create_simple_cosmology(nk=300)
        
        # Test different multipole options (PyBird accepts [0,2] or [0,2,4])
        multipole_options = [2, 3]  # Skip 1 as it's not valid
        
        for multipole in multipole_options:
            try:
                correlator = Correlator()
                config_dict = get_default_correlator_config(fast=True)
                config_dict['multipole'] = multipole
                correlator.set(config_dict)
                
                # Compute
                correlator.compute(cosmo_data, do_core=True, do_survey_specific=True)
                
                # Test get method
                bias_params = get_default_bias_dict()
                result = correlator.get(bias_params)
                
                # Check result
                assert isinstance(result, np.ndarray), f"Result should be numpy array for multipole {multipole}"
                expected_multipoles = 2 if multipole == 2 else 3
                assert result.shape[0] == expected_multipoles, f"Result should have {expected_multipoles} multipoles"
                assert result.shape[1] > 0, f"Result should have bins for multipole {multipole}"
                
                print(f"✓ Multipole {multipole} works")
                
            except Exception as e:
                print(f"✗ Multipole {multipole} failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Correlator multipole options test failed: {e}")
        return False


def test_correlator_jax_mode():
    """Test Correlator with JAX mode if available."""
    jax_available, _, _ = check_imports()
    
    if not jax_available:
        print("⚠ JAX not available, skipping JAX mode test")
        return True
    
    try:
        import jax
        import jax.numpy as jnp
        from pybird.correlator import Correlator
        
        # JAX mode is controlled by test runner - don't override here
        
        # Create mock cosmology and convert to JAX arrays
        cosmo_data_numpy = MockData.create_simple_cosmology(nk=300)
        
        # Convert numpy arrays to JAX arrays for JAX mode
        cosmo_data = {}
        for key, value in cosmo_data_numpy.items():
            if isinstance(value, np.ndarray):
                cosmo_data[key] = jnp.asarray(value)
                print(f"  Converted {key}: numpy.ndarray -> jax.Array")
            else:
                cosmo_data[key] = value
        
        # Initialize and configure Correlator
        correlator = Correlator()
        correlator.set(get_default_correlator_config(fast=True))
        
        # Compute
        correlator.compute(cosmo_data, do_core=True, do_survey_specific=True)
        
        # Test get method
        bias_params = get_default_bias_dict()
        result = correlator.get(bias_params)
        
        # Check result - should be JAX array when in JAX mode
        print(f"  Result type: {type(result)}")
        assert isinstance(result, (np.ndarray, jnp.ndarray)), "Result should be array"
        assert result.shape[0] == 3, "Result should have 3 multipoles"
        assert result.shape[1] > 0, "Result should have k bins"
        
        # Check for reasonable values
        result_np = np.array(result)
        assert TestHelpers.check_no_nan_inf(result_np, "JAX result")
        assert TestHelpers.check_values_reasonable(result_np, min_val=-1e10, max_val=1e10, name="JAX result")
        
        print("✓ Correlator JAX mode successful")
        print(f"  Result shape: {result.shape}")
        print(f"  Result range: [{result_np.min():.2e}, {result_np.max():.2e}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Correlator JAX mode failed: {e}")
        return False


def test_correlator_parameter_variations():
    """Test Correlator with different parameter variations."""
    try:
        from pybird.correlator import Correlator
        from pybird import config
        
        # JAX mode is controlled by test runner - don't override here
        
        # Create mock cosmology
        cosmo_data = MockData.create_simple_cosmology(nk=300)
        
        # Initialize and configure Correlator
        correlator = Correlator()
        correlator.set(get_default_correlator_config(fast=True))
        
        # Compute
        correlator.compute(cosmo_data, do_core=True, do_survey_specific=True)
        
        # Test different parameter sets
        base_params = get_default_bias_dict()
        param_sets = [
            {**base_params, 'b1': 1.0},  # Low b1
            {**base_params, 'b1': 2.0},  # Medium b1
            {**base_params, 'b1': 3.0},  # High b1
            {**base_params, 'b1': 2.0, 'b2': 1.0},  # Non-zero b2
            {**base_params, 'b1': 2.0, 'b4': 1.0},  # Non-zero b4
            {**base_params, 'b1': 2.0, 'b2': 1.0, 'b4': 1.0},  # All non-zero
        ]
        
        results = []
        for i, params in enumerate(param_sets):
            result = correlator.get(params)
            results.append(result)
            
            # Check result
            assert isinstance(result, np.ndarray), f"Result {i} should be numpy array"
            assert result.shape[0] == 3, f"Result {i} should have 3 multipoles"
            assert result.shape[1] > 0, f"Result {i} should have k bins"
            
            # Check for reasonable values
            assert TestHelpers.check_no_nan_inf(result, f"result {i}")
            assert TestHelpers.check_values_reasonable(result, min_val=-1e10, max_val=1e10, name=f"result {i}")
        
        # Check that different parameters give different results
        for i in range(len(results) - 1):
            diff = np.abs(results[i] - results[i+1]).max()
            assert diff > 1e-10, f"Results {i} and {i+1} should be different"
        
        print("✓ Correlator parameter variations successful")
        print(f"  Tested {len(param_sets)} parameter sets")
        
        return True
        
    except Exception as e:
        print(f"✗ Correlator parameter variations failed: {e}")
        return False


def main():
    """Run all Correlator tests."""
    print("Testing PyBird Correlator Class")
    print("=" * 50)
    
    tests = [
        ("Correlator import", test_correlator_import),
        ("Correlator initialization", test_correlator_initialization),
        ("Correlator set config", test_correlator_set_config),
        ("Correlator compute basic", test_correlator_compute_basic),
        ("Correlator get method", test_correlator_get_method),
        ("Correlator output types", test_correlator_output_types),
        ("Correlator multipole options", test_correlator_multipole_options),
        ("Correlator JAX mode", test_correlator_jax_mode),
        ("Correlator with CLASS", test_correlator_with_class),
        ("Correlator parameter variations", test_correlator_parameter_variations),
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        success = TestHelpers.run_test(test_func, test_name)
        results[test_name] = success
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if all_passed:
        print("\n✓ All Correlator tests passed!")
    else:
        print("\n✗ Some Correlator tests failed.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 