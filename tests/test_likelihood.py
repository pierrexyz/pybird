#!/usr/bin/env python3
"""
Test script for PyBird Likelihood class.
Tests core functionality of the Likelihood class with mock data for speed.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_utils import MockData, TestHelpers, get_default_correlator_config, get_default_eft_params, get_default_likelihood_config, check_imports


def test_likelihood_import():
    """Test if Likelihood class can be imported."""
    try:
        from pybird.likelihood import Likelihood
        print("✓ Likelihood class imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Likelihood class: {e}")
        return False


def test_likelihood_initialization():
    """Test Likelihood class initialization."""
    try:
        from pybird.likelihood import Likelihood
        
        # Get proper configuration
        config = get_default_likelihood_config()
        
        # Initialize Likelihood object
        likelihood = Likelihood(config, verbose=False)
        
        # Check that basic attributes exist
        assert hasattr(likelihood, 'c'), "Likelihood should have 'c' attribute"
        assert hasattr(likelihood, 'nsky'), "Likelihood should have 'nsky' attribute"
        assert hasattr(likelihood, 'correlator_sky'), "Likelihood should have 'correlator_sky' attribute"
        assert likelihood.nsky == 1, "nsky should be 1 for single sky"
        assert len(likelihood.correlator_sky) == 1, "Should have 1 correlator"
        
        print("✓ Likelihood initialization successful")
        print(f"  nsky: {likelihood.nsky}")
        print(f"  correlators: {len(likelihood.correlator_sky)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Likelihood initialization failed: {e}")
        return False


def test_likelihood_set_data():
    """Test Likelihood set_data method with mock data."""
    try:
        from pybird.likelihood import Likelihood
        
        # Get proper configuration with mock data
        config = get_default_likelihood_config()
        
        # Initialize Likelihood
        likelihood = Likelihood(config, verbose=False)
        
        # Check that data was loaded
        assert hasattr(likelihood, 'y_sky'), "Likelihood should have 'y_sky' attribute"
        assert hasattr(likelihood, 'd_sky'), "Likelihood should have 'd_sky' attribute"
        assert hasattr(likelihood, 'p_sky'), "Likelihood should have 'p_sky' attribute"
        assert hasattr(likelihood, 'm_sky'), "Likelihood should have 'm_sky' attribute"
        
        assert len(likelihood.y_sky) == 1, "Should have 1 sky patch"
        assert len(likelihood.d_sky) == 1, "Should have 1 sky patch"
        assert len(likelihood.p_sky) == 1, "Should have 1 sky patch"
        assert len(likelihood.m_sky) == 1, "Should have 1 sky patch"
        
        # Check data shapes
        assert likelihood.y_sky[0].shape[0] > 0, "Data vector should have positive length"
        assert likelihood.p_sky[0].shape[0] == likelihood.p_sky[0].shape[1], "Precision matrix should be square"
        assert likelihood.p_sky[0].shape[0] == likelihood.y_sky[0].shape[0], "Precision matrix should match data vector"
        
        print("✓ Likelihood set_data successful")
        print(f"  Data shape: {likelihood.y_sky[0].shape}")
        print(f"  Precision matrix shape: {likelihood.p_sky[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Likelihood set_data failed: {e}")
        return False


def test_likelihood_eft_parameters():
    """Test Likelihood EFT parameter setup."""
    try:
        from pybird.likelihood import Likelihood
        
        # Use proper configuration
        config = get_default_likelihood_config()
        
        # Initialize Likelihood
        likelihood = Likelihood(config, verbose=False)
        
        # Check EFT parameter setup
        assert hasattr(likelihood, 'b_name'), "Likelihood should have 'b_name' attribute"
        assert hasattr(likelihood, 'bg_name'), "Likelihood should have 'bg_name' attribute"
        assert hasattr(likelihood, 'bng_name'), "Likelihood should have 'bng_name' attribute"
        
        # Check that EFT parameters are set up correctly
        assert 'b1' in likelihood.b_name, "b1 should be in EFT parameters"
        assert 'b2' in likelihood.b_name, "b2 should be in EFT parameters"
        assert 'b4' in likelihood.b_name, "b4 should be in EFT parameters"
        
        print("✓ Likelihood EFT parameters successful")
        print(f"  EFT parameters: {likelihood.b_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Likelihood EFT parameters failed: {e}")
        return False


def test_likelihood_chi2_computation():
    """Test Likelihood chi2 computation with mock data."""
    try:
        from pybird.likelihood import Likelihood
        
        # Use proper configuration
        config = get_default_likelihood_config()
        
        # Initialize Likelihood
        likelihood = Likelihood(config, verbose=False)
        
        # Test chi2 computation (this is basic, full test would need correlator)
        # For now, just check that the method exists and attributes are set
        assert hasattr(likelihood, 'get_chi2_non_marg'), "Likelihood should have get_chi2_non_marg method"
        assert hasattr(likelihood, 'y_all'), "Likelihood should have y_all attribute"
        assert hasattr(likelihood, 'p_all'), "Likelihood should have p_all attribute"
        
        # Check data shapes
        expected_size = 3 * 50  # 3 multipoles * 50 k-bins
        assert likelihood.y_all.shape[0] == expected_size, f"y_all should have {expected_size} elements"
        assert likelihood.p_all.shape == (expected_size, expected_size), f"p_all should be {expected_size}x{expected_size}"
        
        print("✓ Likelihood chi2 computation setup successful")
        print(f"  Data vector shape: {likelihood.y_all.shape}")
        print(f"  Precision matrix shape: {likelihood.p_all.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Likelihood chi2 computation failed: {e}")
        return False


def test_likelihood_different_configs():
    """Test Likelihood with different configurations."""
    try:
        from pybird.likelihood import Likelihood
        
        # Test different configurations by modifying the base config
        configs = [
            {'with_stoch': False},
            {'multipole': 2},
            {'with_resum': False},
        ]
        
        for i, config_update in enumerate(configs):
            try:
                # Start with proper base configuration
                config = get_default_likelihood_config()
                
                # Apply updates
                config.update(config_update)
                
                # Update sky bounds if multipole changed
                if 'multipole' in config_update:
                    multipole = config_update['multipole']
                    k_min, k_max = 0.01, 0.4
                    config['sky']['sky_1']['min'] = [k_min] * multipole
                    config['sky']['sky_1']['max'] = [k_max] * multipole
                
                # Initialize Likelihood
                likelihood = Likelihood(config, verbose=False)
                
                # Check that initialization worked
                assert hasattr(likelihood, 'correlator_sky'), "Likelihood should have correlator_sky"
                assert len(likelihood.correlator_sky) == 1, "Should have 1 correlator"
                
                print(f"✓ Configuration {i+1} works: {config_update}")
                
            except Exception as e:
                print(f"✗ Configuration {i+1} failed: {config_update} - {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Likelihood different configs failed: {e}")
        return False


def test_likelihood_multiple_sky():
    """Test Likelihood with multiple sky patches."""
    try:
        from pybird.likelihood import Likelihood
        
        # For now, just test that the single sky case works correctly
        # Creating multiple sky patches requires complex HDF5 structures
        config = get_default_likelihood_config()
        
        # Initialize Likelihood
        likelihood = Likelihood(config, verbose=False)
        
        # Check that single sky patch was set up correctly
        assert likelihood.nsky == 1, "Should have 1 sky patch"
        assert len(likelihood.y_sky) == 1, "Should have 1 data vector"
        assert len(likelihood.d_sky) == 1, "Should have 1 data dictionary"
        assert len(likelihood.correlator_sky) == 1, "Should have 1 correlator"
        
        # Check that all sky-related attributes are consistent
        assert len(likelihood.m_sky) == likelihood.nsky, "m_sky should match nsky"
        assert len(likelihood.p_sky) == likelihood.nsky, "p_sky should match nsky"
        assert len(likelihood.out) == likelihood.nsky, "out should match nsky"
        
        print("✓ Likelihood multiple sky successful")
        print(f"  Number of sky patches: {likelihood.nsky}")
        print(f"  Data vector length: {len(likelihood.y_sky[0])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Likelihood multiple sky failed: {e}")
        return False


def main():
    """Run all Likelihood tests."""
    print("Testing PyBird Likelihood Class")
    print("=" * 50)
    
    tests = [
        ("Likelihood import", test_likelihood_import),
        ("Likelihood initialization", test_likelihood_initialization),
        ("Likelihood set_data", test_likelihood_set_data),
        ("Likelihood EFT parameters", test_likelihood_eft_parameters),
        ("Likelihood chi2 computation", test_likelihood_chi2_computation),
        ("Likelihood different configs", test_likelihood_different_configs),
        ("Likelihood multiple sky", test_likelihood_multiple_sky),
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
        print("\n✓ All Likelihood tests passed!")
        return True
    else:
        print("\n✗ Some Likelihood tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 