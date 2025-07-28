#!/usr/bin/env python3
"""
Test utilities for PyBird tests.
Provides mock data generators and common test helpers.
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class MockData:
    """Class to generate mock cosmological data for testing without CLASS dependency."""
    
    @staticmethod
    def create_simple_cosmology(nk: int = 1000, z: float = 0.57) -> Dict:
        """Create a simple cosmology setup without CLASS.
        
        Args:
            nk: Number of k points
            z: Redshift
            
        Returns:
            Dictionary with cosmology data
        """
        # PyBird requires min(kk) < 1e-4 and max(kk) > 0.5
        kk = np.logspace(-4.5, 0.8, nk)  # 10^-4.5 to 10^0.8 covers requirements
        
        # More realistic linear power spectrum (approximate LCDM)
        # Use a more sophisticated model that won't cause NaN issues
        pk_norm = 2000  # Normalization at k=0.1
        k_pivot = 0.1
        n_s = 0.965  # Spectral index
        
        # Power-law with exponential cutoff at high k
        pk_lin = pk_norm * (kk / k_pivot)**(n_s - 1) * np.exp(-kk**2 / (2 * k_pivot**2))
        
        # Add some small-scale power to avoid zero values
        pk_lin += 10 * (kk / k_pivot)**(-2) * np.exp(-kk / (10 * k_pivot))
        
        # Ensure no zero or negative values
        pk_lin = np.maximum(pk_lin, 1e-10)
        
        # Realistic growth factors and cosmological parameters
        D1 = 1.0
        f1 = 0.5  # Reasonable growth rate
        Omega0_m = 0.3
        H_ap = 1.0
        D_ap = 1.0
        
        return {
            'kk': kk,
            'pk_lin': pk_lin,
            'pk_lin_2': pk_lin,  # Add this key to avoid KeyError in Bird class
            'D': D1,
            'f': f1,
            'z': z,
            'Omega0_m': Omega0_m,
            'H': H_ap,
            'DA': D_ap
        }
    
    @staticmethod
    def create_mock_data_vector(nk: int = 77, multipoles: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Create mock data vector and covariance matrix.
        
        Args:
            nk: Number of k bins
            multipoles: Number of multipoles (0, 2, 4)
            
        Returns:
            Tuple of (data_vector, covariance_matrix)
        """
        # Create realistic power spectrum values
        k_centers = np.linspace(0.01, 0.4, nk)
        
        # Mock monopole, quadrupole, hexadecapole
        data = []
        for ell in [0, 2, 4][:multipoles]:
            if ell == 0:  # monopole
                pk = 1000 * (k_centers / 0.1)**(-1.5) * np.exp(-k_centers**2 / 0.1**2)
            elif ell == 2:  # quadrupole
                pk = 500 * (k_centers / 0.1)**(-1.5) * np.exp(-k_centers**2 / 0.1**2)
            elif ell == 4:  # hexadecapole
                pk = 100 * (k_centers / 0.1)**(-1.5) * np.exp(-k_centers**2 / 0.1**2)
            data.extend(pk)
        
        data = np.array(data)
        
        # Create a realistic covariance matrix
        n_total = len(data)
        cov = np.eye(n_total)
        
        # Add correlations within each multipole
        for i in range(multipoles):
            start_idx = i * nk
            end_idx = (i + 1) * nk
            for j in range(start_idx, end_idx):
                for k in range(start_idx, end_idx):
                    if j != k:
                        cov[j, k] = 0.1 * np.exp(-abs(j - k) / 5.0)
        
        # Scale by realistic errors
        errors = 0.1 * np.abs(data) + 100  # 10% relative error + floor
        cov = cov * np.outer(errors, errors)
        
        return data, cov

class TestHelpers:
    """Common test helper functions."""
    
    @staticmethod
    def check_output_shape(output: np.ndarray, expected_shape: Tuple[int, ...], 
                          name: str = "output") -> bool:
        """Check if output has expected shape.
        
        Args:
            output: Array to check
            expected_shape: Expected shape
            name: Name for error messages
            
        Returns:
            True if shape matches, False otherwise
        """
        if output.shape != expected_shape:
            print(f"✗ {name} shape mismatch: expected {expected_shape}, got {output.shape}")
            return False
        return True
    
    @staticmethod
    def check_no_nan_inf(output: np.ndarray, name: str = "output") -> bool:
        """Check if output contains NaN or infinite values.
        
        Args:
            output: Array to check
            name: Name for error messages
            
        Returns:
            True if no NaN/inf, False otherwise
        """
        if np.any(np.isnan(output)):
            print(f"✗ {name} contains NaN values")
            return False
        if np.any(np.isinf(output)):
            print(f"✗ {name} contains infinite values")
            return False
        return True
    
    @staticmethod
    def check_values_reasonable(output: np.ndarray, min_val: float = -1e10, 
                               max_val: float = 1e10, name: str = "output") -> bool:
        """Check if output values are within reasonable bounds.
        
        Args:
            output: Array to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error messages
            
        Returns:
            True if values are reasonable, False otherwise
        """
        if np.any(output < min_val) or np.any(output > max_val):
            print(f"✗ {name} values outside reasonable range [{min_val}, {max_val}]")
            print(f"  Range: [{output.min():.2e}, {output.max():.2e}]")
            return False
        return True
    
    @staticmethod
    def run_test(test_func, test_name: str, verbose: bool = True) -> bool:
        """Run a test function with error handling.
        
        Args:
            test_func: Test function to run
            test_name: Name of the test
            verbose: Whether to print verbose output
            
        Returns:
            True if test passed, False otherwise
        """
        if verbose:
            print(f"\n{test_name}:")
            print("-" * 30)
        
        try:
            result = test_func()
            if result:
                if verbose:
                    print(f"✓ {test_name} passed")
                return True
            else:
                if verbose:
                    print(f"✗ {test_name} failed")
                return False
        except Exception as e:
            if verbose:
                print(f"✗ {test_name} failed with exception: {e}")
            return False

def get_default_correlator_config(fast: bool = True) -> Dict:
    """Get default correlator configuration.
    
    Args:
        fast: Whether to use fast settings for testing
        
    Returns:
        Configuration dictionary
    """
    config = {
        'output': 'bPk',
        'multipole': 3,
        'kmax': 0.4,
        'z': 0.57,  # Add default redshift
        'with_resum': False,
        'with_exact_time': True,
        'with_time': False,
        'km': 0.7,
        'kr': 0.25,
        'nd': 0.0003,
        'eft_basis': 'eftoflss',
        'with_stoch': True, 
        'fftaccboost': 2,
    }
    
    return config

def get_default_eft_params() -> np.ndarray:
    """Get default EFT parameters for testing.
    
    Returns:
        Array of default EFT parameters
    """
    return np.array([2.0, 0.0, 0.0])  # b1, b2, b4

def get_default_bias_dict() -> Dict:
    """Get default bias parameters as dictionary for PyBird.
    
    Returns:
        Dictionary of bias parameters for PyBird
    """
    return {
        'b1': 2.0,
        'b2': 0.0,
        'b3': 0.0,
        'b4': 0.0,
        'bG2': 0.0,
        'bGamma3': 0.0,
        'c0': 0.0,
        'c2': 0.0,
        'c4': 0.0,
        'ctidal': 0.0,
        'k4': 0.0,
        'k6': 0.0,
        # EFT basis specific parameters
        'cct': 0.0,  # Counterterm for eftoflss/westcoast
        'cr1': 0.0,  # Redshift counterterm 1 for eftoflss/westcoast  
        'cr2': 0.0,  # Redshift counterterm 2 for eftoflss/westcoast
        'cr4': 0.0,  # NNLO counterterm
        'cr6': 0.0,  # NNLO counterterm
        'ct': 0.0,   # Counterterm for eastcoast
        # Stochastic terms
        'ce0': 0.0,  # Stochastic term 0
        'ce1': 0.0,  # Stochastic term 1
        'ce2': 0.0,  # Stochastic term 2
        'cemono': 0.0,
        'cequad': 0.0,
        # Halo-matter terms
        'dct': 0.0,  # Matter counterterm
        'dr1': 0.0,  # Matter redshift counterterm 1
        'dr2': 0.0,  # Matter redshift counterterm 2
        # Tidal alignment
        'bq': 0.0,   # Tidal alignment parameter
    }

def check_imports() -> Tuple[bool, bool, bool]:
    """Check which optional packages are available.
    
    Returns:
        Tuple of (jax_available, class_available, emulator_available)
    """
    jax_available = True
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        jax_available = False
    
    class_available = True
    try:
        from classy import Class
    except ImportError:
        class_available = False
    
    emulator_available = True
    try:
        from pybird import emulator
    except ImportError:
        emulator_available = False
    
    return jax_available, class_available, emulator_available


def get_default_likelihood_config() -> Dict:
    """Get default likelihood configuration for testing.
    
    Returns:
        Configuration dictionary for Likelihood class
    """
    import tempfile
    import h5py
    
    # Create temporary directory for mock data
    temp_dir = tempfile.mkdtemp()
    data_file = os.path.join(temp_dir, "mock_data.h5")
    
    # Create proper HDF5 data structure that PyBird expects
    nk = 50  # Smaller for testing
    k_min, k_max = 0.01, 0.4
    k_centers = np.linspace(k_min, k_max, nk)
    
    # Create mock power spectrum data for 3 multipoles
    pk_0 = 1000 * (k_centers / 0.1)**(-1.5) * np.exp(-k_centers**2 / 0.1**2)  # monopole
    pk_2 = 500 * (k_centers / 0.1)**(-1.5) * np.exp(-k_centers**2 / 0.1**2)   # quadrupole
    pk_4 = 100 * (k_centers / 0.1)**(-1.5) * np.exp(-k_centers**2 / 0.1**2)   # hexadecapole
    
    # Create covariance matrix
    n_total = 3 * nk  # 3 multipoles
    cov = np.eye(n_total) * 0.1  # 10% error
    
    # Add some realistic correlations
    for i in range(3):
        start_idx = i * nk
        end_idx = (i + 1) * nk
        for j in range(start_idx, end_idx):
            for k in range(start_idx, end_idx):
                if j != k:
                    cov[j, k] = 0.05 * np.exp(-abs(j - k) / 5.0)
    
    # Scale by power spectrum values
    errors = 0.1 * np.concatenate([pk_0, pk_2, pk_4]) + 100  # Relative + floor
    cov = cov * np.outer(errors, errors)
    
    # Create HDF5 file with proper structure
    with h5py.File(data_file, 'w') as f:
        # Create sky group
        sky_group = f.create_group('sky_1')
        
        # Redshift information
        z_group = sky_group.create_group('z')
        z_group.create_dataset('min', data=0.5)
        z_group.create_dataset('max', data=0.65)
        z_group.create_dataset('eff', data=0.57)
        
        # Fiducial cosmology
        fid_group = sky_group.create_group('fid')
        fid_group.create_dataset('Omega_m', data=0.3)
        fid_group.create_dataset('H', data=1.0)
        fid_group.create_dataset('D', data=1.0)
        
        # Power spectrum data
        bpk_group = sky_group.create_group('bPk')
        bpk_group.create_dataset('multipole', data=3)
        bpk_group.create_dataset('x', data=k_centers)
        bpk_group.create_dataset('l0', data=pk_0)
        bpk_group.create_dataset('l2', data=pk_2)
        bpk_group.create_dataset('l4', data=pk_4)
        bpk_group.create_dataset('cov', data=cov)
        bpk_group.create_dataset('nsims', data=1000)
    
    # Configuration dictionary
    config = {
        'output': 'bPk',
        'data_path': temp_dir,
        'data_file': 'mock_data.h5',
        'sky': {
            'sky_1': {
                'min': [k_min, k_min, k_min],
                'max': [k_max, k_max, k_max]
            }
        },
        'multipole': 3,
        'with_bias': True,
        'eft_basis': 'eftoflss',
        'with_stoch': True,
        'with_nnlo_counterterm': False,
        'with_tidal_alignments': False,
        'with_bao_rec': False,
        'with_ap': False,
        'with_survey_mask': False,
        'with_binning': False,
        'with_wedge': False,
        'with_redshift_bin': False,
        'with_RSD': True,
        'with_loop_prior': False,
        'with_exact_time': True,
        'with_time': False,
        'with_resum': True,
        'with_emu': False,
        'km': 0.7,
        'kr': 0.25,
        'nd': 0.0003,
        'fftaccboost': 2,
        'eft_prior': {
            'b1': {'type': 'gauss', 'mean': [2.0], 'range': [0.5]},
            'b2': {'type': 'gauss', 'mean': [0.0], 'range': [1.0]},
            'b4': {'type': 'gauss', 'mean': [0.0], 'range': [1.0]},
            'cct': {'type': 'gauss', 'mean': [0.0], 'range': [1.0]},
            'cr1': {'type': 'gauss', 'mean': [0.0], 'range': [1.0]},
            'cr2': {'type': 'gauss', 'mean': [0.0], 'range': [1.0]},
        },
        'write': {
            'save': False,
            'plot': False,
            'fake': False,
            'show': False,
            'out_path': temp_dir,
            'out_name': 'test'
        }
    }
    
    return config


def main():
    """Run utility tests."""
    print("Testing PyBird Test Utilities")
    print("=" * 50)
    
    # Test mock data creation
    print("\nTesting mock data creation:")
    try:
        # Test simple cosmology
        cosmo = MockData.create_simple_cosmology(nk=100)
        print(f"✓ Simple cosmology created: {len(cosmo['kk'])} k-points")
        
        # Test mock data vector
        data, cov = MockData.create_mock_data_vector(nk=50, multipoles=3)
        print(f"✓ Mock data vector created: {len(data)} data points")
        
        # Test likelihood config
        config = get_default_likelihood_config()
        print(f"✓ Likelihood config created: {len(config)} parameters")
        
        print("\n✓ All utility tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Utility tests failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)