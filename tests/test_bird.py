#!/usr/bin/env python3
"""
Test script for PyBird Bird class.
Tests core functionality of the Bird class with mock data for speed.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_utils import MockData, TestHelpers, get_default_correlator_config, get_default_eft_params, get_default_bias_dict, check_imports


def test_bird_import():
    """Test if Bird class can be imported."""
    try:
        from pybird.bird import Bird
        print("✓ Bird class imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Bird class: {e}")
        return False


def test_bird_initialization():
    """Test Bird class initialization."""
    try:
        from pybird.bird import Bird
        
        # Initialize Bird object
        bird = Bird()
        
        # Check that basic attributes exist
        assert hasattr(bird, 'co'), "Bird should have 'co' attribute"
        assert hasattr(bird, 'with_bias'), "Bird should have 'with_bias' attribute"
        assert hasattr(bird, 'eft_basis'), "Bird should have 'eft_basis' attribute"
        
        print("✓ Bird initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ Bird initialization failed: {e}")
        return False


def test_bird_setcosmo():
    """Test Bird setcosmo method with mock data."""
    try:
        from pybird.bird import Bird
        from pybird.common import co
        
        # Create mock cosmology
        cosmo_data = MockData.create_simple_cosmology(nk=500)  # Smaller for speed
        
        # Initialize Bird
        bird = Bird()
        
        # Set basic configuration in common object
        co.with_bias = True
        co.eft_basis = 'eftoflss'
        co.with_stoch = True
        co.with_nnlo_counterterm = False
        co.with_tidal_alignments = False
        co.z = cosmo_data['z']
        co.D = cosmo_data['D']
        co.f = cosmo_data['f']
        co.Omega0_m = cosmo_data['Omega0_m']
        co.H = cosmo_data['H']
        co.DA = cosmo_data['DA']
        
        # Test setcosmo
        bird.setcosmo(cosmo_data)
        
        # Check that cosmology was set
        assert hasattr(bird, 'kin'), "Bird should have 'kin' attribute after setcosmo"
        assert hasattr(bird, 'Pin'), "Bird should have 'Pin' attribute after setcosmo"
        assert hasattr(bird, 'Plin'), "Bird should have 'Plin' attribute after setcosmo"
        assert hasattr(bird, 'P11'), "Bird should have 'P11' attribute after setcosmo"
        
        # Check array shapes
        assert bird.kin.shape == cosmo_data['kk'].shape, "kin should match input k array"
        assert bird.Pin.shape == cosmo_data['pk_lin'].shape, "Pin should match input pk_lin"
        
        # Check that linear power spectrum was computed
        # P11 is 1D array with k bins (not organized by multipoles)
        assert len(bird.P11.shape) == 1, "P11 should be 1D array"
        assert bird.P11.shape[0] > 0, "P11 should have k bins"
        
        # Check for reasonable values
        assert TestHelpers.check_no_nan_inf(bird.P11, "P11")
        assert TestHelpers.check_values_reasonable(bird.P11, min_val=0, max_val=1e10, name="P11")
        
        print("✓ Bird setcosmo successful")
        print(f"  P11 shape: {bird.P11.shape}")
        print(f"  P11 range: [{bird.P11.min():.2e}, {bird.P11.max():.2e}]")
        print(f"  co.Nl (multipoles): {co.Nl}")
        
        return True
        
    except Exception as e:
        print(f"✗ Bird setcosmo failed: {e}")
        return False


def test_bird_setbias():
    """Test Bird setBias method."""
    try:
        from pybird.bird import Bird
        from pybird.common import co
        
        # Create mock cosmology and initialize Bird
        cosmo_data = MockData.create_simple_cosmology(nk=500)
        bird = Bird()
        
        # Set basic configuration
        co.with_bias = True
        co.eft_basis = 'eftoflss'
        co.with_stoch = True
        co.Nloop = 112  # Standard number of loop terms
        co.l = 3  # Number of multipoles
        co.z = cosmo_data['z']
        co.D = cosmo_data['D']
        co.f = cosmo_data['f']
        co.Omega0_m = cosmo_data['Omega0_m']
        co.H = cosmo_data['H']
        co.DA = cosmo_data['DA']
        
        # Set cosmology first
        bird.setcosmo(cosmo_data)
        
        # Test setBias with proper bias dictionary
        bias_params = get_default_bias_dict()
        bird.setBias(bias_params)
        
        # Check that bias matrices were created
        assert hasattr(bird, 'b11'), "Bird should have 'b11' attribute after setBias"
        assert hasattr(bird, 'b13'), "Bird should have 'b13' attribute after setBias"
        assert hasattr(bird, 'b22'), "Bird should have 'b22' attribute after setBias"
        assert hasattr(bird, 'bct'), "Bird should have 'bct' attribute after setBias"
        
        # Check shapes - the bias matrices should have co.Nl elements (one per multipole)
        assert bird.b11.shape[0] == co.Nl, f"b11 should have {co.Nl} multipoles, got {bird.b11.shape[0]}"
        assert bird.b13.shape[0] == co.Nl, f"b13 should have {co.Nl} multipoles, got {bird.b13.shape[0]}"  
        assert bird.b22.shape[0] == co.Nl, f"b22 should have {co.Nl} multipoles, got {bird.b22.shape[0]}"
        assert bird.bct.shape[0] == co.Nl, f"bct should have {co.Nl} multipoles, got {bird.bct.shape[0]}"
        
        # Check for reasonable values
        assert TestHelpers.check_no_nan_inf(bird.b11, "b11")
        assert TestHelpers.check_no_nan_inf(bird.b13, "b13")
        assert TestHelpers.check_no_nan_inf(bird.b22, "b22")
        assert TestHelpers.check_no_nan_inf(bird.bct, "bct")
        
        print("✓ Bird setBias successful")
        print(f"  b11 shape: {bird.b11.shape}")
        print(f"  b13 shape: {bird.b13.shape}")
        print(f"  b22 shape: {bird.b22.shape}")
        print(f"  bct shape: {bird.bct.shape}")
        print(f"  co.Nl: {co.Nl}")
        
        return True
        
    except Exception as e:
        print(f"✗ Bird setBias failed: {e}")
        return False


def test_bird_setps():
    """Test Bird setPs method."""
    try:
        from pybird.bird import Bird
        from pybird.common import co
        
        # Create mock cosmology and initialize Bird
        cosmo_data = MockData.create_simple_cosmology(nk=500)
        bird = Bird()
        
        # Set basic configuration
        co.with_bias = True
        co.eft_basis = 'eftoflss'
        co.with_stoch = True
        co.Nloop = 112
        co.l = 3
        co.z = cosmo_data['z']
        co.D = cosmo_data['D']
        co.f = cosmo_data['f']
        co.Omega0_m = cosmo_data['Omega0_m']
        co.H = cosmo_data['H']
        co.DA = cosmo_data['DA']
        
        # Set cosmology and bias
        bird.setcosmo(cosmo_data)
        bias_params = get_default_bias_dict()
        bird.setBias(bias_params)
        
        # Create some mock loop terms for testing
        # P22 and P13 should have shape (co.N22, co.Nk) and (co.N13, co.Nk)
        print(f"DEBUG: co.N22: {co.N22 if hasattr(co, 'N22') else 'not set'}")
        print(f"DEBUG: co.N13: {co.N13 if hasattr(co, 'N13') else 'not set'}")
        print(f"DEBUG: co.Nk: {co.Nk if hasattr(co, 'Nk') else 'not set'}")
        
        if hasattr(co, 'N22') and hasattr(co, 'N13') and hasattr(co, 'Nk'):
            bird.P22 = np.random.rand(co.N22, co.Nk) * 1000
            bird.P13 = np.random.rand(co.N13, co.Nk) * 1000
        else:
            # Fallback: use reasonable dimensions
            bird.P22 = np.random.rand(28, len(co.k)) * 1000  # Approximate N22
            bird.P13 = np.random.rand(15, len(co.k)) * 1000  # Approximate N13
        
        # Test setPs with bias parameters
        bird.setPs(bias_params)
        
        # Check that power spectrum was computed
        assert hasattr(bird, 'Ps'), "Bird should have 'Ps' attribute after setPs"
        
        # Ps should be array with shape (n_orders, co.Nl, co.Nk)
        print(f"DEBUG: Ps shape: {bird.Ps.shape}")
        assert len(bird.Ps.shape) == 3, "Ps should be 3D array (orders, multipoles, k)"
        assert bird.Ps.shape[1] == co.Nl, f"Ps should have {co.Nl} multipoles"
        assert bird.Ps.shape[2] > 0, "Ps should have k bins"
        
        # Check for reasonable values
        assert TestHelpers.check_no_nan_inf(bird.Ps, "Ps")
        assert TestHelpers.check_values_reasonable(bird.Ps, min_val=-1e10, max_val=1e10, name="Ps")
        
        print("✓ Bird setPs successful")
        print(f"  Ps shape: {bird.Ps.shape}")
        print(f"  Ps range: [{bird.Ps.min():.2e}, {bird.Ps.max():.2e}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Bird setPs failed: {e}")
        return False


def test_bird_setfullps():
    """Test Bird setfullPs method."""
    try:
        from pybird.bird import Bird
        from pybird.common import co
        
        # Create mock cosmology and initialize Bird
        cosmo_data = MockData.create_simple_cosmology(nk=500)
        bird = Bird()
        
        # Set basic configuration
        co.with_bias = True
        co.eft_basis = 'eftoflss'
        co.with_stoch = True
        co.Nloop = 112
        co.l = 3
        co.z = cosmo_data['z']
        co.D = cosmo_data['D']
        co.f = cosmo_data['f']
        co.Omega0_m = cosmo_data['Omega0_m']
        co.H = cosmo_data['H']
        co.DA = cosmo_data['DA']
        
        # Set cosmology and bias
        bird.setcosmo(cosmo_data)
        bias_params = get_default_bias_dict()
        bird.setBias(bias_params)
        
        # Create some mock loop terms
        if hasattr(co, 'N22') and hasattr(co, 'N13') and hasattr(co, 'Nk'):
            bird.P22 = np.random.rand(co.N22, co.Nk) * 1000
            bird.P13 = np.random.rand(co.N13, co.Nk) * 1000
        else:
            bird.P22 = np.random.rand(28, len(co.k)) * 1000
            bird.P13 = np.random.rand(15, len(co.k)) * 1000
        
        # Set power spectrum
        bird.setPs(bias_params)
        
        # Test setfullPs
        bird.setfullPs()
        
        # Check that full power spectrum was computed
        assert hasattr(bird, 'fullPs'), "Bird should have 'fullPs' attribute after setfullPs"
        
        # fullPs should have shape (co.Nl, co.Nk) 
        print(f"DEBUG: fullPs shape: {bird.fullPs.shape}")
        assert bird.fullPs.shape[0] == co.Nl, f"fullPs should have {co.Nl} multipoles"
        assert bird.fullPs.shape[1] > 0, "fullPs should have k bins"
        
        # Check for reasonable values
        assert TestHelpers.check_no_nan_inf(bird.fullPs, "fullPs")
        assert TestHelpers.check_values_reasonable(bird.fullPs, min_val=-1e10, max_val=1e10, name="fullPs")
        
        print("✓ Bird setfullPs successful")
        print(f"  fullPs shape: {bird.fullPs.shape}")
        print(f"  fullPs range: [{bird.fullPs.min():.2e}, {bird.fullPs.max():.2e}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Bird setfullPs failed: {e}")
        return False


def test_bird_different_eft_basis():
    """Test Bird with different EFT basis options."""
    try:
        from pybird.bird import Bird
        from pybird.common import co
        
        # Test different EFT basis options
        basis_options = ['eftoflss', 'westcoast', 'eastcoast']
        
        for basis in basis_options:
            try:
                # Create mock cosmology and initialize Bird
                cosmo_data = MockData.create_simple_cosmology(nk=300)  # Smaller for speed
                bird = Bird()
                
                # Set configuration
                co.with_bias = True
                co.eft_basis = basis
                co.with_stoch = True
                co.Nloop = 112
                co.l = 3
                co.z = cosmo_data['z']
                co.D = cosmo_data['D']
                co.f = cosmo_data['f']
                co.Omega0_m = cosmo_data['Omega0_m']
                co.H = cosmo_data['H']
                co.DA = cosmo_data['DA']
                
                # Set cosmology and bias
                bird.setcosmo(cosmo_data)
                bias_params = get_default_bias_dict()
                bird.setBias(bias_params)
                
                print(f"✓ EFT basis '{basis}' works")
                
            except Exception as e:
                print(f"✗ EFT basis '{basis}' failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Bird EFT basis test failed: {e}")
        return False


def test_bird_jax_compatibility():
    """Test Bird JAX compatibility if available."""
    jax_available, _, _ = check_imports()
    
    if not jax_available:
        print("⚠ JAX not available, skipping JAX compatibility test")
        return True
    
    try:
        import jax
        import jax.numpy as jnp
        from pybird.bird import Bird
        from pybird.common import co
        
        # JAX mode is controlled by test runner - don't override here
        
        # Create mock cosmology with JAX arrays
        cosmo_data = MockData.create_simple_cosmology(nk=300)
        cosmo_data['kk'] = jnp.array(cosmo_data['kk'])
        cosmo_data['pk_lin'] = jnp.array(cosmo_data['pk_lin'])
        
        # Initialize Bird
        bird = Bird()
        
        # Set configuration
        co.with_bias = True
        co.eft_basis = 'eftoflss'
        co.with_stoch = True
        co.Nloop = 112
        co.l = 3
        co.z = cosmo_data['z']
        co.D = cosmo_data['D']
        co.f = cosmo_data['f']
        co.Omega0_m = cosmo_data['Omega0_m']
        co.H = cosmo_data['H']
        co.DA = cosmo_data['DA']
        
        # Test with JAX arrays
        bird.setcosmo(cosmo_data)
        
        # Check that JAX arrays work
        assert hasattr(bird, 'kin'), "Bird should work with JAX arrays"
        assert hasattr(bird, 'Pin'), "Bird should work with JAX arrays"
        
        print("✓ Bird JAX compatibility successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Bird JAX compatibility failed: {e}")
        return False


def main():
    """Run all Bird tests."""
    print("Testing PyBird Bird Class")
    print("=" * 50)
    
    tests = [
        ("Bird import", test_bird_import),
        ("Bird initialization", test_bird_initialization),
        ("Bird setcosmo", test_bird_setcosmo),
        ("Bird setBias", test_bird_setbias),
        ("Bird setPs", test_bird_setps),
        ("Bird setfullPs", test_bird_setfullps),
        ("Bird EFT basis", test_bird_different_eft_basis),
        ("Bird JAX compatibility", test_bird_jax_compatibility),
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
        print("\n✓ All Bird tests passed!")
        return True
    else:
        print("\n✗ Some Bird tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 