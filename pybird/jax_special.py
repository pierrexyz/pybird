from jax.numpy import swapaxes
from interpax import Interpolator1D

"""
JAX-compatible special functions
Syntax chosen such as it matches scipy    
"""

def legendre(n): 
    """From Mathematica"""
    match n:
        case 0: return lambda x: x**0.
        case 1: return lambda x: x
        case 2: return lambda x: (-1 + 3*x**2)/2.
        case 3: return lambda x: (-3*x + 5*x**3)/2.
        case 4: return lambda x: (3 - 30*x**2 + 35*x**4)/8.
        case 5: return lambda x: (15*x - 70*x**3 + 63*x**5)/8.
        case 6: return lambda x: (-5 + 105*x**2 - 315*x**4 + 231*x**6)/16.
        case 7: return lambda x: (-35*x + 315*x**3 - 693*x**5 + 429*x**7)/16.
        case 8: return lambda x: (35 - 1260*x**2 + 6930*x**4 - 12012*x**6 + 6435*x**8)/128.
        case _: raise Exception('Legendre of order >= %s not implemented' % n)

        
class Interpolator1D_():
    def __init__(self, *args, interp_axis=0, **kwargs):
        self.interp_axis = interp_axis
        self.I1D = Interpolator1D(*args, **kwargs)
    def __call__(self, *args, **kwargs):
        result = self.I1D.__call__(*args, **kwargs)
        return swapaxes(result, 0, self.interp_axis)
        
def interp1d(x, f, axis=0, kind='cubic', bounds_error=False, fill_value='extrapolate'):
    """See https://interpax.readthedocs.io/en/latest/api.html"""
    
    if bounds_error == False: 
        extrap = False
    else: 
        if fill_value == 'extrapolate': extrap = True
        elif isinstance(fill_value, (int,float)): extrap = fill_value
        else: raise Exception('No support for fill_value = %s' % fill_value)
        
    if axis != 0: f_ = swapaxes(f, 0, axis)
    else: f_ = 1. * f
    
    return Interpolator1D_(x, f_, method=kind, extrap=extrap, interp_axis=axis) 
    
    





