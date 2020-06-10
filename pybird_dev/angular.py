import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
from scipy.special import legendre
from scipy.integrate import quad, dblquad, simps
from common import co

class Angular(object):

    def __init__(self, theta, co=co):
        self.co = co
        self.theta = theta

    def mesheval1d(self, z1d, zm, func):
        ifunc = interp1d(z1d, func, axis=-1, kind='cubic')
        return ifunc(zm)

    def mesheval2d(self, z1d, z1, z2, func):
        ifunc = interp1d(z1d, func, axis=-1, kind='cubic')
        return ifunc(z1), ifunc(z2)

    def w(self, bird, Dz, fz, rz, z, nz):

        x, z1, z2 = np.meshgrid(self.theta, z, z, indexing='ij')

        n1, n2 = self.mesheval2d(z, z1, z2, nz)
        np2 = n1 * n2

        zm = 0.5 * (z1 + z2)
        Dm = self.mesheval1d(z, zm, Dz/bird.D)
        fm = self.mesheval1d(z, zm, fz/bird.f)
        
        Dp2 = Dm**2
        Dp4 = Dp2**2
        fp0 = fm**0
        fp1 = fm
        fp2 = fm**2
        fp3 = fp2*fm
        fp4 = fp2**2

        f11 = np.array([fp0, fp1, fp2])
        fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
        floop = np.array([fp2, fp3, fp4, fp1, fp2, fp3, fp1, fp2, fp1, fp1, fp2, fp0, fp1, fp2, fp0, fp1, fp0, fp0, fp1, fp0, fp0, fp0])

        tlin = np.einsum('n...,...->n...', f11, Dp2 * np2)
        tct = np.einsum('n...,...->n...', fct, Dp2 * np2)
        tloop = np.einsum('n...,...->n...', floop, Dp4 * np2)

        # elif 'geom' in self.co.redshift:
        #     D1, D2 = self.mesheval2d(z, z1, z2, Dz/bird.D)
        #     f1, f2 = self.mesheval2d(z, z1, z2, fz) 

        #     Dp2 = D1 * D2
        #     Dp22 = Dp2 * Dp2
        #     Dp13 = 0.5 * (D1**2 + D2**2) * Dp2
        #     fp0 = f1**0
        #     fp1 = 0.5 * (f1 + f2)
        #     fp2 = f1 * f2
        #     fp3 = fp1 * fp2
        #     fp4 = fp2 * fp2
            
        #     f11 = np.array([fp0, fp1, fp2])
        #     fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
        #     floop = np.concatenate([6*[fp0], 6*[fp1], 9*[fp2], 4*[fp3], 3*[fp4], 2*[fp0],  3*[fp1], 3*[fp2], 2*[fp3]])
            
        #     tlin = np.einsum('n...,...->n...', f11, Dp2 * np2)
        #     tct = np.einsum('n...,...->n...', fct, Dp2 * np2)
        #     tloop = np.empty_like(floop)
        #     tloop[:self.co.N22] = np.einsum('n...,...->n...', floop[:self.co.N22], Dp22 * np2)
        #     tloop[self.co.N22:] = np.einsum('n...,...->n...', floop[self.co.N22:], Dp13 * np2)
        
        r1, r2 = self.mesheval2d(z, z1, z2, rz)
        s = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(x))
        mu = (r2-r1)/s

        L = np.array([legendre(2*l)(mu) for l in range(self.co.Nl)])
        
        C11l = interp1d(self.co.s, bird.C11l, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(s)
        C11 = np.einsum('l...,ln...,n...->n...', L, C11l, tlin)

        Cctl = interp1d(self.co.s, bird.Cctl, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(s)
        Cct = np.einsum('l...,ln...,n...->n...', L, Cctl, tct)

        Cloopl = interp1d(self.co.s, bird.Cloopl, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(s)
        Cloop = np.einsum('l...,ln...,n...->n...', L, Cloopl, tloop)

        bird.wlin = np.trapz(np.trapz(C11, x=z2, axis=-1), x=z, axis=-1)
        bird.wct = np.trapz(np.trapz(Cct, x=z2, axis=-1), x=z, axis=-1)
        bird.wloop = np.trapz(np.trapz(Cloop, x=z2, axis=-1), x=z, axis=-1)

        if bird.with_nlo_bias:
            Cnlol = interp1d(self.co.s, bird.Cnlol, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(s)
            Cnlo = np.einsum('l...,l...,...->...', L, Cnlol, Dp2 * np2)
            bird.wnlo = np.trapz(np.trapz(Cnlo, x=z2, axis=-1), x=z, axis=-1)
