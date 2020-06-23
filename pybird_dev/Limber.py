import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
from scipy.special import legendre
from scipy.integrate import quad, dblquad, simps
from common import co
import scipy.constants as conts

class Limber(object):

    def __init__(self, theta, z, nzlens, nzsource, NFFT=256):

        self.theta = theta
        self.z = z
        self.t1, self.z1 = np.meshgrid(theta, z, indexing='ij')
        self.nlens = self.interp(self.z, self.z1, nzlens)
        self.nsource = self.interp(self.z, self.z1, nzsource)
        
        self.fftsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1.e3, bias=-1.6)
        self.fft = FFTLog(**self.fftsettings)
        self.setsPow()
        self.setM()
        
    def setsPow(self):
        
        slog = np.geomspace(1e-4, 3., 40)
        slin = np.arange(3, 200., 1)
        slog2 = np.geomspace(200, 1e4, 20)
        self.s = np.unique(np.concatenate([slog, slin, slog2, [1e5]]))

        self.sPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3. + 0.5, log(self.s)))
    
    def setM(self):
        self.M = np.empty(shape=(3, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(3): self.M[l] = (2*pi)**.5 * MPC(2*l-.5, -0.5 * self.fft.Pow)
    
    def interp(self, z1d, z1, func):
        ifunc = interp1d(z1d, func, axis=-1, kind='cubic')
        return ifunc(z1)
    
    def lensing_efficiency(self, z, rz, nz):
        r1, z2 = np.meshgrid(rz, z, indexing='ij')
        r2 = self.interp(z, z2, rz)
        n2 = self.interp(z, z2, nz)
        return self.lensing_factor * rz * (1+z) * np.trapz(np.heaviside(r2-r1, 0.) * n2 * (r2-r1)/r2, x=z, axis=-1)

    def Xi(self, kin, Pin, rz, dz_by_dr, Dz, Dfid, h, Omega0_m):

        self.lensing_factor = 1.5/conts.c**2 * h**2 * 1e10 * Omega0_m 
        
        Cf = np.empty(shape=(3,self.s.shape[0]))
        Coef = self.fft.Coef(kin, kin**-0.5 * Pin, extrap='extrap', window=None)
        CoefsPow = np.einsum('n,nt->nt', Coef, self.sPow)
        Cf = np.real(np.einsum('ns,ln->ls', CoefsPow, self.M))
        Cf[:,-1] = np.zeros((3))
        Cf = np.vstack([Cf, Cf[0]])
        
        r1 = self.interp(self.z, self.z, rz)
        qg = self.interp(rz, r1, self.lensing_efficiency(self.z, rz, self.nsource))
        qd = self.interp(self.z, self.z1, dz_by_dr) * self.nlens
        
        qq = np.array([qd**2, qg*qd, qg**2, qg**2])
        Cf1 = interp1d(self.s, Cf, kind='cubic', axis=-1)(self.t1 * r1)
        
        return np.trapz(qq * Dz**2 / Dfid**2 * Cf1, x=rz, axis=-1)
    