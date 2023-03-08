import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from scipy.special import gamma
from pybird.fftlog import FFTLog, MPC, CoefWindow
from pybird.common import co

class NNLO_counterterm(object): # k^4 P11

    def __init__(self, load=True, save=True, path='./', NFFT=256, co=co):

        self.co = co
        self.fftsettings = dict(Nmax=NFFT, xmin=1.e-3, xmax=1.e4, bias=.01)
        self.fft = FFTLog(**self.fftsettings)
        self.setMcf()
        self.setsPow()
        self.kdeep = np.logspace(-3, 4, 400)
        self.smask_out = np.where(self.co.s>50.)[0]
        self.smask_in = np.where(self.co.s<=50.)[0]
        self.swin = CoefWindow(len(self.smask_in)-1, window=.2, left=False, right=True)

    def setMcf(self):
        """ Compute the next-to-next-to-leading counterterm correlation function matrices. """
        self.Mcf = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                self.Mcf[l, u] = 1j**(2*l) * MPC(2 * l, n1)

    def setsPow(self):
        self.sPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3., log(self.co.s)))

    def CoefsPow(self, coef):
        return np.einsum( 'n,ns->ns', coef, self.sPow )

    def Cf(self, bird, PEH_interp):
        k4PEH = self.kdeep**4/(1.+self.kdeep**4/self.co.km**4) * exp(PEH_interp(log(self.kdeep)))
        coef = self.fft.Coef(self.kdeep, k4PEH)
        bird.Cnnlo = np.real(np.einsum('ns,ln->ls', self.CoefsPow(coef), self.Mcf)) 
        bird.Cnnlo[:,self.smask_out] = 0.
        bird.Cnnlo[:,self.smask_in] *= self.swin

    def Ps(self, bird, PEH_interp):
        bird.Pnnlo = self.co.k**4 * exp(PEH_interp(log(self.co.k)))

class NNLO_higher_derivative(object): # k^2 P1Loop

    def __init__(self, xdata, with_cf=False, NFFT=256, co=co):

        self.co = co
        self.fftsettings = dict(Nmax=NFFT, xmin=1.e-3, xmax=1.e3, bias=.01) 
        self.fft = FFTLog(**self.fftsettings)
        if with_cf:
            self.setM()
            self.setsPow(xdata)
            self.kdeep = np.geomspace(self.co.k[0], 1.e3, 400) 
            self.kmask = np.where(self.co.k < 0.25)[0]  # < 0.25 is the best choice to FT[1loop]: less, we are cutting physical signal ; more, we are starting to add unphysical junks from the loop divergence : one will see appearing spurious ringing if using higher k's
            self.smask_out = np.where(xdata>75.)[0]
            self.smask_in = np.where(xdata<=75.)[0]
            self.swin = CoefWindow(len(self.smask_in)-1, window=1, left=False, right=True)
        else:
            self.k = xdata

    def setsPow(self, s):
        """ Multiply the coefficients with the s's to the powers of the FFTLog. """
        self.sPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3., log(s))) 

    def setM(self):
        """ Compute the matrices of the spherical-Bessel transform from Ps to Cf. Called at instantiation. """
        self.M = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl): self.M[l] = 1j**(2*l) * MPC(2*l, -0.5*self.fft.Pow)

    def FT(self, Ps, l=0, window=None):
        """ Compute the spherical Bessel transform from Ps to Cf"""
        Coef = self.fft.Coef(self.kdeep, Ps, extrap='padding', window=window)
        CoefsPow = np.einsum('n,ns->ns', Coef, self.sPow)
        return np.real(np.einsum('ns,n->s', CoefsPow, self.M[l]))

    def Ps2Cf(self, Ps):
        Ps_interp = interp1d(np.log(self.co.k[self.kmask]), np.log( Ps[:,self.kmask] ), fill_value='extrapolate', axis=-1)
        k2Ps = self.kdeep**2  / self.co.km**2 / (1.+ self.kdeep**2 / self.co.km**2) * np.exp( Ps_interp(np.log(self.kdeep)) )
        return np.array([ self.FT( k2Ps[i],  l=i, window=.2) for i in range(self.co.Nl) ])

    def Cf(self, bird):
        cfnnlo = self.Ps2Cf(bird.Ps[0] + bird.Ps[1]) - self.Ps2Cf(bird.Ps[0]) # we do this way because the loop is negative for s <~ 0.15, so the log() doesn't like it
        cfnnlo[:,self.smask_out] = 0. # we make sure to kill any residual signal above 75: we don't want the spurious stuff around the BAO for example
        cfnnlo[:,self.smask_in] *= self.swin
        if np.isnan(np.sum(cfnnlo)): return np.zeros_like(cfnnlo)
        else: return cfnnlo

    def Ps(self, bird):
        return self.k**2 / self.co.km**2 / (1. + self.k**2/self.co.km**2) * bird.Ps[1]
