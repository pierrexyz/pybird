import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from pybird.fftlog import FFTLog, MPC, CoefWindow
from pybird.common import co

class FourierTransform(object):

    def __init__(self, s=None, NFFT=512, co=co):

        self.co = co
        
        self.fftsettings = dict(Nmax=NFFT, xmin=.1, xmax=10000., bias=-1.6)
        self.fft = FFTLog(**self.fftsettings)
        self.setM()
        self.setkPow()

        self.fftPs2Cfsettings = dict(Nmax=NFFT, xmin=1.e-3, xmax=1.e5, bias=.01) 
        #self.fftPs2Cfsettings = dict(Nmax=NFFT, xmin=1.5e-3, xmax=100., bias=-.6)
        self.fftPs2Cf = FFTLog(**self.fftPs2Cfsettings)
        self.setMPs2Cf()
        self.setsPow(s=s)

    def setkPow(self):
        """ Multiply the coefficients with the k's to the powers of the FFTLog. """
        self.kPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3., log(self.co.k))) 

    def setM(self):
        """ Compute the matrices of the spherical-Bessel transform from Cf to Ps. Called at instantiation. """
        self.M = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl): self.M[l] = 8.*pi**3 * (-1j)**(2*l) * MPC(2 * l, -0.5 * self.fft.Pow)

    def FT_Cf2Ps(self, Cf, l=0, window=None):
        """ Compute the spherical Bessel transform from Cf to Ps"""
        Coef = self.fft.Coef(self.co.s, Cf, extrap='padding', window=window)
        CoefkPow = np.einsum('n,nk->nk', Coef, self.kPow)
        return np.real(np.einsum('nk,n->k', CoefkPow, self.M[l]))

    def Cf2Ps(self, bird):
        for l in range(self.co.Nl):
            for i in range(self.co.N11): bird.P11l[l,i] = self.FT_Cf2Ps(bird.C11l[l,i], l)
            for i in range(self.co.Nct): bird.Pctl[l,i] = self.FT_Cf2Ps(bird.Cctl[l,i], l)
            for i in range(self.co.Nloop): bird.Ploopl[l,i] = self.FT_Cf2Ps(bird.Cloopl[l,i], l)

    def setsPow(self, s=None):
        """ Multiply the coefficients with the s's to the powers of the FFTLog. """
        if s is None: ss = self.co.s
        else: ss = s 
        self.sPow = exp(np.einsum('n,s->ns', -self.fftPs2Cf.Pow - 3., log(ss))) 

    def setMPs2Cf(self):
        """ Compute the matrices of the spherical-Bessel transform from Ps to Cf. Called at instantiation. """
        self.MPs2Cf = np.empty(shape=(self.co.Nl, self.fftPs2Cf.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl): self.MPs2Cf[l] = 1j**(2*l) * MPC(2*l, -0.5*self.fftPs2Cf.Pow)

    def FT_Ps2Cf(self, Ps, k=None, l=0, window=None):
        """ Compute the spherical Bessel transform from Ps to Cf"""
        if k is None: kk = self.co.k
        else: kk = k
        Coef = self.fftPs2Cf.Coef(kk, Ps, extrap='padding', window=window)
        CoefsPow = np.einsum('n,ns->ns', Coef, self.sPow)
        return np.real(np.einsum('ns,n->s', CoefsPow, self.MPs2Cf[l]))

    def Ps2Cf(self, bird):
        for l in range(self.co.Nl):
            for i in range(self.co.N11): bird.C11l[l,i] = self.FT_Ps2Cf(bird.P11l[l,i], l)
            for i in range(self.co.Nct): bird.Cctl[l,i] = self.FT_Ps2Cf(bird.Pctl[l,i], l)
            for i in range(self.co.Nloop): bird.Cloopl[l,i] = self.FT_Ps2Cf(bird.Ploopl[l,i], l)

