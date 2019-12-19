import os
import numpy as np
from pyfftw.builders import rfft
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt

class FFTLog(object):
    def __init__(self, **kwargs):
        self.Nmax = kwargs['Nmax']
        self.xmin = kwargs['xmin']
        self.xmax = kwargs['xmax']
        self.bias = kwargs['bias']
        self.dx = np.log(self.xmax/self.xmin) / (self.Nmax-1.)
        self.setx()
        self.setPow()
    
    def setx(self):
        self.x = self.xmin * np.exp(np.arange(self.Nmax) * self.dx)
    
    def setPow(self):
        self.Pow = self.bias + 1j * 2. * np.pi / (self.Nmax * self.dx) * (np.arange(self.Nmax+1) - self.Nmax/2.)
    
    def Coef(self, xin, f, window=1, co=common):
        interpfunc = interp1d(xin, f, kind='cubic')
        if xin[0] > self.x[0]:
            print ('low extrapolation')
            nslow = (log(f[1])-log(f[0])) / (log(xin[1])-log(xin[0]))
            Aslow = f[0] / xin[0]**nslow
        if xin[-1] < self.x[-1]:
            print ('high extrapolation')
            nshigh = (log(f[-1])-log(f[-2])) / (log(xin[-1])-log(xin[-2]))
            Ashigh = f[-1] / xin[-1]**nshigh
    
        fx = np.empty(self.Nmax)
        tmp = np.empty(int(self.Nmax/2+1), dtype = complex)
        Coef = np.empty(self.Nmax+1, dtype = complex)
        
        for i in range(self.Nmax): 
            if xin[0] > self.x[i]: fx[i] = Aslow * self.x[i]**nslow * np.exp(-self.bias*i*self.dx)
            elif xin[-1] < self.x[i]: fx[i] = Ashigh * self.x[i]**nshigh * np.exp(-self.bias*i*self.dx)
            else: fx[i] = interpfunc(self.x[i]) * np.exp(-self.bias*i*self.dx)
        
        #tmp = rfft(fx) ### numpy
        tmp = rfft(fx)() ### pyfftw
        
        for i in range(self.Nmax+1):
            if (i < self.Nmax/2): Coef[i] = np.conj(tmp[int(self.Nmax/2-i)]) * self.xmin**(-self.Pow[i]) / float(self.Nmax)
            else: Coef[i] = tmp[int(i-self.Nmax/2)] * self.xmin**(-self.Pow[i]) / float(self.Nmax)
        
        if window is not None: Coef = Coef*CoefWindow(self.Nmax, window=window)
        else:
            Coef[0] /= 2.
            Coef[self.Nmax] /= 2.
        
        return Coef
        #return self.x, 
    
    def sumCoefxPow(self, xin, f, x, window=1):    
        Coef = self.Coef(xin, f, window=window)
        fFFT = np.empty_like(x)
        for i, xi in enumerate(x):
            fFFT[i] = np.real( np.sum(Coef * xi**self.Pow) )
        return fFFT