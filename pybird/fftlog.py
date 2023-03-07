import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from numpy.fft import rfft
from scipy.interpolate import interp1d
from scipy.special import gamma

def MPC(l, pn):
    """ matrix for spherical bessel transform from power spectrum to correlation function """
    return pi**-1.5 * 2.**(-2. * pn) * gamma(1.5 + l / 2. - pn) / gamma(l / 2. + pn)

def CoefWindow(N, window=1, left=True, right=True):
    """ FFTLog auxiliary function: window sending the FFT coefficients to 0 at the edges. Adapted from fast-pt """
    n = np.arange(-N // 2, N // 2 + 1)
    if window == 1:
        n_cut = N // 2
    else:
        n_cut = int(window * N // 2.)

    n_right = n[-1] - n_cut
    n_left = n[0] + n_cut

    n_r = n[n[:] > n_right]
    n_l = n[n[:] < n_left]

    theta_right = (n[-1] - n_r) / float(n[-1] - n_right - 1)
    theta_left = (n_l - n[0]) / float(n_left - n[0] - 1)

    W = np.ones(n.size)
    if right: W[n[:] > n_right] = theta_right - 1 / (2. * pi) * sin(2 * pi * theta_right)
    if left: W[n[:] < n_left] = theta_left - 1 / (2. * pi) * sin(2 * pi * theta_left)

    return W

class FFTLog(object):
    """
    A class implementing the FFTLog algorithm.

    Attributes
    ----------
    Nmax : int, optional
        maximum number of points used to discretize the function
    xmin : float, optional
        minimum of the function to transform
    xmax : float, optional
        maximum of the function to transform
    bias : float, optional
        power by which we modify the function as x**bias * f

    Methods
    -------
    setx()
        Calculates the discrete x points for the transform

    setPow()
        Calculates the power in front of the function

    Coef()
        Calculates the single coefficients

    sumCoefxPow(xin, f, x, window=1)
        Sums over the Coef * Pow reconstructing the input function
    """

    def __init__(self, **kwargs):
        self.Nmax = kwargs['Nmax']
        self.xmin = kwargs['xmin']
        self.xmax = kwargs['xmax']
        self.bias = kwargs['bias']
        self.dx = log(self.xmax / self.xmin) / (self.Nmax - 1.)
        self.setx()
        self.setPow()

    def setx(self):
        self.x = np.empty(self.Nmax)
        for i in range(self.Nmax):
            self.x[i] = self.xmin * exp(i * self.dx)

    def setPow(self):
        self.Pow = np.empty(self.Nmax + 1, dtype=complex)
        for i in range(self.Nmax + 1):
            self.Pow[i] = self.bias + 1j * 2. * pi / (self.Nmax * self.dx) * (i - self.Nmax / 2.)

    def Coef(self, xin, f, extrap='extrap', window=1):

        interpfunc = interp1d(xin, f, kind='cubic')

        fx = np.empty(self.Nmax)
        tmp = np.empty(int(self.Nmax / 2 + 1), dtype=complex)
        Coef = np.empty(self.Nmax + 1, dtype=complex)

        if extrap is 'extrap':
            if xin[0] > self.x[0]:
                #print ('low extrapolation')
                nslow = (log(f[1]) - log(f[0])) / (log(xin[1]) - log(xin[0]))
                Aslow = f[0] / xin[0]**nslow
            if xin[-1] < self.x[-1]:
                #print ('high extrapolation')
                nshigh = (log(f[-1]) - log(f[-2])) / (log(xin[-1]) - log(xin[-2]))
                Ashigh = f[-1] / xin[-1]**nshigh

            for i in range(self.Nmax):
                if xin[0] > self.x[i]:
                    fx[i] = Aslow * self.x[i]**nslow * exp(-self.bias * i * self.dx)
                elif xin[-1] < self.x[i]:
                    fx[i] = Ashigh * self.x[i]**nshigh * exp(-self.bias * i * self.dx)
                else:
                    fx[i] = interpfunc(self.x[i]) * exp(-self.bias * i * self.dx)

        elif extrap is'padding':
            for i in range(self.Nmax):
                if xin[0] > self.x[i]:
                    fx[i] = 0.
                elif xin[-1] < self.x[i]:
                    fx[i] = 0.
                else:
                    fx[i] = interpfunc(self.x[i]) * exp(-self.bias * i * self.dx)

        tmp = rfft(fx)  # numpy
        # tmp = rfft(fx, planner_effort='FFTW_ESTIMATE')() ### pyfftw

        for i in range(self.Nmax + 1):
            if (i < self.Nmax / 2):
                Coef[i] = np.conj(tmp[int(self.Nmax / 2 - i)]) * self.xmin**(-self.Pow[i]) / float(self.Nmax)
            else:
                Coef[i] = tmp[int(i - self.Nmax / 2)] * self.xmin**(-self.Pow[i]) / float(self.Nmax)

        if window is not None:
            Coef = Coef * CoefWindow(self.Nmax, window=window)
        else:
            Coef[0] /= 2.
            Coef[self.Nmax] /= 2.

        return Coef

    def sumCoefxPow(self, xin, f, x, window=1):
        Coef = self.Coef(xin, f, window=window)
        fFFT = np.empty_like(x)
        for i, xi in enumerate(x):
            fFFT[i] = np.real(np.sum(Coef * xi**self.Pow))
        return fFFT
