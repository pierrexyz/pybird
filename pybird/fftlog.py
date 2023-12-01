from pybird.module import *

def MPC(l, pn):
    """ matrix for spherical bessel transform from power spectrum to correlation function """
    return pi**-1.5 * 2.**(-2. * pn) * gamma(1.5 + l / 2. - pn) / gamma(l / 2. + pn)

def CoefWindow(N, window=1, left=True, right=True):
    """ FFTLog auxiliary function: window sending the FFT coefficients to 0 at the edges. Adapted from fast-pt """
    n = arange(-N // 2, N // 2 + 1)
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

    # W = ones(n.size)
    # if right: W[n[:] > n_right] = theta_right - 1 / (2. * pi) * sin(2 * pi * theta_right)
    # if left: W[n[:] < n_left] = theta_left - 1 / (2. * pi) * sin(2 * pi * theta_left)
    
    W = concatenate((theta_left - 1 / (2. * pi) * sin(2 * pi * theta_left), 
                     ones(n.size - theta_right.size - theta_left.size), 
                     theta_right - 1 / (2. * pi) * sin(2 * pi * theta_right)))
    
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
        self.x = array([self.xmin * exp(i * self.dx) for i in range(self.Nmax)])

    def setPow(self):
        self.Pow = array([self.bias + 1j * 2. * pi / (self.Nmax * self.dx) * (i - self.Nmax / 2.) for i in range(self.Nmax + 1)])

    def Coef(self, xin, f, extrap='extrap', window=1):
        
        interpfunc = InterpolatedUnivariateSpline(xin, f)

        fx = empty(self.Nmax)
        tmp = empty(int(self.Nmax / 2 + 1), dtype=complex)
        Coef = empty(self.Nmax + 1, dtype=complex)

        if extrap == 'extrap':
            nslow, Aslow = 0, 0
            if xin[0] > self.x[0]:
                #print ('low extrapolation')
                if f[0] * f[1] != 0.:
                    nslow = (log(f[1]) - log(f[0])) / (log(xin[1]) - log(xin[0]))
                    Aslow = f[0] / xin[0]**nslow
            nshigh, Ashigh = 0, 0
            if xin[-1] < self.x[-1]:
                #print ('high extrapolation')
                if f[-1] * f[-2] != 0.:
                    nshigh = (log(f[-1]) - log(f[-2])) / (log(xin[-1]) - log(xin[-2]))
                    Ashigh = f[-1] / xin[-1]**nshigh
            
            for i in range(self.Nmax):
                if xin[0] > self.x[i]: fi = Aslow * self.x[i]**nslow * exp(-self.bias * i * self.dx)
                elif xin[-1] < self.x[i]: fi = Ashigh * self.x[i]**nshigh * exp(-self.bias * i * self.dx)
                else: fi = interpfunc(self.x[i]) * exp(-self.bias * i * self.dx)
                if is_jax: fx = fx.at[i].set(fi)
                else: fx[i] = fi

        elif extrap == 'padding':
            for i in range(self.Nmax):
                if xin[0] > self.x[i] or xin[-1] < self.x[i]: fi = 0.
                else: fi = interpfunc(self.x[i]) * exp(-self.bias * i * self.dx)
                if is_jax: fx = fx.at[i].set(fi)
                else: fx[i] = fi

        tmp = rfft(fx)  # numpy
        # tmp = rfft(fx, planner_effort='FFTW_ESTIMATE')() ### pyfftw

        for i in range(self.Nmax + 1):
            if (i < self.Nmax / 2):
                c = conj(tmp[int(self.Nmax / 2 - i)]) * self.xmin**(-self.Pow[i]) / float(self.Nmax)
                if is_jax: Coef = Coef.at[i].set(c)
                else: Coef[i] = c
            else:
                c = tmp[int(i - self.Nmax / 2)] * self.xmin**(-self.Pow[i]) / float(self.Nmax)
                if is_jax: Coef = Coef.at[i].set(c)
                else: Coef[i] = c

        if window:
            Coef = Coef * CoefWindow(self.Nmax, window=window)
        else:
            if is_jax: 
                Coef = Coef.at[0].divide(2.)
                Coef = Coef.at[self.Nmax].divide(2.)
            else: 
                Coef[0] /= 2.
                Coef[self.Nmax] /= 2.

        return Coef
    

    def sumCoefxPow(self, xin, f, x, window=1):
        Coef = self.Coef(xin, f, window=window)
        # fFFT = empty_like(x)
        # for i, xi in enumerate(x):
        #     fFFT[i] = real(sum(Coef * xi**self.Pow))
        # return fFFT
        return array([real(sum(Coef * xi**self.Pow)) for xi in x])
