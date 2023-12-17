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
        self.x = array([self.xmin * exp(i * self.dx) for i in range(self.Nmax)])
        self.xpb = array([exp(-self.bias * i * self.dx) for i in range(self.Nmax)])
        self.Pow = array([self.bias + 1j * 2. * pi * i / (self.Nmax * self.dx) for i in arange(-self.Nmax//2, self.Nmax//2+1)])
        
        if 'window' in kwargs: 
            self.window = kwargs['window']
            self.W = CoefWindow(self.Nmax, window=self.window)
        else: 
            self.window = None
        
    
    def Coef(self, xin, f, extrap='extrap'):
        
        if extrap == 'extrap': 
            iloglog = InterpolatedUnivariateSpline(log(xin), log(f), k=1)
            fx = exp(iloglog(log(self.x)))
        
        elif extrap == 'padding': # this is kind of slow but I don't know how to avoid dynamic tracing in order to JAX-jit
            ifunc = InterpolatedUnivariateSpline(xin, f, k=3)
            def f(x): return where((x < xin[0]) | (xin[-1] < x), 0.0, ifunc(x))
            if is_jax: fx = vmap(lambda x: f(x))(self.x)
            else: fx = f(self.x)
        
        fx = fx * self.xpb
        
        tmp = rfft(fx)
        
        Coef = concatenate((conj(tmp[::-1][:-1]), tmp))
        
        Coef = Coef * exp(-self.Pow*log(self.xmin)) / float(self.Nmax)
        
        if self.window:
            Coef = Coef * self.W
        
        else:
            if is_jax: 
                Coef = Coef.at[0].divide(2.).at[self.Nmax].divide(2.)
            else: 
                Coef[0] /= 2.
                Coef[self.Nmax] /= 2.

        return Coef
    
    def sumCoefxPow(self, xin, f, x):
        Coef = self.Coef(xin, f)
        return array([real(sum(Coef * xi**self.Pow)) for xi in x])