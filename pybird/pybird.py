import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from numpy.fft import rfft
# from pyfftw.builders import rfft
from scipy.interpolate import interp1d
from scipy.special import gamma, legendre, j1, spherical_jn
from scipy.integrate import quad
from .pyfeather import Qa, Qakp2x12, Qakp2x16, Qawithhex, Qawithhex20

def cH(Om, a):
    """ LCDM growth rate auxiliary function """
    return np.sqrt(Om / a + a**2 * (1 - Om))


def DgN(Om, a):
    """ LCDM growth rate auxiliary function """
    return 5. / 2 * Om * cH(Om, a) / a * quad(lambda x: cH(Om, x)**-3, 0, a)[0]


def fN(Om, z):
    """ LCDM growth rate """
    a = 1. / (1. + z)
    return (Om * (5 * a - 3 * DgN(Om, a))) / (2. * (a**3 * (1 - Om) + Om) * DgN(Om, a))


def Hubble(Om, z):
    """ LCDM AP parameter auxiliary function """
    return ((Om) * (1 + z)**3. + (1 - Om))**0.5


def DA(Om, z):
    """ LCDM AP parameter auxiliary function """
    r = quad(lambda x: 1. / Hubble(Om, x), 0, z)[0]
    return r / (1 + z)


def W2D(x):
    """ Fiber collision effective window method auxiliary function  """
    return (2. * j1(x)) / x


def Hllp(l, lp, x):
    """ Fiber collision effective window method auxiliary function  """
    if l == 2 and lp == 0:
        return x ** 2 - 1.
    if l == 4 and lp == 0:
        return 1.75 * x**4 - 2.5 * x**2 + 0.75
    if l == 4 and lp == 2:
        return x**4 - x**2
    if l == 6 and lp == 0:
        return 4.125 * x**6 - 7.875 * x**4 + 4.375 * x**2 - 0.625
    if l == 6 and lp == 2:
        return 2.75 * x**6 - 4.5 * x**4 + 7. / 4. * x**2
    if l == 6 and lp == 4:
        return x**6 - x**4
    else:
        return x * 0.


def fllp_IR(l, lp, k, q, Dfc):
    """ Fiber collision effective window method auxiliary function  """
    # IR q < k
    # q is an array, k is a scalar
    if l == lp:
        return (q / k) * W2D(q * Dfc) * (q / k)**l
    else:
        return (q / k) * W2D(q * Dfc) * (2. * l + 1.) / 2. * Hllp(max(l, lp), min(l, lp), q / k)


def fllp_UV(l, lp, k, q, Dfc):
    """ Fiber collision effective window method auxiliary function  """
    # UV q > k
    # q is an array, k is a scalar
    if l == lp:
        return W2D(q * Dfc) * (k / q)**l
    else:
        return W2D(q * Dfc) * (2. * l + 1.) / 2. * Hllp(max(l, lp), min(l, lp), k / q)


# powers of mu to Legendre polynomials
mu = {
    0: {0: 1., 2: 0., 4: 0.},
    2: {0: 1. / 3., 2: 2. / 3., 4: 0.},
    4: {0: 1. / 5., 2: 4. / 7., 4: 8. / 35.},
    6: {0: 1. / 7., 2: 10. / 21., 4: 24. / 77.},
    8: {0: 1. / 9., 2: 40. / 99., 4: 48. / 148.}
}

# specific part of the 13-loop matrices
M13b = {
    0: lambda n1: 1.125,
    1: lambda n1: -(1 / (1 + n1)),
    2: lambda n1: 2.25,
    3: lambda n1: (3 * (-1 + 3 * n1)) / (4. * (1 + n1)),
    4: lambda n1: -(1 / (1 + n1)),
    5: lambda n1: -9 / (4 + 4 * n1),
    6: lambda n1: (9 + 18 * n1) / (4 + 4 * n1),
    7: lambda n1: (3 * (-5 + 3 * n1)) / (8. * (1 + n1)),
    8: lambda n1: -9 / (4 + 4 * n1),
    9: lambda n1: (9 * n1) / (4 + 4 * n1),
}


def M13a(n1):
    """ Common part of the 13-loop matrices """
    return np.tan(n1 * pi) / (14. * (-3 + n1) * (-2 + n1) * (-1 + n1) * n1 * pi)


# specific part of the 22-loop matrices
M22b = {
    0: lambda n1, n2: (6 + n1**4 * (4 - 24 * n2) - 7 * n2 + 8 * n1**5 * n2 - 13 * n2**2 + 4 * n2**3 + 4 * n2**4 + n1**2 * (-13 + 38 * n2 + 12 * n2**2 - 8 * n2**3) + 2 * n1**3 * (2 - 5 * n2 - 4 * n2**2 + 8 * n2**3) + n1 * (-7 - 6 * n2 + 38 * n2**2 - 10 * n2**3 - 24 * n2**4 + 8 * n2**5)) / (4. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (-18 + n1**2 * (1 - 11 * n2) - 12 * n2 + n2**2 + 10 * n2**3 + 2 * n1**3 * (5 + 7 * n2) + n1 * (-12 - 38 * n2 - 11 * n2**2 + 14 * n2**3)) / (7. * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: (-3 * n1 + 2 * n1**2 + n2 * (-3 + 2 * n2)) / (n1 * n2),
    3: lambda n1, n2: (-4 * (-24 + n2 + 10 * n2**2) + 2 * n1 * (-2 + 51 * n2 + 21 * n2**2) + n1**2 * (-40 + 42 * n2 + 98 * n2**2)) / (49. * n1 * (1 + n1) * n2 * (1 + n2)),
    4: lambda n1, n2: (4 * (3 - 2 * n2 + n1 * (-2 + 7 * n2))) / (7. * n1 * n2),
    5: lambda n1, n2: 2,
    6: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-2 + 3 * n2 + 4 * n1**4 * n2 + 3 * n2**2 - 2 * n2**3 + n1**3 * (-2 - 2 * n2 + 4 * n2**2) + n1**2 * (3 - 10 * n2 - 4 * n2**2 + 4 * n2**3) + n1 * (3 + 2 * n2 - 10 * n2**2 - 2 * n2**3 + 4 * n2**4))) / (2. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    7: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (2 + 4 * n2 + 5 * n2**2 + n1**2 * (5 + 7 * n2) + n1 * (4 + 10 * n2 + 7 * n2**2))) / (7. * n1 * (1 + n1) * n2 * (1 + n2)),
    8: lambda n1, n2: ((n1 + n2) * (-3 + 2 * n1 + 2 * n2)) / (n1 * n2),
    9: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (10 - 23 * n2 + 28 * n1**4 * n2 + 5 * n2**2 + 2 * n2**3 + n1**3 * (2 - 46 * n2 + 28 * n2**2) + n1**2 * (5 - 38 * n2 - 28 * n2**2 + 28 * n2**3) + n1 * (-23 + 94 * n2 - 38 * n2**2 - 46 * n2**3 + 28 * n2**4))) / (14. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    10: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-58 + 4 * n2 + 35 * n2**2 + 7 * n1**2 * (5 + 7 * n2) + n1 * (4 + 14 * n2 + 49 * n2**2))) / (49. * n1 * (1 + n1) * n2 * (1 + n2)),
    11: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-8 + 7 * n1 + 7 * n2)) / (7. * n1 * n2),
    12: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (2 + 2 * n1**3 - n2 - n2**2 + 2 * n2**3 - n1**2 * (1 + 2 * n2) - n1 * (1 + 2 * n2 + 2 * n2**2))) / (8. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    13: lambda n1, n2: ((1 + n1 + n2) * (2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (8. * n1 * (1 + n1) * n2 * (1 + n2)),
    14: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-6 - n1 + 2 * n1**2 - n2 + 2 * n2**2)) / (8. * n1 * (1 + n1) * n2 * (1 + n2)),
    15: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (38 + 41 * n2 + 112 * n1**3 * n2 - 66 * n2**2 + 2 * n1**2 * (-33 - 18 * n2 + 56 * n2**2) + n1 * (41 - 232 * n2 - 36 * n2**2 + 112 * n2**3))) / (56. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    16: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (9 + 3 * n1 + 3 * n2 + 7 * n1 * n2)) / (14. * n1 * (1 + n1) * n2 * (1 + n2)),
    17: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (5 + 5 * n1 + 5 * n2 + 7 * n1 * n2)) / (14. * n1 * (1 + n1) * n2 * (1 + n2)),
    18: lambda n1, n2: (3 - 2 * n1 - 2 * n2) / (2. * n1 * n2),
    19: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (2. * n1 * n2),
    20: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (50 - 9 * n2 + 98 * n1**3 * n2 - 35 * n2**2 + 7 * n1**2 * (-5 - 18 * n2 + 28 * n2**2) + n1 * (-9 - 66 * n2 - 126 * n2**2 + 98 * n2**3))) / (196. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    21: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (2 + n1 + 4 * n1**3 + n2 - 8 * n1 * n2 - 8 * n1**2 * n2 - 8 * n1 * n2**2 + 4 * n2**3)) / (8. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    22: lambda n1, n2: ((2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2)) / (8. * n1 * (1 + n1) * n2 * (1 + n2)),
    23: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (-2 + 7 * n1 + 7 * n2)) / (56. * n1 * (1 + n1) * n2 * (1 + n2)),
    24: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (26 + 9 * n2 + 56 * n1**3 * n2 - 38 * n2**2 + 2 * n1**2 * (-19 - 18 * n2 + 56 * n2**2) + n1 * (9 - 84 * n2 - 36 * n2**2 + 56 * n2**3))) / (56. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    25: lambda n1, n2: (3 * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (32. * n1 * (1 + n1) * n2 * (1 + n2)),
    26: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (1 + 2 * n1**2 - 8 * n1 * n2 + 2 * n2**2)) / (16. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    27: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (3 + 2 * n1 + 2 * n2)) / (32. * n1 * (1 + n1) * n2 * (1 + n2)),
}


def M22a(n1, n2):
    """ Common part of the 22-loop matrices """
    return (gamma(1.5 - n1) * gamma(1.5 - n2) * gamma(-1.5 + n1 + n2)) / (8. * pi**1.5 * gamma(n1) * gamma(3 - n1 - n2) * gamma(n2))


def MPC(l, pn):
    """ matrix for spherical bessel transform from power spectrum to correlation function """
    return pi**-1.5 * 2.**(-2. * pn) * gamma(1.5 + l / 2. - pn) / gamma(l / 2. + pn)


# precomputed k/q-arrays, in [h/Mpc] and [Mpc/h]
kbird = np.array([0.001, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02])
sbird = np.array([1.000e+00, 1.124e+00, 1.264e+00, 1.421e+00, 1.597e+00, 1.796e+00, 2.019e+00, 2.270e+00, 2.551e+00, 2.868e+00, 3.225e+00, 3.625e+00, 4.075e+00, 4.582e+00, 5.151e+00, 5.790e+00, 6.510e+00, 7.318e+00, 8.227e+00, 9.249e+00, 1.040e+01, 1.169e+01, 1.314e+01, 1.477e+01, 1.661e+01, 1.867e+01, 2.099e+01, 2.360e+01, 2.653e+01, 2.982e+01, 3.353e+01, 3.769e+01, 4.238e+01, 4.764e+01, 5.356e+01, 6.000e+01, 6.021e+01, 6.526e+01, 6.769e+01,
                  7.053e+01, 7.579e+01, 7.609e+01, 8.105e+01, 8.555e+01, 8.632e+01, 9.158e+01, 9.617e+01, 9.684e+01, 1.021e+02, 1.074e+02, 1.081e+02, 1.126e+02, 1.179e+02, 1.215e+02, 1.232e+02, 1.284e+02, 1.337e+02, 1.366e+02, 1.389e+02, 1.442e+02, 1.495e+02, 1.536e+02, 1.547e+02, 1.600e+02, 1.727e+02, 1.941e+02, 2.183e+02, 2.454e+02, 2.759e+02, 3.101e+02, 3.486e+02, 3.919e+02, 4.406e+02, 4.954e+02, 5.569e+02, 6.261e+02, 7.038e+02, 7.912e+02, 8.895e+02, 1.000e+03])

class Common(object):
    """
    A class to share data among different objects
    
    Attributes
    ----------
    Nl : int
        The maximum multipole to calculate (default 2)
    """

    def __init__(self, Nl=2, kmin=0.001, kmax=0.25, smin=1., smax=None, optiresum=True, accboost=1.):
        
        self.optiresum = optiresum

        self.Nl = Nl
        self.N11 = 3
        self.Nct = 6
        self.N22 = 28  # number of 22-loops
        self.N13 = 10  # number of 13-loops
        self.Nloop = 12  # number of bias-independent loops

        self.smin = smin # no support for smin: keep default
        self.smax = smax # no support for smax: will compute up to 1000

        if self.smax is not None:
            kmax = 0.5
            self.optiresum = True
            slog = np.geomspace(1., 1000., 100)
            slin = np.arange(40./accboost, 200., 2.5/accboost)
            slogmask = np.where((slog > slin[-1]) | (slog < slin[0] ))[0]
            self.s = np.unique( np.sort( np.concatenate((slog[slogmask], slin)) ) )
        else:
            if self.optiresum is True: self.s = np.arange(60., 200., 2.5/accboost)
            else: self.s = sbird
        self.Ns = self.s.shape[0]
        
        if kmax is not None:
            self.kmin = kmin # no support for kmin: keep default
            self.kmax = kmax
            self.k = kbird
            if self.kmax > kbird[-1]:
                kextra = np.arange(kbird[-1], 0.3+1e-3, 0.005/accboost)
                self.k = np.concatenate([self.k, kextra[1:]])
            if self.kmax > 0.3:
                kextra = np.arange(0.3, self.kmax+1e-3, 0.01/accboost)
                self.k = np.concatenate([self.k, kextra[1:]])

            self.Nk = self.k.shape[0]

        self.l11 = np.empty(shape=(self.Nl, self.N11))
        self.lct = np.empty(shape=(self.Nl, self.Nct))
        self.l22 = np.empty(shape=(self.Nl, self.N22))
        self.l13 = np.empty(shape=(self.Nl, self.N13))

        for i in range(self.Nl):
            l = 2 * i
            self.l11[i] = np.array([mu[0][l], mu[2][l], mu[4][l]])
            self.lct[i] = np.array([mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l]])
            self.l22[i] = np.array([6 * [mu[0][l]] + 7 * [mu[2][l]] + [mu[4][l], mu[2][l], mu[4][l], mu[2][l],
                                                                       mu[4][l], mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]]])
            self.l13[i] = np.array([2 * [mu[0][l]] + 4 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l]]])

common = Common()

NoPsException = 'Power spectrum not computed: please specify a kmax in the Common class.'

class Bird(object):
    """
    Main class which contains the power spectrum and correlation function, given a cosmology and a set of EFT parameters.
    Bird: Biased tracers in redshift space

    Attributes
    ----------
    co : class, optional
        An object of type Common() used to share data
    which : string, optional
        Options to choose (default: 'full'):
        - 'full': to compute with a given cosmology and a given set of EFT parameters. This is the fastest evaluation.
        - 'all': to compute with a given cosmology only. Bird(object) will store all terms factorized from the EFT parameters.
    f : float
        Growth rate (for redshift space distortion)
    DA : float, optional
        Angular distance (for AP effect)
    H : float, optional
        Hubble parameter (for AP effect)
    z : float, optional
        Redshift (for AP effect)
    kin : array
        k-array on which the input linear power spectrum is evaluated
    Pin : array
        Input linear power spectrum
    Plin : scipy.interpolate.interp1d
        Interpolated function of the linear power spectrum
    P11 : ndarray
        Linear power spectrum evaluated on co.k (the internal k-array on which PyBird evaluates the power spectrum)
    P22 : ndarray
        To store the power spectrum 22-loop terms
    P13 : ndarray
        To store the power spectrum 13-loop terms
    C11 : ndarray
        To store the correlation function multipole linear terms
    C22 : ndarray
        To store the correlation function multipole 22-loop terms
    C13 : ndarray
        To store the correlation function multipole 13-loop terms
    Cct : ndarray
        To store the correlation function multipole counter terms
    Ps : ndarray
        To store the power spectrum multipole full linear part and full loop part (the loop including the counterterms)
    Cf : ndarray
        To store the correlation function multipole full linear part and full loop part (the loop including the counterterms)
    fullPs : ndarray
        To store the full power spectrum multipole (linear + loop)
    b11 : ndarray
        EFT parameters for the linear terms per multipole
    b13 : ndarray
        EFT parameters for the 13-loop terms per multipole
    b22 : ndarray
        EFT parameters for the 22-loop terms per multipole
    bct : ndarray
        EFT parameters for the counter terms per multipole
    """

    def __init__(self, kin, Plin, f, DA=None, H=None, z=None, which='full', co=common):
        self.co = co

        self.which = which

        self.f = f  # fN(Omega_m, z)
        self.DA = DA
        self.H = H
        self.z = z

        self.kin = kin
        self.Pin = Plin
        self.Plin = interp1d(kin, Plin, kind='cubic')

        self.P11 = self.Plin(self.co.k)
        self.P22 = np.empty(shape=(self.co.N22, self.co.Nk))
        self.P13 = np.empty(shape=(self.co.N13, self.co.Nk))

        self.Ps = np.empty(shape=(2, self.co.Nl, self.co.Nk))

        if 'all' in self.which:
            self.Ploopl = np.empty(shape=(self.co.Nl, self.co.Nloop, self.co.Nk))
            self.P11l = np.empty(shape=(self.co.Nl, self.co.N11, self.co.Nk))
            self.Pctl = np.empty(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
            self.P22l = np.empty(shape=(self.co.Nl, self.co.N22, self.co.Nk))
            self.P13l = np.empty(shape=(self.co.Nl, self.co.N13, self.co.Nk))

        self.C11 = np.empty(shape=(self.co.Nl, self.co.Ns))
        self.C22 = np.empty(shape=(self.co.Nl, self.co.N22, self.co.Ns))
        self.C13 = np.empty(shape=(self.co.Nl, self.co.N13, self.co.Ns))
        self.Cct = np.empty(shape=(self.co.Nl, self.co.Ns))
        self.Cf = np.empty(shape=(2, self.co.Nl, self.co.Ns))

        if 'all' in self.which:
            self.Cloopl = np.empty(shape=(self.co.Nl, self.co.Nloop, self.co.Ns))
            self.C11l = np.empty(shape=(self.co.Nl, self.co.N11, self.co.Ns))
            self.Cctl = np.empty(shape=(self.co.Nl, self.co.Nct, self.co.Ns))

        ### DEPRECIATED
        # elif self.which is 'marg':
        #     self.full = False
        #     self.Pctl = np.empty(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
        #     self.P13l = np.empty(shape=(self.co.Nl, self.co.N13, self.co.Nk))
        #     self.Pb3 = np.empty(shape=(self.co.Nl, self.co.Nk))
        #     self.Cb3 = np.empty(shape=(self.co.Nl, self.co.Ns))

        self.b11 = np.empty(shape=(self.co.Nl))
        self.b13 = np.empty(shape=(self.co.Nl, self.co.N13))
        self.b22 = np.empty(shape=(self.co.Nl, self.co.N22))
        self.bct = np.empty(shape=(self.co.Nl))

        if 'angular' in self.which:
            self.chi = self.z/self.H
            self.ellmax = np.int(self.co.kmax*self.chi)

            if 'full' in self.which:
                self.Cell = np.empty(shape=(self.ellmax))
                self.w = None

            elif 'all' in self.which:
                self.Cell11 = np.empty(shape=(self.co.N11, self.ellmax))
                self.Cellct = np.empty(shape=(self.co.Nct, self.ellmax))
                self.Cellloop = np.empty(shape=(self.co.Nloop, self.ellmax))
                self.w11 = None
                self.wct = None
                self.wloop = None

    def setBias(self, bs):
        """ For option: which='full'. Given an array of EFT parameters, set them among linear, loops and counter terms, and among multipoles

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f
        for i in range(self.co.Nl):
            l = 2 * i
            self.b11[i] = b1**2 * mu[0][l] + 2. * b1 * f * mu[2][l] + f**2 * mu[4][l]
            self.b22[i] = np.array([b1**2 * mu[0][l], b1 * b2 * mu[0][l], b1 * b4 * mu[0][l], b2**2 * mu[0][l], b2 * b4 * mu[0][l], b4**2 * mu[0][l], b1**2 * f * mu[2][l], b1 * b2 * f * mu[2][l], b1 * b4 * f * mu[2][l], b1 * f * mu[2][l], b2 * f * mu[2][l], b4 * f * mu[2][l], b1**2 * f**2 * mu[2][l], b1**2 *
                                    f**2 * mu[4][l], b1 * f**2 * mu[2][l], b1 * f**2 * mu[4][l], b2 * f**2 * mu[2][l], b2 * f**2 * mu[4][l], b4 * f**2 * mu[2][l], b4 * f**2 * mu[4][l], f**2 * mu[4][l], b1 * f**3 * mu[4][l], b1 * f**3 * mu[6][l], f**3 * mu[4][l], f**3 * mu[6][l], f**4 * mu[4][l], f**4 * mu[6][l], f**4 * mu[8][l]])
            self.b13[i] = np.array([b1**2 * mu[0][l], b1 * b3 * mu[0][l], b1**2 * f * mu[2][l], b1 * f * mu[2][l], b3 *
                                    f * mu[2][l], b1 * f**2 * mu[2][l], b1 * f**2 * mu[4][l], f**2 * mu[4][l], f**3 * mu[4][l], f**3 * mu[6][l]])
            self.bct[i] = 2. * b1 * (b5 * mu[0][l] + b6 * mu[2][l] + b7 * mu[4][l]) + 2. * \
                f * (b5 * mu[2][l] + b6 * mu[4][l] + b7 * mu[6][l])

    def setPs(self, bs):
        """ For option: which='full'. Given an array of EFT parameters, multiplies them accordingly to the power spectrum multipole terms and adds the resulting terms together per loop order

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.Ps[0] = np.einsum('l,x->lx', self.b11, self.P11)
        self.Ps[1] = np.einsum('lb,bx->lx', self.b22, self.P22)
        for l in range(self.co.Nl):
            self.Ps[1, l] -= self.Ps[1, l, 0]
        self.Ps[1] += np.einsum('lb,bx->lx', self.b13, self.P13) + np.einsum('l,x,x->lx', self.bct, self.co.k**2, self.P11)

    def setCf(self, bs):
        """ For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the correlation function multipole terms

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.Cf[0] = np.einsum('l,lx->lx', self.b11, self.C11)
        self.Cf[1] = np.einsum('lb,lbx->lx', self.b22, self.C22) + np.einsum('lb,lbx->lx', self.b13, self.C13) + np.einsum('l,lx->lx', self.bct, self.Cct)

    def setPsCf(self, bs, setfull=True):
        """ For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the power spectrum and correlation function multipole terms

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)

        self.Ps[0] = np.einsum('l,x->lx', self.b11, self.P11)
        self.Ps[1] = np.einsum('lb,bx->lx', self.b22, self.P22)
        for l in range(self.co.Nl): self.Ps[1, l] -= self.Ps[1, l, 0]
        self.Ps[1] += np.einsum('lb,bx->lx', self.b13, self.P13) + np.einsum('l,x,x->lx', self.bct, self.co.k**2, self.P11)
        if setfull: self.setfullPs()

        self.Cf[0] = np.einsum('l,lx->lx', self.b11, self.C11)
        self.Cf[1] = np.einsum('lb,lbx->lx', self.b22, self.C22) + np.einsum('lb,lbx->lx', self.b13, self.C13) + np.einsum('l,lx->lx', self.bct, self.Cct)
        if setfull: self.setfullCf()

    def setfullPs(self):
        """ For option: which='full'. Adds together the linear and the loop parts to get the full power spectrum multipoles """
        self.fullPs = np.sum(self.Ps, axis=0)

    def setfullCf(self):
        """ For option: which='full'. Adds together the linear and the loop parts to get the full correlation function multipoles """
        self.fullCf = np.sum(self.Cf, axis=0)

    def setPsCfl(self):
        """ For option: which='full'. Creates multipoles for each term weighted accordingly """
        self.P11l = np.einsum('x,ln->lnx', self.P11, self.co.l11)
        self.Pctl = np.einsum('x,x,ln->lnx', self.co.k**2, self.P11, self.co.lct)
        self.P22l = np.einsum('nx,ln->lnx', self.P22, self.co.l22)
        self.P13l = np.einsum('nx,ln->lnx', self.P13, self.co.l13)

        self.C11l = np.einsum('lx,ln->lnx', self.C11, self.co.l11)
        self.Cctl = np.einsum('lx,ln->lnx', self.Cct, self.co.lct)
        self.C22 = np.einsum('lnx,ln->lnx', self.C22, self.co.l22)
        self.C13 = np.einsum('lnx,ln->lnx', self.C13, self.co.l13)

        self.reducePsCfl()

    def reducePsCfl(self):
        """ For option: which='all'. Regroups terms that share the same EFT parameter(s) """
        f1 = self.f

        if self.co.Nk is not 0:
            self.Ploopl[:, 0] = f1**2 * self.P22l[:, 20] + f1**3 * self.P22l[:, 23] + f1**3 * self.P22l[:, 24] + f1**4 * self.P22l[:, 25] + \
                f1**4 * self.P22l[:, 26] + f1**4 * self.P22l[:, 27] + f1**2 * \
                self.P13l[:, 7] + f1**3 * self.P13l[:, 8] + f1**3 * self.P13l[:, 9]  # *1
            self.Ploopl[:, 1] = f1 * self.P22l[:, 9] + f1**2 * self.P22l[:, 14] + f1**2 * self.P22l[:, 15] + f1**3 * self.P22l[:, 21] + f1**3 * self.P22l[:, 22] + f1 * self.P13l[:, 3] + f1**2 * self.P13l[:, 5] + f1**2 * self.P13l[:, 6]  # *b1
            self.Ploopl[:, 2] = f1 * self.P22l[:, 10] + f1**2 * self.P22l[:, 16] + f1**2 * self.P22l[:, 17]  # *b2
            self.Ploopl[:, 3] = f1 * self.P13l[:, 4]  # *b3
            self.Ploopl[:, 4] = f1 * self.P22l[:, 11] + f1**2 * self.P22l[:, 18] + f1**2 * self.P22l[:, 19]  # *b4
            self.Ploopl[:, 5] = self.P22l[:, 0] + f1 * self.P22l[:, 6] + f1**2 * self.P22l[:, 12] + \
                f1**2 * self.P22l[:, 13] + self.P13l[:, 0] + f1 * self.P13l[:, 2]  # *b1*b1
            self.Ploopl[:, 6] = self.P22l[:, 1] + f1 * self.P22l[:, 7]  # *b1*b2
            self.Ploopl[:, 7] = self.P13l[:, 1]  # *b1*b3
            self.Ploopl[:, 8] = self.P22l[:, 2] + f1 * self.P22l[:, 8]  # *b1*b4
            self.Ploopl[:, 9] = self.P22l[:, 3]  # *b2*b2
            self.Ploopl[:, 10] = self.P22l[:, 4]  # *b2*b4
            self.Ploopl[:, 11] = self.P22l[:, 5]  # *b4*b4

        self.Cloopl[:, 0] = f1**2 * self.C22[:, 20] + f1**3 * self.C22[:, 23] + f1**3 * self.C22[:, 24] + f1**4 * self.C22[:, 25] + \
            f1**4 * self.C22[:, 26] + f1**4 * self.C22[:, 27] + f1**2 * \
            self.C13[:, 7] + f1**3 * self.C13[:, 8] + f1**3 * self.C13[:, 9]  # *1
        self.Cloopl[:, 1] = f1 * self.C22[:, 9] + f1**2 * self.C22[:, 14] + f1**2 * self.C22[:, 15] + f1**3 * self.C22[:, 21] + f1**3 * self.C22[:, 22] + f1 * self.C13[:, 3] + f1**2 * self.C13[:, 5] + f1**2 * self.C13[:, 6]  # *b1
        self.Cloopl[:, 2] = f1 * self.C22[:, 10] + f1**2 * self.C22[:, 16] + f1**2 * self.C22[:, 17]  # *b2
        self.Cloopl[:, 3] = f1 * self.C13[:, 4]  # *b3
        self.Cloopl[:, 4] = f1 * self.C22[:, 11] + f1**2 * self.C22[:, 18] + f1**2 * self.C22[:, 19]  # *b4
        self.Cloopl[:, 5] = self.C22[:, 0] + f1 * self.C22[:, 6] + f1**2 * self.C22[:, 12] + \
            f1**2 * self.C22[:, 13] + self.C13[:, 0] + f1 * self.C13[:, 2]  # *b1*b1
        self.Cloopl[:, 6] = self.C22[:, 1] + f1 * self.C22[:, 7]  # *b1*b2
        self.Cloopl[:, 7] = self.C13[:, 1]  # *b1*b3
        self.Cloopl[:, 8] = self.C22[:, 2] + f1 * self.C22[:, 8]  # *b1*b4
        self.Cloopl[:, 9] = self.C22[:, 3]  # *b2*b2
        self.Cloopl[:, 10] = self.C22[:, 4]  # *b2*b4
        self.Cloopl[:, 11] = self.C22[:, 5]  # *b4*b4

        self.subtractShotNoise()

    def setreducePslb(self, bs):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the power spectrum multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f

        b11 = np.array([b1**2, 2. * b1 * f, f**2])
        bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
        bloop = np.array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
        # self.Ps[0] = np.einsum('b,lbx->lx', b11, self.P11l)
        # self.Ps[1] = np.einsum('b,lbx->lx', bloop, self.Ploopl)
        # for l in range(self.co.Nl): self.Ps[1,l] -= self.Ps[1,l,0]
        # self.Ps[1] += np.einsum('b,lbx->lx', bct, self.Pctl)

        Ps0 = np.einsum('b,lbx->lx', b11, self.P11l)
        Ps1 = np.einsum('b,lbx->lx', bloop, self.Ploopl) + np.einsum('b,lbx->lx', bct, self.Pctl)
        self.fullPs = Ps0 + Ps1

    def setreduceCflb(self, bs):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the correlation multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f

        b11 = np.array([b1**2, 2. * b1 * f, f**2])
        bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
        bloop = np.array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])

        Cf0 = np.einsum('b,lbx->lx', b11, self.C11l)
        Cf1 = np.einsum('b,lbx->lx', bloop, self.Cloopl) + np.einsum('b,lbx->lx', bct, self.Cctl)
        self.fullCf = Cf0 + Cf1

    def subtractShotNoise(self):
        """ For option: which='all'. Subtract the constant stochastic term from the (22-)loop """
        if self.co.Nk is not 0:
            for l in range(self.co.Nl):
                for n in range(self.co.Nloop):
                    shotnoise = self.Ploopl[l, n, 0]
                    self.Ploopl[l, n] -= shotnoise

    def formatTaylor(self):
        """ An auxiliary to pipe PyBird with TBird: puts Bird(object) power spectrum multipole terms into the right shape for TBird """
        allk = np.concatenate([self.co.k, self.co.k]).reshape(-1, 1)
        Plin = np.flip(np.einsum('n,lnk->lnk', np.array([1., 2. * self.f, self.f**2]), self.P11l), axis=1)
        Plin = np.concatenate(np.einsum('lnk->lkn', Plin), axis=0)
        Plin = np.hstack((allk, Plin))
        Ploop1 = np.concatenate(np.einsum('lnk->lkn', self.Ploopl), axis=0)
        Ploop2 = np.einsum('n,lnk->lnk', np.array([2., 2., 2., 2. * self.f, 2. * self.f, 2. * self.f]), self.Pctl)
        Ploop2 = np.concatenate(np.einsum('lnk->lkn', Ploop2), axis=0)
        Ploop = np.hstack((allk, Ploop1, Ploop2))
        return Plin, Ploop

    ### DEPRECIATED
    # def setmargPsCfl(self, bs):
    #     """ For option: which='marg'. Given an array of EFT parameters, multiply them accordingly to the power spectrum multipole terms and adds the resulting terms together per loop-order and differentiating parts with an EFT parameter appearing only linearly in the power spectrum from the others: {b_3}, {c_{i}} in one side, {b_1, b_2, b_4} on the other.

    #     Parameters
    #     ----------
    #     bs : array
    #         An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
    #     """
    #     b1, b2, _, b4, _, _, _ = bs
    #     f = self.f

    #     for i in range(self.co.Nl):
    #         l = 2 * i
    #         self.b11[i] = b1**2 * mu[0][l] + 2. * b1 * f * mu[2][l] + f**2 * mu[4][l]
    #         self.b22[i] = np.array([b1**2 * mu[0][l], b1 * b2 * mu[0][l], b1 * b4 * mu[0][l], b2**2 * mu[0][l], b2 * b4 * mu[0][l], b4**2 * mu[0][l], b1**2 * f * mu[2][l], b1 * b2 * f * mu[2][l], b1 * b4 * f * mu[2][l], b1 * f * mu[2][l], b2 * f * mu[2][l], b4 * f * mu[2][l], b1**2 * f**2 * mu[2][l], b1**2 *
    #                                 f**2 * mu[4][l], b1 * f**2 * mu[2][l], b1 * f**2 * mu[4][l], b2 * f**2 * mu[2][l], b2 * f**2 * mu[4][l], b4 * f**2 * mu[2][l], b4 * f**2 * mu[4][l], f**2 * mu[4][l], b1 * f**3 * mu[4][l], b1 * f**3 * mu[6][l], f**3 * mu[4][l], f**3 * mu[6][l], f**4 * mu[4][l], f**4 * mu[6][l], f**4 * mu[8][l]])

    #     self.P13l = np.einsum('nx,ln->lnx', self.P13, self.co.l13)
    #     self.C13 = np.einsum('lnx,ln->lnx', self.C13, self.co.l13)

    #     b13nob3 = np.array([1., b1, b1**2])
    #     P13nob3 = np.array([f**2 * self.P13l[:, 7] + f**3 * self.P13l[:, 8] + f**3 * self.P13l[:, 9],   # *1
    #                         + f * self.P13l[:, 3] + f**2 * self.P13l[:, 5] + f**2 * self.P13l[:, 6],     # *b1
    #                         + self.P13l[:, 0] + f * self.P13l[:, 2]                                 # *b1*b1
    #                         ])
    #     C13nob3 = np.array([f**2 * self.C13[:, 7] + f**3 * self.C13[:, 8] + f**3 * self.C13[:, 9],   # *1
    #                         + f * self.C13[:, 3] + f**2 * self.C13[:, 5] + f**2 * self.C13[:, 6],     # *b1
    #                         + self.C13[:, 0] + f * self.C13[:, 2]                                # *b1*b1
    #                         ])

    #     self.Ps[0] = np.einsum('l,x->lx', self.b11, self.P11)
    #     self.Ps[1] = np.einsum('lb,bx->lx', self.b22, self.P22)
    #     for l in range(self.co.Nl):
    #         self.Ps[1, l] -= self.Ps[1, l, 0]
    #     self.Ps[1] += np.einsum('b,blx->lx', b13nob3, P13nob3)

    #     self.Cf[0] = np.einsum('l,lx->lx', self.b11, self.C11)
    #     self.Cf[1] = np.einsum('lb,lbx->lx', self.b22, self.C22) + np.einsum('b,blx->lx', b13nob3, C13nob3)

    #     self.Pb3 = f * self.P13l[:, 4] + b1 * self.P13l[:, 1]     # *b3 + *b1*b3
    #     self.Cb3 = f * self.C13[:, 4] + b1 * self.C13[:, 1]     # *b3 + *b1*b3

    #     self.Pctl = np.einsum('x,x,ln->lnx', self.co.k**2, self.P11, self.co.lct)
    #     self.Pctl = np.einsum('x,x,ln->lnx', self.co.k**2, self.P11, self.co.lct)

    # def setmargPslb(self, bs):
    #     """ For option: which='marg'. Adds together all pieces to get the full power spectrum multipoles """
    #     b1, _, b3, _, b5, b6, b7 = bs
    #     f = self.f
    #     bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
    #     self.fullPs += b3 * self.Pb3 + np.einsum('b,lbx->lx', bct, self.Pctl)

    def setCellb(self, bs):
        """ For option: which='angular_all'. Given an array of EFT parameters, multiply them accordingly to the angular power spectrum regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f

        b11 = np.array([b1**2, 2. * b1 * f, f**2])
        bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
        bloop = np.array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])

        Cell0 = np.einsum('b,bx->x', b11, self.Cell11)
        Cell1 = np.einsum('b,bx->x', bloop, self.Cellloop) + np.einsum('b,bx->x', bct, self.Cellct)
        self.Cell = Cell0 + Cell1

    def setwb(self, bs):
        """ For option: which='angular_all'. Given an array of EFT parameters, multiply them accordingly to the angular correlation function regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        b1, b2, b3, b4, b5, b6, b7 = bs
        f = self.f

        b11 = np.array([b1**2, 2. * b1 * f, f**2])
        bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
        bloop = np.array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])

        w0 = np.einsum('b,bx->x', b11, self.w11)
        w1 = np.einsum('b,bx->x', bloop, self.wloop) + np.einsum('b,bx->x', bct, self.wct)
        self.w = w0 + w1


def CoefWindow(N, window=1, left=True, right=True):
    """ FFTLog auxiliary function: window sending the FFT coefficients to 0 at the edges. Adapted from fast-pt """
    n = np.arange(-N // 2, N // 2 + 1)
    if window is 1:
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
    if right: W[n[:] > n_right] = theta_right - 1 / (2 * pi) * sin(2 * pi * theta_right)
    if left: W[n[:] < n_left] = theta_left - 1 / (2 * pi) * sin(2 * pi * theta_left)

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


class NonLinear(object):
    """
    given a Bird() object, computes the one-loop power spectrum and one-loop correlation function. 
    The correlation function is useful to perform the IR-resummation of the power spectrum.
    The loop and spherical Bessel transform matrices are either loaded either precomputed and stored at the instanciation of the class. 

    Attributes
    ----------
    co : class
        An object of type Common() used to share data
    fftsettings: dict
    fft : class
        An object of type FFTLog() to perform the FFTLog
    M22 : ndarray
        22-loop power spectrum matrices
    M13 : ndarray 
        13-loop power spectrum matrices
    Mcf11 : ndarray
        Spherical Bessel transform matrices of the linear power spectrum to correlation function
    Ml : ndarray
        Spherical Bessel transform matrices of the loop power spectrum to correlation function multipole. Auxiliary matrices used for the loop correlation function matrices.
    Mcf22 : ndarray
        22-loop correlation function multipole matrices
    Mcf13 : ndarray
        13-loop correlation function multipole matrices
    kPow : ndarray
        k's to the powers on which to perform the FFTLog to evaluate the loop power spectrum
    sPow : ndarray
        s's to the powers on which to perform the FFTLog to evaluate the loop correlation function
    optipathP22 : NumPy einsum_path
        Optimization settings for NumPy einsum when performing matrix multiplications to compute the 22-loop power spectrum. For speedup purpose in repetitive evaluations.
    optipathC13 : NumPy einsum_path
        Optimization settings for NumPy einsum when performing matrix multiplications to compute the 13-loop correlation function. For speedup purpose in repetitive evaluations.
    optipathC22 : NumPy einsum_path
        Optimization settings for NumPy einsum when performing matrix multiplications to compute the 22-loop correlation function. For speedup purpose in repetitive evaluations. 
    """

    def __init__(self, load=True, save=True, path='./', NFFT=256, co=common):

        self.co = co

        self.fftsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1000., bias=-1.6)

        self.fft = FFTLog(**self.fftsettings)

        if load is True:
            try:
                L = np.load(os.path.join(path, 'pyegg%s_nl%s.npz') % (NFFT, self.co.Nl) )
                if (self.fft.Pow - L['Pow']).any():
                    print ('Loaded loop matrices do not correspond to asked FFTLog configuration. \n Computing new matrices.')
                    load = False
                else:
                    self.M22, self.M13, self.Mcf11, self.Mcf22, self.Mcf13, self.Mcfct = L['M22'], L['M13'], L['Mcf11'], L['Mcf22'], L['Mcf13'], L['Mcfct']
                    save = False
            except:
                print ('Can\'t load loop matrices at %s. \n Computing new matrices.' % path)
                load = False

        if load is False:
            self.setM22()
            self.setM13()
            self.setMl()
            self.setMcf11()
            self.setMcf22()
            self.setMcf13()
            self.setMcfct()

        if save is True:
            try:
                np.savez(os.path.join(path, 'pyegg%s_nl%s.npz') % (NFFT, self.co.Nl), Pow=self.fft.Pow,
                         M22=self.M22, M13=self.M13, Mcf11=self.Mcf11, Mcf22=self.Mcf22, Mcf13=self.Mcf13, Mcfct=self.Mcfct)
            except:
                print ('Can\'t save loop matrices at %s.' % path)

        self.setkPow()
        self.setsPow()

        # To speed-up matrix multiplication:
        self.optipathP22 = np.einsum_path('nk,mk,bnm->bk', self.kPow, self.kPow, self.M22, optimize='optimal')[0]
        self.optipathC13 = np.einsum_path('ns,ms,blnm->bls', self.sPow, self.sPow, self.Mcf22, optimize='optimal')[0]
        self.optipathC22 = np.einsum_path('ns,ms,blnm->bls', self.sPow, self.sPow, self.Mcf13, optimize='optimal')[0]

    def setM22(self):
        """ Compute the 22-loop power spectrum matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M22 = np.empty(shape=(self.co.N22, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        # common piece of M22
        Ma = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        for u, n1 in enumerate(-0.5 * self.fft.Pow):
            for v, n2 in enumerate(-0.5 * self.fft.Pow):
                Ma[u, v] = M22a(n1, n2)
        for i in range(self.co.N22):
            # singular piece of M22
            Mb = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                for v, n2 in enumerate(-0.5 * self.fft.Pow):
                    Mb[u, v] = M22b[i](n1, n2)
            self.M22[i] = Ma * Mb

    def setM13(self):
        """ Compute the 13-loop power spectrum matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M13 = np.empty(shape=(self.co.N13, self.fft.Pow.shape[0]), dtype='complex')
        Ma = M13a(-0.5 * self.fft.Pow)
        for i in range(self.co.N13):
            self.M13[i] = Ma * M13b[i](-0.5 * self.fft.Pow)

    def setMcf11(self):
        """ Compute the 11-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf11 = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                self.Mcf11[l, u] = 1j**(2*l) * MPC(2 * l, n1)

    def setMl(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Ml = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                for v, n2 in enumerate(-0.5 * self.fft.Pow):
                    self.Ml[l, u, v] = 1j**(2*l) * MPC(2 * l, n1 + n2 - 1.5)

    def setMcf22(self):
        """ Compute the 22-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf22 = np.einsum('lnm,bnm->blnm', self.Ml, self.M22)

    def setMcf13(self):
        """ Compute the 13-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf13 = np.einsum('lnm,bn->blnm', self.Ml, self.M13)

    def setMcfct(self):
        """ Compute the counterterm correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcfct = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5 * self.fft.Pow - 1.):
                self.Mcfct[l, u] = 1j**(2*l) * MPC(2 * l, n1)

    def setkPow(self):
        """ Compute the k's to the powers of the FFTLog to evaluate the loop power spectrum. Called at the instantiation of the class. """
        self.kPow = exp(np.einsum('n,k->nk', self.fft.Pow, log(self.co.k)))

    def setsPow(self):
        """ Compute the s's to the powers of the FFTLog to evaluate the loop correlation function. Called at the instantiation of the class. """
        self.sPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3., log(self.co.s)))

    def CoefkPow(self, Coef):
        """ Multiply the coefficients with the k's to the powers of the FFTLog to evaluate the loop power spectrum. """
        return np.einsum('n,nk->nk', Coef, self.kPow)

    def CoefsPow(self, Coef):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the correlation function. """
        return np.einsum('n,ns->ns', Coef, self.sPow)

    def makeP22(self, CoefkPow, bird):
        """ Perform the 22-loop power spectrum matrix multiplications """
        bird.P22 = self.co.k**3 * np.real(np.einsum('nk,mk,bnm->bk', CoefkPow,
                                                    CoefkPow, self.M22, optimize=self.optipathP22))

    def makeP13(self, CoefkPow, bird):
        """ Perform the 13-loop power spectrum matrix multiplications """
        bird.P13 = self.co.k**3 * bird.P11 * np.real(np.einsum('nk,bn->bk', CoefkPow, self.M13))

    def makeC11(self, CoefsPow, bird):
        """ Perform the 11-loop correlation function matrix multiplications """
        bird.C11 = np.real(np.einsum('ns,ln->ls', CoefsPow, self.Mcf11))

    def makeCct(self, CoefsPow, bird):
        """ Perform the counterterm correlation function matrix multiplications """
        bird.Cct = self.co.s**-2 * np.real(np.einsum('ns,ln->ls', CoefsPow, self.Mcfct))

    def makeC22(self, CoefsPow, bird):
        """ Perform the 22-loop correlation function matrix multiplications """
        bird.C22 = np.real(np.einsum('ns,ms,blnm->lbs', CoefsPow, CoefsPow, self.Mcf22, optimize=self.optipathC22))

    def makeC13(self, CoefsPow, bird):
        """ Perform the 13-loop correlation function matrix multiplications """
        bird.C13 = np.real(np.einsum('ns,ms,blnm->lbs', CoefsPow, CoefsPow, self.Mcf13, optimize=self.optipathC13))

    def Coef(self, bird, window=None):
        """ Perform the FFTLog (i.e. calculate the coefficients of the FFTLog) of the input linear power spectrum in the given a Bird().

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        return self.fft.Coef(bird.kin, bird.Pin, window=window)

    def Ps(self, bird, window=None):
        """ Compute the loop power spectrum given a Bird(). Perform the FFTLog and the matrix multiplications.

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        coef = self.Coef(bird, window=.2)
        coefkPow = self.CoefkPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)

    def Cf(self, bird, window=None):
        """ Compute the loop correlation function given a Bird(). Perform the FFTLog and the matrix multiplications.

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        coef = self.Coef(bird, window=.2)
        coefsPow = self.CoefsPow(coef)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22(coefsPow, bird)
        self.makeC13(coefsPow, bird)

    def PsCf(self, bird, window=None):
        """ Compute the loop power spectrum and correlation function given a Bird(). Perform the FFTLog and the matrix multiplications.

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        coef = self.Coef(bird, window=.2)

        coefkPow = self.CoefkPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)
        
        coefsPow = self.CoefsPow(coef)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22(coefsPow, bird)
        self.makeC13(coefsPow, bird)


class Resum(object):
    """
    given a Bird() object, performs the IR-resummation of the power spectrum. 
    There are two options:
    1.  fullresum: the FFTLog's are performed on the full integrands from s = .1 to s = 10000. in (Mpc/h) (default)
    2. 'optiresum: the FFTLog's are performed only on the BAO peak that is extracted by removing the smooth part of the correlation function. What is left is then padded with zeros and the FFTLog's run from s = .1 to s = 1000. in (Mpc/h).


    Attributes
    ----------
    co : class
        An object of type Common() used to share data
    LambdaIR : float
        Integral cutoff for IR-filters X and Y (fullresum: LambdaIR=.2 (default), optiresum: LambdaIR= 1 ; either value can do for either resummation)
    NIR : float
        Number of IR-correction terms in the sums over n and alpha, where n is the order of the Taylor expansion in powers of k^2 of the exponential of the bulk displacements, and for each n, alpha = { 0, 2 } are the orders of spherical Bessel functions. The ordering of the IR-corrections is given by (n,alpha), where alpha is running faster, e.g. (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2), ...
    k2p: ndarray
        powers of k^2
    alllpr : ndarray
        alpha = { 0, 2 } orders of spherical Bessel functions, for each n
    Q : ndarray
        IR-resummation bulk coefficients Q^{ll'}_{||N-j}(n, \alpha, f) of the IR-resummation matrices. f is the growth rate. Computed in method Ps().
    IRcorr : ndarray
        Q-independent pieces in the IR-correction sums over n and alpha of the power spectrum, for bird.which = 'full'. Computed in method Ps().
    IR11 : ndarray
        Q-independent in the IR-correction sums over n and alpha of the power spectrum linear part, for bird.which = 'all'. Computed in method Ps().
    IRct : ndarray
        Q-independent pieces in the IR-correction sums over n and alpha of the power spectrum counterterm, for bird.which = 'all'. Computed in method Ps().
    IRloop : ndarray
        Q-independent loop pieces in the IR-correction sums over n and alpha of the power spectrum loop part, for bird.which = 'all'. Computed in method Ps().
    IRresum : ndarray
        IR-corrections to the power spectrum, for bird.which = 'full'. Computed in method Ps().
    IR11resum : ndarray
        IR-corrections to the power spectrum linear parts, for bird.which = 'all'. Computed in method Ps().
    IRctresum : ndarray
        IR-corrections to the power spectrum counterterms, for bird.which = 'all'. Computed in method Ps().
    IRloopresum : ndarray
        IR-corrections to the power spetrum loop parts, for bird.which = 'all'. Computed in method Ps().
    fftsettings : dict
        Number of points and boundaries of the FFTLog's for the computing the IR-corrections
    fft : class
        An object of type FFTLog() to evaluate the IR-corrections
    M : ndarray
        spherical Bessel transform matrices to evaluate the IR-corrections
    kPow : ndarray
        k's to the powers on which to perform the FFTLog to evaluate the IR-corrections.
    Xfftsettings : dict
        Number of points and boundaries of the FFTLog's for evaluating the IR-filters X and Y
    Xfft : class
        An object of type FFTLog() to evaluate the IR-filters X and Y
    XM : ndarray
        spherical Bessel transform matrices to evaluate the IR-filters X and Y
    XsPow : ndarray
        s's to the powers on which to perform the FFTLog to evaluate the IR-filters X and Y    
    """

    def __init__(self, LambdaIR=1., NFFT=192, co=common, high=False):

        self.high = high

        self.co = co      

        if self.co.optiresum is True:
            self.LambdaIR = LambdaIR
            self.sLow = 60.
            self.sHigh = 190.
            self.idlow = np.where(self.co.s > self.sLow)[0][0]
            self.idhigh = np.where(self.co.s > self.sHigh)[0][0]
            self.sbao = self.co.s[self.idlow:self.idhigh]
            self.snobao = np.concatenate([self.co.s[:self.idlow], self.co.s[self.idhigh:]])
            self.sr = self.sbao
        else:
            self.LambdaIR = .2
            self.sr = self.co.s

        self.klow = 0.02
        self.kr = self.co.k[self.klow <= self.co.k]
        self.Nkr = self.kr.shape[0]
        self.idklow = np.where(self.klow <= self.co.k)[0][0]

        self.IRorder = 16
        k2pi = np.array([self.kr**(2*(p+1)) for p in range(self.IRorder)])
        self.k2p = np.concatenate((k2pi, k2pi))
        self.alllpr = np.array([[0, 1, 2] for p in range(self.IRorder*2)])

        # if self.co.Nl is 2:
        #     # By default, we perform a 8-loop resummation. 
        #     k2pi = np.array([self.kr**2, self.kr**4, self.kr**6, self.kr**8, self.kr**10, self.kr**12, self.kr**14, self.kr**16])
        #     self.k2p = np.concatenate((k2pi, k2pi))
        #     self.alllpr = np.array([
        #         [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
        #         [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
        #     ])

        #     # For k's higher than ~ 0.2, we perform a 12-loop resummation.
        #     self.k12loop = 0.21
        #     if self.co.kmax > self.k12loop:
        #         # self.idk12loop = np.where(self.k12loop <= self.co.k)[0][0]
        #         k2pi12loop = np.array([self.kr**18, self.kr**20, self.kr**22, self.kr**24])
        #         self.k2p12loop = np.concatenate((k2pi12loop, k2pi12loop))
        #         self.alllpr12loop = np.array([
        #             [0, 1], [0, 1], [0, 1], [0, 1], 
        #             [0, 1], [0, 1], [0, 1], [0, 1]
        #         ])

        #     # For k's higher than ~ 0.25, we perform a 16-loop resummation.
        #     self.k16loop = 0.25
        #     if self.co.kmax > self.k16loop:
        #         # self.idk16loop = np.where(self.k16loop <= self.co.k)[0][0]
        #         k2pi16loop = np.array([self.kr**26, self.kr**28, self.kr**30, self.kr**32])
        #         self.k2p16loop = np.concatenate((k2pi16loop, k2pi16loop))
        #         self.alllpr16loop = np.array([
        #             [0, 1], [0, 1], [0, 1], [0, 1], 
        #             [0, 1], [0, 1], [0, 1], [0, 1]
        #         ])

        # # resummation with hexadecapole: we perform directly the 16-loop resummation, as the hexadecapole is mainly important for wedges for which we can go to pretty high k's
        # elif self.co.Nl is 3:
        #     if self.high: self.IRorder = 20
        #     else: self.IRorder = 16

        #     k2pi = np.array([self.kr**(2*(p+1)) for p in range(self.IRorder)])
        #     self.k2p = np.concatenate((k2pi, k2pi, k2pi))
        #     self.alllpr = np.array([[0, 1, 2] for p in range(self.IRorder*2)])

        # if self.co.optiresum is True:
        #     self.fftsettings = dict(Nmax=NFFT, xmin=.1, xmax=1000., bias=-0.6)
        # else:
        self.fftsettings = dict(Nmax=NFFT, xmin=.1, xmax=10000., bias=-0.6)
        self.fft = FFTLog(**self.fftsettings)
        self.setM()
        self.setkPow()

        self.Xfftsettings = dict(Nmax=32, xmin=1.5e-5, xmax=10., bias=-2.6)
        self.Xfft = FFTLog(**self.Xfftsettings)
        self.setXM()
        self.setXsPow()

        self.Cfftsettings = dict(Nmax=256, xmin=1.e-3, xmax=10., bias=-0.6)
        self.Cfft = FFTLog(**self.Cfftsettings)
        self.setMl()
        self.setsPow()

    def setXsPow(self):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the IR-filters X and Y. """
        self.XsPow = exp(np.einsum('n,s->ns', -self.Xfft.Pow - 3., log(self.sr)))

    def setXM(self):
        """ Compute the matrices to evaluate the IR-filters X and Y. Called at instantiation. """
        self.XM = np.empty(shape=(2, self.Xfft.Pow.shape[0]), dtype='complex')
        for l in range(2):
            self.XM[l] = MPC(2 * l, -0.5 * self.Xfft.Pow)

    def IRFilters(self, bird, soffset=1., LambdaIR=None, RescaleIR=1., window=None):
        """ Compute the IR-filters X and Y. """
        if LambdaIR is None:
            LambdaIR = self.LambdaIR
        Coef = self.Xfft.Coef(bird.kin, bird.Pin * exp(-bird.kin**2 / LambdaIR**2) / bird.kin**2, window=window)
        CoefsPow = np.einsum('n,ns->ns', Coef, self.XsPow)
        X02 = np.real(np.einsum('ns,ln->ls', CoefsPow, self.XM))
        X0offset = np.real(np.einsum('n,n->', np.einsum('n,n->n', Coef, soffset**(-self.Xfft.Pow - 3.)), self.XM[0]))
        X02[0] = X0offset - X02[0]
        X = RescaleIR * 2. / 3. * (X02[0] - X02[1])
        Y = 2. * X02[1]
        return X, Y

    def setkPow(self):
        """ Multiply the coefficients with the k's to the powers of the FFTLog to evaluate the IR-corrections. """
        self.kPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3., log(self.kr)))

    def setM(self, Nl=3):
        """ Compute the matrices to evaluate the IR-corrections. Called at instantiation. """
        self.M = np.empty(shape=(Nl, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(Nl):
            self.M[l] = 8. * pi**3 * MPC(2 * l, -0.5 * self.fft.Pow)

    def IRCorrection(self, XpYpC, k2p, lpr=None, window=None):
        """ Compute the IR-corrections of order n given [XY]^n and k^{2n} """
        Coef = self.fft.Coef(self.sr, XpYpC, extrap='padding', window=window)
        CoefkPow = np.einsum('n,nk->nk', Coef, self.kPow)
        return k2p * np.real(np.einsum('nk,ln->lk', CoefkPow, self.M[lpr]))

    def extractBAO(self, cf):
        """ Given a correlation function cf, 
            - if fullresum, return cf 
            - if optiresum, extract the BAO peak """
        if self.co.optiresum is True:
            cfnobao = np.concatenate([cf[..., :self.idlow], cf[..., self.idhigh:]], axis=-1)
            nobao = interp1d(self.snobao, self.snobao**2 * cfnobao, kind='linear', axis=-1)(self.sbao) * self.sbao**-2
            bao = cf[..., self.idlow:self.idhigh] - nobao
            return bao
        else:
            return cf

    def makeQ(self, f, which='8loop'):
        """ Compute the bulk coefficients Q^{ll'}_{||N-j}(n, \alpha, f) """
        if which is '8loop': NIR = 32
        elif which is '12loop': NIR = 16
        elif which is '16loop': NIR = 16
        elif which is 'withhex': NIR = 96
        elif which is 'withhex20': NIR = 120
        Q = np.empty(shape=(2, self.co.Nl, self.co.Nl, NIR))
        for a in range(2):
            for l in range(self.co.Nl):
                for lpr in range(self.co.Nl):
                    for u in range(NIR): 
                        if which is '8loop': Q[a][l][lpr][u] = Qa[1 - a][2 * l][2 * lpr][u](f)
                        elif which is '12loop': Q[a][l][lpr][u] = Qakp2x12[1 - a][2 * l][2 * lpr][u](f)
                        elif which is '16loop': Q[a][l][lpr][u] = Qakp2x16[1 - a][2 * l][2 * lpr][u](f)
                        elif which is 'withhex': Q[a][l][lpr][u] = Qawithhex[1 - a][2 * l][2 * lpr][u](f)
                        elif which is 'withhex20': Q[a][l][lpr][u] = Qawithhex20[1 - a][2 * l][2 * lpr][u](f)
        return Q

    def setMl(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation. """
        self.Ml = np.empty(shape=(self.co.Nl, self.Cfft.Pow.shape[0]), dtype='complex')
        for l in range(self.co.Nl):
            self.Ml[l] = MPC(2 * l, -0.5 * self.Cfft.Pow)

    def setsPow(self):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the IR corrections in configuration space. """
        self.sPow = exp(np.einsum('n,s->ns', -self.Cfft.Pow - 3., log(self.co.s)))

    def IRCorrPsCf(self, bird, XpYp, k2p, alllpr, Q, idkmin=0, idkmax=-1, window=None):
        """ Compute the IR corrections in configuration space by spherical Bessel transforming the IR corrections in Fourier space.  """
        self.IRCorrPs(bird, XpYp, k2p, alllpr, Q, idkmin=idkmin, idkmax=idkmax, window=window)

        DampingWindow = CoefWindow(self.co.Nk-1, window=0.6, left=False, right=True)

        if 'all' in bird.which:
            self.IRCf11 = np.zeros(shape=(self.co.Nl, self.co.N11, self.co.Ns))
            self.IRCfct = np.zeros(shape=(self.co.Nl, self.co.Nct, self.co.Ns))
            self.IRCfloop = np.zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Ns))
            for l, IRl in enumerate(self.IRPs11):
                for j, IRlj in enumerate(IRl):
                    Coef = 1j**(2*l) * self.Cfft.Coef(self.co.k, IRlj * DampingWindow, extrap='padding', window=None)
                    CoefsPow = np.einsum('n,ns->ns', Coef, self.sPow)
                    self.IRCf11[l,j] = np.real(np.einsum('ns,n->s', CoefsPow, self.Ml[l]))
            for l, IRl in enumerate(self.IRPsct):
                for j, IRlj in enumerate(IRl):
                    Coef = 1j**(2*l) * self.Cfft.Coef(self.co.k, IRlj * DampingWindow, extrap='padding', window=None)
                    CoefsPow = np.einsum('n,ns->ns', Coef, self.sPow)
                    self.IRCfct[l,j] = np.real(np.einsum('ns,n->s', CoefsPow, self.Ml[l]))
            for l, IRl in enumerate(self.IRPsloop):
                for j, IRlj in enumerate(IRl):
                    Coef = 1j**(2*l) * self.Cfft.Coef(self.co.k, IRlj * DampingWindow, extrap='padding', window=None)
                    CoefsPow = np.einsum('n,ns->ns', Coef, self.sPow)
                    self.IRCfloop[l,j] = np.real(np.einsum('ns,n->s', CoefsPow, self.Ml[l]))
        
        elif 'full' in bird.which:
            self.IRCf = np.zeros(shape=(2, self.co.Nl, self.co.Ns))
            for a, IRa in enumerate(self.IRPs):
                for l, IRal in enumerate(IRa):
                    Coef = 1j**(2*l) * self.Cfft.Coef(self.co.k, IRal * DampingWindow, extrap='padding', window=None)
                    CoefsPow = np.einsum('n,ns->ns', Coef, self.sPow)
                    self.IRCf[a,l] = np.real(np.einsum('ns,n->s', CoefsPow, self.Ml[l]))


    def PsCf(self, bird, window=None):

        Q = self.makeQ(bird.f, which='withhex')
        X, Y = self.IRFilters(bird)
        Xp = np.array([X**(p+1) for p in range(self.IRorder)])
        XpY = np.array([Y * X**p for p in range(self.IRorder)])
        XpYp = np.concatenate((Xp, XpY))

        self.IRCorrPsCf(bird, XpYp, self.k2p, self.alllpr, Q, idkmin=self.idklow, window=window)

        if 'all' in bird.which:
            bird.P11l += self.IRPs11
            bird.Pctl += self.IRPsct
            bird.Ploopl += self.IRPsloop

            bird.C11l += self.IRCf11
            bird.Cctl += self.IRCfct
            bird.Cloopl += self.IRCfloop

        if 'full' in bird.which:
            bird.Ps += self.IRPs
            bird.Cf += self.IRCf
            bird.setfullPs()
            bird.setfullCf()

    def Ps(self, bird, window=None):

        Q = self.makeQ(bird.f, which='withhex')
        X, Y = self.IRFilters(bird)
        Xp = np.array([X**(p+1) for p in range(self.IRorder)])
        XpY = np.array([Y * X**p for p in range(self.IRorder)])
        XpYp = np.concatenate((Xp, XpY))

        self.IRCorrPs(bird, XpYp, self.k2p, self.alllpr, Q, idkmin=self.idklow, window=window)

        if 'all' in bird.which:
            bird.P11l += self.IRPs11
            bird.Pctl += self.IRPsct
            bird.Ploopl += self.IRPsloop

        elif 'full' in bird.which:
            bird.Ps += self.IRPs
            bird.setfullPs()

        # if self.co.Nl is 2:
        #     Q = self.makeQ(bird.f, which='8loop')

        #     X, Y = self.IRFilters(bird)
        #     Xp = np.array([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8])
        #     XpY = np.array([Y, X * Y, X**2 * Y, X**3 * Y, X**4 * Y, X**5 * Y, X**6 * Y, X**7 * Y])
        #     XpYp = np.concatenate((Xp, XpY))
            
        #     self.IRCorrections(bird, XpYp, self.k2p, self.alllpr, Q, idkmin=self.idklow, window=window)

        #     if self.co.kmax > self.k12loop:
        #         Q12loop = self.makeQ(bird.f, which='12loop')
        #         Xp12loop = np.array([X**9, X**10, X**11, X**12])
        #         XpY12loop = np.array([X**8 * Y, X**9 * Y, X**10 * Y, X**11 * Y])
        #         XpYp12loop = np.concatenate((Xp12loop, XpY12loop))
        #         self.IRCorrections(bird, XpY12loop, self.k2p12loop, self.alllpr12loop, Q12loop, idkmin=self.idklow, window=window)

        #     if self.co.kmax > self.k16loop:
        #         Q16loop = self.makeQ(bird.f, which='16loop')
        #         Xp16loop = np.array([X**13, X**14, X**15, X**16])
        #         XpY16loop = np.array([X**12 * Y, X**13 * Y, X**14 * Y, X**15 * Y])
        #         XpYp16loop = np.concatenate((Xp16loop, XpY16loop))
        #         self.IRCorrections(bird, XpYp16loop, self.k2p16loop, self.alllpr16loop, Q16loop, idkmin=self.idklow, window=window)
        
        # elif self.co.Nl is 3:
        #     if self.IRorder is 16: Q = self.makeQ(bird.f, which='withhex')
        #     elif self.IRorder is 20: Q = self.makeQ(bird.f, which='withhex20')

        #     X, Y = self.IRFilters(bird)
        #     Xp = np.array([X**(p+1) for p in range(self.IRorder)])
        #     XpY = np.array([Y * X**p for p in range(self.IRorder)])
        #     XpYp = np.concatenate((Xp, XpY, XpY))
            
        #     self.IRCorrections(bird, XpYp, self.k2p, self.alllpr, Q, idkmin=self.idklow, window=window)


    def IRCorrPs(self, bird, XpYp, k2p, alllpr, Q, idkmin=0, idkmax=-1, window=None):
        """ This is the main method of the class. Compute the IR corrections in Fourier space. """
        NIR = alllpr.size

        if 'all' in bird.which:
            self.IR11 = np.zeros(shape=(self.co.Nl, NIR, self.Nkr))
            self.IRct = np.zeros(shape=(self.co.Nl, NIR, self.Nkr))
            self.IRloop = np.zeros(shape=(self.co.Nl, self.co.Nloop, NIR, self.Nkr))
            self.IRPs11 = np.zeros(shape=(self.co.Nl, self.co.N11, self.co.Nk))
            self.IRPsct = np.zeros(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
            self.IRPsloop = np.zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nk))
            for l, cl in enumerate(self.extractBAO(bird.C11)):
                u = 0
                for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
                    IRcorrUnsorted = np.real(1j**(2*l)) * self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
                    for v in range(len(lpr)):
                        self.IR11[l, u + v] = IRcorrUnsorted[v]
                    u += len(lpr)
            for l, cl in enumerate(self.extractBAO(bird.Cct)):
                u = 0
                for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
                    IRcorrUnsorted = np.real(1j**(2*l)) * self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
                    for v in range(len(lpr)):
                        self.IRct[l, u + v] = IRcorrUnsorted[v]
                    u += len(lpr)
            for l, cl in enumerate(self.extractBAO(bird.Cloopl)):
                for i, cli in enumerate(cl):
                    u = 0
                    for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
                        IRcorrUnsorted = np.real(1j**(2*l)) * self.IRCorrection(xy * cli, k2pj, lpr=lpr, window=window)
                        for v in range(len(lpr)):
                            self.IRloop[l, i, u + v] = IRcorrUnsorted[v]
                        u += len(lpr)
            self.IRPs11[..., idkmin:] = np.einsum('lpn,pnk,pi->lik', Q[0], self.IR11, self.co.l11)
            self.IRPsct[..., idkmin:] = np.einsum('lpn,pnk,pi->lik', Q[1], self.IRct, self.co.lct)
            self.IRPsloop[..., idkmin:] = np.einsum('lpn,pink->lik', Q[1], self.IRloop)
            #return self.IR11resum, self.IRctresum, self.IRloopresum

        elif 'full' in bird.which:
            self.IRcorr = np.zeros(shape=(2, self.co.Nl, NIR, self.Nkr))
            self.IRPs = np.zeros(shape=(2, self.co.Nl, self.co.Nk))
            for a, cf in enumerate(self.extractBAO(bird.Cf)):
                for l, cl in enumerate(cf):
                    u = 0
                    for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
                        IRcorrUnsorted = np.real(1j**(2*l)) * self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
                        for v in range(len(lpr)):
                            self.IRcorr[a, l, u + v] = IRcorrUnsorted[v]
                        u += len(lpr)
            self.IRPs[..., idkmin:] = np.einsum('alpn,apnk->alk', Q, self.IRcorr)
            #return self.IRresum

        ### DEPRECIATED
        # if bird.which is 'marg':
        #     self.IRcorr = np.empty(shape=(2, self.co.Nl, NIR, self.Nkr))
        #     self.IRb3 = np.empty(shape=(self.co.Nl, NIR, self.Nkr))
        #     self.IRct = np.empty(shape=(self.co.Nl, NIR, self.Nkr))
        #     self.IRresum = np.zeros(shape=(2, self.co.Nl, self.co.Nk))
        #     self.IRb3resum = np.zeros(shape=(self.co.Nl, self.co.Nk))
        #     self.IRctresum = np.zeros(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
        #     for a, cf in enumerate(self.extractBAO(bird.Cf)):
        #         for l, cl in enumerate(cf):
        #             u = 0
        #             for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
        #                 IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
        #                 for v in range(len(lpr)):
        #                     self.IRcorr[a, l, u + v] = IRcorrUnsorted[v]
        #                 u += len(lpr)
        #     for l, cl in enumerate(self.extractBAO(bird.Cct)):
        #         u = 0
        #         for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
        #             IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
        #             for v in range(len(lpr)):
        #                 self.IRct[l, u + v] = IRcorrUnsorted[v]
        #             u += len(lpr)
        #     for l, cl in enumerate(self.extractBAO(bird.Cb3)):
        #         u = 0
        #         for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
        #             IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
        #             for v in range(len(lpr)):
        #                 self.IRb3[l, u + v] = IRcorrUnsorted[v]
        #             u += len(lpr)
        #     self.IRresum[..., idkmin:] = np.einsum('alpn,apnk->alk', Q, self.IRcorr)
        #     self.IRctresum[..., idkmin:] = np.einsum('lpn,pnk,pi->lik', Q[1], self.IRct, self.co.lct)
        #     self.IRb3resum[..., idkmin:] = np.einsum('lpn,pnk->lk', Q[1], self.IRb3)
        #     bird.Ps += self.IRresum
        #     bird.Pctl += self.IRctresum
        #     bird.Pb3 += self.IRb3resum
        #     bird.setfullPs()

    # def Ps(self, bird, XpYp, k2p, alllpr, NIRmin, window=None):
    #     """ This is the main method of the class. Compute the IR-corrections and add them in the power spectrum. """
    #     self.makeQ(bird.f)

    #     X, Y = self.IRFilters(bird)
    #     Xp = np.array([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8])
    #     XpY = np.array([Y, X * Y, X**2 * Y, X**3 * Y, X**4 * Y, X**5 * Y, X**6 * Y, X**7 * Y])

    #     XpYp = np.concatenate((Xp, XpY))

    #     if bird.which is 'marg':
    #         for a, cf in enumerate(self.extractBAO(bird.Cf)):
    #             for l, cl in enumerate(cf):
    #                 u = 0
    #                 for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
    #                     IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
    #                     for v in range(len(lpr)):
    #                         self.IRcorr[a, l, u + v] = IRcorrUnsorted[v]
    #                     u += len(lpr)
    #         for l, cl in enumerate(self.extractBAO(bird.Cct)):
    #             u = 0
    #             for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
    #                 IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
    #                 for v in range(len(lpr)):
    #                     self.IRct[l, u + v] = IRcorrUnsorted[v]
    #                 u += len(lpr)
    #         for l, cl in enumerate(self.extractBAO(bird.Cb3)):
    #             u = 0
    #             for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
    #                 IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
    #                 for v in range(len(lpr)):
    #                     self.IRb3[l, u + v] = IRcorrUnsorted[v]
    #                 u += len(lpr)
    #         self.IRresum[..., IRmin:] = np.einsum('alpn,apnk->alk', self.Q, self.IRcorr)
    #         self.IRctresum[..., IRmin:] = np.einsum('lpn,pnk,pi->lik', self.Q[1], self.IRct, self.co.lct)
    #         self.IRb3resum[..., IRmin:] = np.einsum('lpn,pnk->lk', self.Q[1], self.IRb3)
    #         bird.Ps += self.IRresum
    #         bird.Pctl += self.IRctresum
    #         bird.Pb3 += self.IRb3resum
    #         bird.setfullPs()

    #     elif bird.which is 'all':
    #         for l, cl in enumerate(self.extractBAO(bird.C11)):
    #             u = 0
    #             for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
    #                 IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
    #                 for v in range(len(lpr)):
    #                     self.IR11[l, u + v] = IRcorrUnsorted[v]
    #                 u += len(lpr)
    #         for l, cl in enumerate(self.extractBAO(bird.Cct)):
    #             u = 0
    #             for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
    #                 IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
    #                 for v in range(len(lpr)):
    #                     self.IRct[l, u + v] = IRcorrUnsorted[v]
    #                 u += len(lpr)
    #         for l, cl in enumerate(self.extractBAO(bird.Cloopl)):
    #             for i, cli in enumerate(cl):
    #                 u = 0
    #                 for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
    #                     IRcorrUnsorted = self.IRCorrection(xy * cli, k2pj, lpr=lpr, window=window)
    #                     for v in range(len(lpr)):
    #                         self.IRloop[l, i, u + v] = IRcorrUnsorted[v]
    #                     u += len(lpr)
    #         self.IR11resum[..., IRmin:] = np.einsum('lpn,pnk,pi->lik', self.Q[0], self.IR11, self.co.l11)
    #         self.IRctresum[..., IRmin:] = np.einsum('lpn,pnk,pi->lik', self.Q[1], self.IRct, self.co.lct)
    #         self.IRloopresum[..., IRmin:] = np.einsum('lpn,pink->lik', self.Q[1], self.IRloop)
    #         bird.P11l += self.IR11resum
    #         bird.Pctl += self.IRctresum
    #         bird.Ploopl += self.IRloopresum
    #         bird.subtractShotNoise()

    #     elif 'full' in bird.which:
    #         for a, cf in enumerate(self.extractBAO(bird.Cf)):
    #             for l, cl in enumerate(cf):
    #                 u = 0
    #                 for j, (xy, k2pj, lpr) in enumerate(zip(XpYp, k2p, self.alllpr)):
    #                     IRcorrUnsorted = self.IRCorrection(xy * cl, k2pj, lpr=lpr, window=window)
    #                     for v in range(len(lpr)):
    #                         self.IRcorr[a, l, u + v] = IRcorrUnsorted[v]
    #                     u += len(lpr)
    #         self.IRresum[..., IRmin:] = np.einsum('alpn,apnk->alk', self.Q, self.IRcorr)
    #         bird.Ps += self.IRresum
    #         bird.setfullPs()

class Projection(object):
    """
    A class to apply projection effects:
    - Alcock-Pascynski (AP) effect
    - Window functions (survey masks)
    - k-binning or interpolation over the data k-array
    - Fiber collision corrections
    - Wedges
    """
    def __init__(self, kout, Om_AP, z_AP, nbinsmu=200, 
        window_fourier_name=None, path_to_window=None, window_configspace_file=None, 
        binning=False, fibcol=False, Nwedges=0, cf=False, co=common):

        self.co = co
        self.cf = cf
        self.kout = kout

        self.Om = Om_AP
        self.z = z_AP

        self.DA = DA(self.Om, self.z)
        self.H = Hubble(self.Om, self.z)

        self.muacc = np.linspace(0., 1., nbinsmu)
        if self.cf: self.sgrid, self.mugrid = np.meshgrid(self.co.s, self.muacc, indexing='ij')
        else: self.kgrid, self.mugrid = np.meshgrid(self.co.k, self.muacc, indexing='ij')
        self.arrayLegendremugrid = np.array([(2*2*l+1)/2.*legendre(2*l)(self.mugrid) for l in range(self.co.Nl)])


        if window_configspace_file is not None: 
            if window_fourier_name is not None:
                self.path_to_window = path_to_window
                self.window_fourier_name = window_fourier_name
            self.window_configspace_file = window_configspace_file
            self.setWindow(Nl=self.co.Nl)

        if binning:
            self.loadBinning(self.kout)

        # wedges
        if Nwedges is not 0:
            self.Nw = Nwedges
            self.IL = self.IntegralLegendreArray(Nw=self.Nw, Nl=self.co.Nl)

    def get_AP_param(self, bird):
        """
        Compute the AP parameters
        """
        qperp = bird.DA / self.DA
        qpar = self.H / bird.H
        return qperp, qpar

    def integrAP(self, k, Pk, kp, arrayLegendremup, many=False):
        """
        AP integration
        Credit: Jerome Gleyzes
        """
        Pkint = interp1d(k, Pk, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')
        if many:
            Pkmu = np.einsum('lpkm,lkm->pkm', Pkint(kp), arrayLegendremup)
            Integrandmu = np.einsum('pkm,lkm->lpkm', Pkmu, self.arrayLegendremugrid)
        else:
            Pkmu = np.einsum('lkm,lkm->km', Pkint(kp), arrayLegendremup)
            Integrandmu = np.einsum('km,lkm->lkm', Pkmu, self.arrayLegendremugrid)
        return 2 * np.trapz(Integrandmu, x=self.mugrid, axis=-1)

    def AP(self, bird=None, q=None):
        """
        Apply the AP effect to the bird power spectrum or correlation function
        Credit: Jerome Gleyzes
            """
        if q is None: qperp, qpar = self.get_AP_param(bird)
        else: qperp, qpar = q

        if self.cf:
            G = (self.mugrid**2 * qpar**2 + (1-self.mugrid**2) * qperp**2)**0.5
            sp = self.sgrid * G
            mup = self.mugrid * qpar / G
            arrayLegendremup = np.array([legendre(2*l)(mup) for l in range(self.co.Nl)])

            if 'all' in bird.which:
                bird.C11l = self.integrAP(self.co.s, bird.C11l, sp, arrayLegendremup, many=True)
                bird.Cctl = self.integrAP(self.co.s, bird.Cctl, sp, arrayLegendremup, many=True)
                bird.Cloopl = self.integrAP(self.co.s, bird.Cloopl, sp, arrayLegendremup, many=True)
            
            elif 'full' in bird.which:
                bird.fullCf = self.integrAP(self.co.s, bird.fullCf, sp, arrayLegendremup, many=False)
        
        else:
            F = qpar / qperp
            kp = self.kgrid / qperp * (1 + self.mugrid**2 * (F**-2 - 1))**0.5
            mup = self.mugrid / F * (1 + self.mugrid**2 * (F**-2 - 1))**-0.5
            arrayLegendremup = np.array([legendre(2*l)(mup) for l in range(self.co.Nl)])

            if bird.which is 'marg':
                bird.fullPs = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.fullPs, kp, arrayLegendremup, many=False)
                bird.Pb3 = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Pb3, kp, arrayLegendremup, many=False)
                bird.Pctl = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Pctl, kp, arrayLegendremup, many=True)

            elif 'all' in bird.which:
                bird.P11l = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.P11l, kp, arrayLegendremup, many=True)
                bird.Pctl = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Pctl, kp, arrayLegendremup, many=True)
                bird.Ploopl = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Ploopl, kp, arrayLegendremup, many=True)

            elif 'full' in bird.which:
                bird.fullPs = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.fullPs, kp, arrayLegendremup, many=False)


    def setWindow(self, load=True, save=True, Nl=3, withmask=True, windowk=0.05):
        """
        Pre-load the window function to apply to the power spectrum by convolution in Fourier space
        Wal is an array of shape l,l',k',k where so that $P_a(k) = \int dk' \sum_l W_{a l}(k,k') P_l(k')$
        If it cannot find the file, it will compute a new one from a provided mask in configuration space.

        Inputs
        ------
        withmask: whether to only do the convolution over a small range around k
        windowk: the size of said range
        """

        compute = True

        if self.cf:
            try:
                swindow_config_space = np.loadtxt(self.window_configspace_file)
            except:
                print ('Error: can\'t load mask file: %s.'%self.window_configspace_file)
                compute = False

        else:
            self.p = np.concatenate([ np.geomspace(1e-5, 0.015, 100, endpoint=False) , np.arange(0.015, self.co.kmax, 1e-3) ])
            window_fourier_file = os.path.join(self.path_to_window, '%s_Nl%s_kmax%.2f.npy') % (self.window_fourier_name, self.co.Nl, self.co.kmax)
            
            if load:
                try:
                    self.Wal = np.load(window_fourier_file)
                    print ('Loaded mask: %s' % window_fourier_file)
                    save = False
                    compute = False
                except:
                    print ('Can\'t load mask: %s \n instead,' % window_fourier_file )
                    load = False

            if not load: # do not change to else
                print ('Computing new mask.')
                if self.window_configspace_file is None:
                    print ('Error: please specify a configuration-space mask file.')
                    compute = False
                else:    
                    try:
                        swindow_config_space = np.loadtxt(self.window_configspace_file)
                    except:
                        print ('Error: can\'t load mask file: %s.'%self.window_configspace_file)
                        compute = False
        
        if compute is True:
            Calp = np.array([ 
                [ [1., 0., 0.],
                  [0., 1/5., 0.],
                  [0., 0., 1/9.] ],
                [ [0., 1., 0.],
                  [1., 2/7., 2/7.],
                  [0., 2/7., 100/693.] ],
                [ [0., 0., 1.],
                  [0., 18/35., 20/77.],
                  [1., 20/77., 162/1001.] ],
                ])

            sw = swindow_config_space[:,0]
            Qp = np.moveaxis(swindow_config_space[:,1:].reshape(-1,3), 0, -1 )[:Nl]
            Qal = np.einsum('alp,ps->als', Calp[:Nl,:Nl,:Nl], Qp)

            if self.cf: 
                self.Qal = interp1d(sw, Qal, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(self.co.s)

            else:
                self.fftsettings = dict(Nmax=4096, xmin=sw[0], xmax=sw[-1]*100., bias=-1.6) # 1e-2 - 1e6 [Mpc/h]
                self.fft = FFTLog(**self.fftsettings)
                self.pPow = exp(np.einsum('n,s->ns', -self.fft.Pow-3., log(self.p)))
                self.M = np.empty(shape=(Nl, self.fft.Pow.shape[0]), dtype='complex')
                for l in range(Nl): self.M[l] = 4*pi * MPC(2*l, -0.5*self.fft.Pow)

                self.Coef = np.empty(shape=(self.co.Nl, Nl, self.co.Nk, self.fft.Pow.shape[0]), dtype='complex')
                for a in range(self.co.Nl):
                    for l in range(Nl):
                        for i,k in enumerate(self.co.k):
                            self.Coef[a,l,i] = (-1j)**(2*a) * 1j**(2*l) * self.fft.Coef(sw, Qal[a,l]*spherical_jn(2*a, k*sw), extrap = 'padding')

                self.Wal = self.p**2 * np.real( np.einsum('alkn,np,ln->alkp', self.Coef, self.pPow, self.M) )

                if save: 
                    print ( 'Saving mask: %s' % window_fourier_file )
                    np.save(window_fourier_file, self.Wal)

        if not self.cf:
            self.Wal = self.Wal[:,:self.co.Nl]

            # Apply masking centered around the value of k
            if withmask:
                kpgrid, kgrid = np.meshgrid(self.p, self.co.k, indexing='ij')
                mask = (kpgrid < kgrid + windowk) & (kpgrid > kgrid - windowk)
                Wal_masked = np.einsum('alkp,pk->alkp', self.Wal, mask)

            # the spacing (needed to do the convolution as a sum)
            deltap = self.p[1:] - self.p[:-1]
            deltap = np.concatenate([[0], deltap])
            self.Waldk = np.einsum('alkp,p->alkp', Wal_masked, deltap)

    def integrWindow(self, P, many=False):
        """
        Convolve the window functions to a power spectrum P
        """
        Pk = interp1d(self.co.k, P, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(self.p)
        # (multipole l, multipole ' p, k, k' m) , (multipole ', power pectra s, k' m)
        #print (self.Qlldk.shape, Pk.shape)
        if many:
            return np.einsum('alkp,lsp->ask', self.Waldk, Pk)
        else:
            return np.einsum('alkp,lp->ak', self.Waldk, Pk)

    def Window(self, bird):
        """
        Apply the survey window function to the bird power spectrum 
        """
        if self.cf:
            if 'all' in bird.which:
                bird.C11l = np.einsum('als,lns->ans', self.Qal, bird.C11l)
                bird.Cctl = np.einsum('als,lns->ans', self.Qal, bird.Cctl)
                bird.Cloopl = np.einsum('als,lns->ans', self.Qal, bird.Cloopl)

            elif 'full' in bird.which:
                bird.fullCf = np.einsum('als,ls->as', self.Qal, bird.fullCf)

        else:
            if bird.which is 'marg':
                bird.fullPs = self.integrWindow(bird.fullPs, many=False)
                bird.Pb3 = self.integrWindow(bird.Pb3, many=False)
                bird.Pctl = self.integrWindow(bird.Pctl, many=True)

            elif 'all' in bird.which:
                bird.P11l = self.integrWindow(bird.P11l, many=True)
                bird.Pctl = self.integrWindow(bird.Pctl, many=True)
                bird.Ploopl = self.integrWindow(bird.Ploopl, many=True)

            elif 'full' in bird.which:
                bird.fullPs = self.integrWindow(bird.fullPs, many=False)

    def dPuncorr(self, kout, fs=0.6, Dfc=0.43 / 0.6777):
        """
        Compute the uncorrelated contribution of fiber collisions

        kPS : a cbird wavenumber output, typically a (39,) np array
        fs : fraction of the survey affected by fiber collisions
        Dfc : angular distance of the fiber channel Dfc(z = 0.55) = 0.43Mpc

        Credit: Thomas Colas
        """
        dPunc = np.zeros((3, len(kout)))
        for l in [0, 2, 4]:
            dPunc[int(l / 2)] = - fs * pi * Dfc**2. * (2. * pi / kout) * (2. * l + 1.) / \
                2. * special.legendre(l)(0) * (1. - (kout * Dfc)**2 / 8.)
        return dPunc

    def dPcorr(self, kout, kPS, PS, many=False, ktrust=0.25, fs=0.6, Dfc=0.43 / 0.6777):
        """
        Compute the correlated contribution of fiber collisions

        kPS : a cbird wavenumber output, typically a (39,) np array
        PS : a cbird power spectrum output, typically a (3, 39) np array
        ktrust : a UV cutoff
        fs : fraction of the survey affected by fiber collisions
        Dfc : angular distance of the fiber channel Dfc(z = 0.55) = 0.43Mpc

        Credit: Thomas Colas
        """
        q_ref = np.geomspace(min(kPS), ktrust, num=1024)
        # create log bin
        dq_ref = q_ref[1:] - q_ref[:-1]
        dq_ref = np.concatenate([[0], dq_ref])

        PS_interp = interp1d(kPS, PS, axis=-1, bounds_error=False, fill_value='extrapolate')(q_ref)

        if many:
            dPcorr = np.zeros(shape=(PS.shape[0], PS.shape[1], len(kout)))
            for j in range(PS.shape[1]):
                for l in range(self.co.Nl):
                    for lp in range(self.co.Nl):
                        for i, k in enumerate(kout):
                            if lp <= l:
                                maskIR = (q_ref < k)
                                dPcorr[l, j, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskIR],
                                                                                   dq_ref[maskIR], PS_interp[lp, j, maskIR], fllp_IR(2 * l, 2 * lp, k, q_ref[maskIR], Dfc))
                            if lp >= l:
                                maskUV = ((q_ref > k) & (q_ref < ktrust))
                                dPcorr[l, j, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskUV],
                                                                                   dq_ref[maskUV], PS_interp[lp, j, maskUV], fllp_UV(2 * l, 2 * lp, k, q_ref[maskUV], Dfc))
        else:
            dPcorr = np.zeros(shape=(PS.shape[0], len(kout)))
            for l in range(self.co.Nl):
                for lp in range(self.co.Nl):
                    for i, k in enumerate(kout):
                        if lp <= l:
                            maskIR = (q_ref < k)
                            dPcorr[l, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskIR],
                                                                            dq_ref[maskIR], PS_interp[lp, maskIR], fllp_IR(2 * l, 2 * lp, k, q_ref[maskIR], Dfc))
                        if lp >= l:
                            maskUV = ((q_ref > k) & (q_ref < ktrust))
                            dPcorr[l, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskUV],
                                                                            dq_ref[maskUV], PS_interp[lp, maskUV], fllp_UV(2 * l, 2 * lp, k, q_ref[maskUV], Dfc))
        return dPcorr

    def fibcolWindow(self, bird):
        """
        Apply window effective method correction to fiber collisions to the bird power spectrum
        """
        if 'all' in bird.which:
            bird.P11l += self.dPcorr(self.co.k, self.co.k, bird.P11l, many=True)
            bird.Pctl += self.dPcorr(self.co.k, self.co.k, bird.Pctl, many=True)
            bird.Ploopl += self.dPcorr(self.co.k, self.co.k, bird.Ploopl, many=True)

    def loadBinning(self, setkout):
        """
        Create the bins of the data k's
        """
        delta_k = np.round(setkout[-1] - setkout[-2], 2)
        kcentral = (setkout[-1] - delta_k * np.arange(len(setkout)))[::-1]
        binmin = kcentral - delta_k / 2
        binmax = kcentral + delta_k / 2
        self.binvol = np.array([quad(lambda k: k**2, kbinmin, kbinmax)[0]
                                for (kbinmin, kbinmax) in zip(binmin, binmax)])

        self.points = [np.linspace(kbinmin, kbinmax, 100) for (kbinmin, kbinmax) in zip(binmin, binmax)]

    def integrBinning(self, P):
        """
        Integrate over each bin of the data k's
        """
        Pkint = interp1d(self.co.k, P, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')
        res = np.array([np.trapz(Pkint(pts) * pts**2, x=pts) for pts in self.points])
        return np.moveaxis(res, 0, -1) / self.binvol

    def kbinning(self, bird):
        """
        Apply binning in k-space for linear-spaced data k-array
        """
        if 'all' in bird.which:
            bird.P11l = self.integrBinning(bird.P11l)
            bird.Pctl = self.integrBinning(bird.Pctl)
            bird.Ploopl = self.integrBinning(bird.Ploopl)

    def kdata(self, bird):
        """
        Interpolate the bird power spectrum on the data k-array
        """
        if self.cf:
            if 'all' in bird.which:
                bird.C11l = interp1d(self.co.s, bird.C11l, axis=-1, kind='cubic', bounds_error=False)(self.kout)
                bird.Cctl = interp1d(self.co.s, bird.Cctl, axis=-1, kind='cubic', bounds_error=False)(self.kout)
                bird.Cloopl = interp1d(self.co.s, bird.Cloopl, axis=-1, kind='cubic', bounds_error=False)(self.kout)
            if 'full' in bird.which:
                bird.fullCf = interp1d(self.co.s, bird.fullCf, axis=-1, kind='cubic', bounds_error=False)(self.kout)
        else:
            if 'all' in bird.which:
                bird.P11l = interp1d(self.co.k, bird.P11l, axis=-1, kind='cubic', bounds_error=False)(self.kout)
                bird.Pctl = interp1d(self.co.k, bird.Pctl, axis=-1, kind='cubic', bounds_error=False)(self.kout)
                bird.Ploopl = interp1d(self.co.k, bird.Ploopl, axis=-1, kind='cubic', bounds_error=False)(self.kout)
            if 'full' in bird.which:
                bird.fullPs = interp1d(self.co.k, bird.fullPs, axis=-1, kind='cubic', bounds_error=False)(self.kout)

    def IntegralLegendre(self, l, a, b):
        if l == 0: return 1.
        if l == 2: return 0.5*(b**3-b-a**3+a)/(b-a)
        if l == 4: return 0.25*(-3/2.*a+5*a**3-7/2.*a**5+3/2.*b-5*b**3+7/2.*b**5)/(b-a)

    def IntegralLegendreArray(self, Nw=3, Nl=2):
        deltamu = 1./float(Nw)
        boundsmu = np.arange(0., 1., deltamu)
        IntegrLegendreArray = np.empty(shape=(Nw, Nl))
        for w, boundmu in enumerate(boundsmu):
            for l in range(Nl):
                IntegrLegendreArray[w,l] = self.IntegralLegendre(2*l, boundmu, boundmu+deltamu)
        return IntegrLegendreArray

    def integrWedges(self, P, many=False):
        if many: w = np.einsum('lpk,wl->wpk', P, self.IL)
        else: w = np.einsum('lk,wl->wk', P, self.IL)
        return w

    def Wedges(self, bird):
        """
        Produce wedges
        """
        if 'all' in bird.which:
            bird.P11l = self.integrWedges(bird.P11l, many=True)
            bird.Pctl = self.integrWedges(bird.Pctl, many=True)
            bird.Ploopl = self.integrWedges(bird.Ploopl, many=True)

        elif 'full' in bird.which:
            bird.fullPs = self.integrWedges(bird.fullPs, many=False)

    def Wedges_external(self, P):
        return self.integrWedges(P, many=False)


class Angular(object):
    """
    A class to ...
    """
    def __init__(self, theta=None, ellmax=1000, co=common, legendre_elltheta=None, NFFT=512):

        self.co = co
        self.ellmax = ellmax
        self.ell = np.arange(1, self.ellmax+1)
        self.mu = np.linspace(0., 1., 100)
        self.ellij, self.muij = np.meshgrid(self.ell, self.mu, indexing='ij')
        self.arrayLegendremu = np.array([legendre(2*l)(self.muij) for l in range(self.co.Nl)])

        if theta is not None: 
            self.theta = theta
            self.fftsettings = dict(Nmax=NFFT, xmin=.1, xmax=1000., bias=-0.6)
            self.fft = FFTLog(**self.fftsettings)
            self.setM()
            self.setthetaPow()

            if legendre_elltheta is None: self.Legendre_elltheta = np.array([legendre(l)(cos(theta)) for l in self.ell])
            else: self.Legendre_elltheta = legendre_elltheta

    def integrCell(self, k, Pk, nz_sigma, kpara, knorm, H, chi, many=False):
        Pkint = interp1d(k, Pk, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(knorm)
        if many: Integrandmu = np.einsum('lm,aplm,alm->plm', knorm * exp( -(nz_sigma*kpara/H)**2 ), Pkint, self.arrayLegendremu)
        else: Integrandmu = np.einsum('lm,alm,alm->lm', knorm * exp( -(nz_sigma*kpara/H)**2 ), Pkint, self.arrayLegendremu)
        return sqrt(2)/pi/chi**2 * trapz(Integrandmu, x=self.muij, axis=-1)
    
    def Cell(self, bird, nz_sigma=0.05):
        if bird.ellmax > 1000: bird.ellmax = 1000 ### TO ERASE

        kperp = self.ellij/bird.chi
        knorm = kperp * sqrt(1-self.muij**2)
        kpara = knorm * self.muij
        
        if bird.which is 'angular_full':
            bird.Cell = self.integrCell(self.co.k, bird.fullPs, nz_sigma, kpara, knorm, bird.H, bird.chi)
        elif bird.which is 'angular_all':
            bird.Cell11 = self.integrCell(self.co.k, bird.P11l, nz_sigma, kpara, knorm, bird.H, bird.chi, many=True)
            bird.Cellct = self.integrCell(self.co.k, bird.Pctl, nz_sigma, kpara, knorm, bird.H, bird.chi, many=True)
            bird.Cellloop = self.integrCell(self.co.k, bird.Ploopl, nz_sigma, kpara, knorm, bird.H, bird.chi, many=True)
    
    def setthetaPow(self):
        """ Multiply the coefficients with the ell's to the powers of the FFTLog to . """
        self.thetaPow = exp(np.einsum('n,t->nt', -self.fft.Pow - 3., log(self.theta)))

    def setM(self, Nl=3):
        """ Compute the matrices to transform the angular power spectrum to angular correlation function. Called at instantiation. """
        #self.M = np.empty(shape=(self.fft.Pow.shape[0]), dtype='complex')
        self.M = (2*pi)**.5 * MPC(-0.5, -0.5 * self.fft.Pow)

    def w(self, bird, nz_sigma=0.05, window=None):
        """ Compute the  """
        self.Cell(bird, nz_sigma)
        self.cutell(bird)

        self.fftsettings = dict(Nmax=512, xmin=.1, xmax=bird.ellmax*3., bias=-0.6)
        self.fft = FFTLog(**self.fftsettings)

        if bird.which is 'angular_full':
            Coef = self.fft.Coef(self.ell[:bird.ellmax], self.ell[:bird.ellmax]**-0.5 * bird.Cell, extrap='extrap', window=window)
            CoefthetaPow = np.einsum('n,nl->nl', Coef, self.thetaPow)
            bird.w = self.theta**0.5 * np.real(np.einsum('nl,n->l', CoefthetaPow, self.M))
        elif bird.which is 'angular_all':
            for i in range(self.co.N11):
                print ('lin %s'%i)
                Coef = self.fft.Coef(self.ell[:bird.ellmax], self.ell[:bird.ellmax]**-0.5 * bird.Cell11[i], extrap='extrap', window=window)
                CoefthetaPow = np.einsum('n,nl->nl', Coef, self.thetaPow)
                bird.w11[i] = self.theta**0.5 * np.real(np.einsum('nl,n->l', CoefthetaPow, self.M))
            for i in range(self.co.Nct):
                print ('ct %s'%i)
                Coef = self.fft.Coef(self.ell[:bird.ellmax], self.ell[:bird.ellmax]**-0.5 * bird.Cellct[i], extrap='extrap', window=window)
                CoefthetaPow = np.einsum('n,nl->nl', Coef, self.thetaPow)
                bird.wct[i] = self.theta**0.5 * np.real(np.einsum('nl,n->l', CoefthetaPow, self.M))
            for i in range(self.co.Nloop):
                print ('loop %s'%i)
                Coef = self.fft.Coef(self.ell[:bird.ellmax], self.ell[:bird.ellmax]**-0.5 * bird.Cellloop[i], extrap='extrap', window=window)
                CoefthetaPow = np.einsum('n,nl->nl', Coef, self.thetaPow)
                bird.wloop[i] = self.theta**0.5 * np.real(np.einsum('nl,n->l', CoefthetaPow, self.M))

    def w_sum(self, bird, nz_sigma=0.05):
        self.Cell(bird, nz_sigma)
        self.cutell(bird)
        if bird.which is 'angular_full':
            bird.w = np.einsum('lt,l->t', self.Legendre_elltheta[:bird.ellmax], (2*self.ell[:bird.ellmax]+1)/(4*pi)*bird.Cell)
        elif bird.which is 'angular_all':
            bird.w11 = np.einsum('lt,nl->nt', self.Legendre_elltheta[:bird.ellmax], (2*self.ell[:bird.ellmax]+1)/(4*pi)*bird.Cell11)
            bird.wct = np.einsum('lt,nl->nt', self.Legendre_elltheta[:bird.ellmax], (2*self.ell[:bird.ellmax]+1)/(4*pi)*bird.Cellct)
            bird.wloop = np.einsum('lt,nl->nt', self.Legendre_elltheta[:bird.ellmax], (2*self.ell[:bird.ellmax]+1)/(4*pi)*bird.Cellloop)

    def cutell(self, bird):
        if bird.which is 'angular_full':
            bird.Cell = bird.Cell[:bird.ellmax]
        elif bird.which is 'angular_all':
            bird.Cell11 = bird.Cell11[:, :bird.ellmax]
            bird.Cellct = bird.Cellct[:, :bird.ellmax]
            bird.Cellloop = bird.Cellloop[:, :bird.ellmax]

    def Cell_external(self, k, Pk, H, chi, nz_sigma=0.05):
        kperp = self.ellij/chi
        knorm = kperp * sqrt(1-self.muij**2)
        kpara = knorm * self.muij
        return self.integrCell(k, Pk, nz_sigma, kpara, knorm, H, chi, many=False)

    def w_external(self, k, Pk, H, chi, nz_sigma=0.05):
        Cell_ext = self.Cell_external(k, Pk, H, chi, nz_sigma)
        return np.einsum('lt,l->t', self.Legendre_elltheta, (2*self.ell+1)/(4*pi)*Cell_ext)
